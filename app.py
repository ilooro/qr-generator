import torch
import gradio as gr
from PIL import Image
import qrcode
from pathlib import Path
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
)
import numpy as np
import cv2

base_model_path = "stable-diffusion-v1-5/stable-diffusion-v1-5"
controlnet_path = "ilooro/controlnet-qr"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)
pipe = StableDiffusionControlNetPipeline.from_pretrained(base_model_path, controlnet=controlnet, torch_dtype=torch_dtype).to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


def gen_mask(input_image: Image.Image, step: int, resolution: int):
    img_1 = np.array(input_image)[:, :, ::-1].copy()
    img_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    w, h = img_gray.shape
    avg = np.mean(img_gray)

    mask = np.zeros((w, h, 3), np.uint8)
    mask[:, :, 2] = 255

    for i in range(0, w, step):
        for j in range(0, h, step):
            center = (i + (step // 2) + 1,
                      j + (step // 2) + 1)

            if (img_gray[center[0], center[1]] >= avg * 1.2):
                mask[center[0]-1:center[0]+1, center[1]-1:center[1]+1, :] = 255
            elif (img_gray[center[0], center[1]] < avg * 0.8):
                mask[center[0]-1:center[0]+1, center[1]-1:center[1]+1, :] = 0
            else:
                mask[center[0]-1:center[0]+1, center[1]-1:center[1]+1, :] = 128
        
    mask = cv2.resize(mask, (resolution, resolution))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = Image.fromarray(mask)
    return mask


def inference(
    qr_code_content: str,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float = 10.0,
    controlnet_conditioning_scale: float = 2.0,
    contorol_guidence_start: float = 0.1,
    contorol_guidence_end: float = 0.9,
    seed: int = -1,
    qrcode_image: Image.Image | None = None,
):
    if not prompt:
        raise gr.Error("Prompt is required")

    if not qrcode_image and not qr_code_content:
        raise gr.Error("QR Code Image or QR Code Content is required")

    if qrcode_image and qr_code_content:
        raise gr.Error("It is allowed to set either QR Code Content or QR Code Image, but not both")

    generator = torch.manual_seed(seed) if seed != -1 else torch.Generator()

    box_size = 10
    if qr_code_content or (qrcode_image and qrcode_image.size == (1, 1)):
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=box_size,
            border=4,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)
        qrcode_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qrcode_image = gen_mask(qrcode_image, box_size, 512)
    else:
        qrcode_image = gen_mask(qrcode_image, box_size, 512)


    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=qrcode_image,
        guidance_scale=float(guidance_scale),
        controlnet_conditioning_scale=float(controlnet_conditioning_scale),
        control_guidance_start=float(contorol_guidence_start),
        control_guidance_end=float(contorol_guidence_end),
        generator=generator,
        num_inference_steps=20,
    )
    return out.images[0]


with gr.Blocks() as blocks:
    with gr.Row():
        with gr.Column():
            qr_code_content = gr.Textbox(
                label="QR Code Content",
                info="QR Code Content or URL",
                value="",
            )
            with gr.Accordion(label="QR Code Image (Optional)", open=False):
                qr_code_image = gr.Image(
                    label="QR Code Image (Optional). Leave blank to automatically generate QR code",
                    type="pil",
                )
            prompt = gr.Textbox(
                label="Prompt",
                info="Prompt that guides the generation towards",
            )
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="ugly, disfigured, low quality, blurry, nsfw",
            )
            with gr.Accordion(label="Params", open=True):
                controlnet_conditioning_scale = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    step=0.01,
                    value=1.1,
                    label="Controlnet conditioning scale",
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=50.0,
                    step=0.25,
                    value=7.5,
                    label="Guidance scale",
                )
                contorol_guidence_start = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.1,
                    label="Control guidence start"
                )
                contorol_guidence_end = gr.Slider(
                    minimum=0,
                    maximum=1,
                    step=0.01,
                    value=0.9,
                    label="Control guidence end"
                )
                seed = gr.Slider(
                    minimum=-1,
                    maximum=9999999999,
                    step=1,
                    value=2313123,
                    label="Seed",
                    randomize=True,
                )
            run_btn = gr.Button("Run")
        with gr.Column():
            result_image = gr.Image(label="Result Image")

    run_btn.click(
        inference,
        inputs=[
            qr_code_content,
            prompt,
            negative_prompt,
            guidance_scale,
            controlnet_conditioning_scale,
            contorol_guidence_start,
            contorol_guidence_end,
            seed,
            qr_code_image,
        ],
        outputs=[result_image],
        concurrency_limit=1
    )

if __name__ == "__main__":
    blocks.queue(max_size=30).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        max_threads=8
    )