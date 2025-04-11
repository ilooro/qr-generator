import numpy as np
import cv2
import os
from glob import glob
import json
from tqdm import tqdm
import qrcode
from PIL import Image


class MaskGenerator():
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
    

    def preprocess(self, img):
        w, h, _ = img.shape
        s = np.max((w, h))

        step = s // self.num_blocks
        if step % 2 == 0 and step != 0:
            s += step
        
        residual = s % self.num_blocks
        step = s // self.num_blocks
        if residual != 0:
            if step != 0:
                s -= residual
            else:
                s += self.num_blocks - residual

        step = s // self.num_blocks

        img = cv2.resize(img, (s, s))

        return img, step


    def mask_gen(self, img, step):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        w, h = img_gray.shape
        avg = np.mean(img_gray)

        mask = np.zeros((w, h, 3), np.uint8)
        mask[:, :, :] = 255

        for i in range(0, w, step):
            for j in range(0, h, step):
                center = (i + (step // 2) + 1,
                          j + (step // 2) + 1)

                cur_slice = img_gray[i:i+step, j:j+step]

                if (cur_slice >= avg * 1.2).all():
                    mask[center[0], center[1], :] = 255
                elif (cur_slice < avg * 0.8).all():
                    mask[center[0], center[1], :] = 0
                else:
                    mask[center[0], center[1], :]  = 128
        
        return mask
    

    def work_diffusiondb(self, source_path, output_path):
        j = 0
        for i in range(1, 101):
            if i < 10:
                part_num = f'part-00000{i}'
            elif i >= 10 and i < 100:
                part_num = f'part-0000{i}'
            elif i >= 100:
                part_num = f'part-000{i}'
            else:
                raise IndexError
            images = glob(os.path.join(source_path, part_num, '*.png'))
            json_file = os.path.join(source_path, part_num, part_num + '.json')

            with open(json_file) as f:
                json_dict = json.load(f)
                for img_path in tqdm(images, desc=part_num):
                    img = cv2.imread(img_path)
                    img_1 = cv2.resize(img, (512, 512))
                    prompt = json_dict[img_path.split('\\')[-1]]['p']

                    img_1, step = self.preprocess(img_1)
                    try:
                        mask = self.mask_gen(img_1, step)
                    except IndexError as e:
                        print(img_path)
                        continue

                    img_1 = cv2.resize(img_1, (512, 512))
                    mask = cv2.resize(mask, (512, 512))

                    cv2.imwrite(os.path.join(output_path, 'source', f'{j}.jpeg'), img_1)
                    cv2.imwrite(os.path.join(output_path, 'target', f'{j}.jpeg'), mask)

                    dict_ = {"image": "source/" + f"{j}.jpeg", "conditioning_image": "target/" + f"{j}.jpeg", "text": prompt}
                    with open(os.path.join(output_path, 'train.json'), 'a', encoding='utf-8') as f:
                        json.dump(dict_, f, ensure_ascii=False)
                        f.write('\n')
                    j += 1
        return


if __name__ == "__main__":
    a = MaskGenerator(61)
    a.work_diffusiondb('./dataset/diffusiondb/2M/', './dataset/diffusiondb_QR_masks_9x9')
