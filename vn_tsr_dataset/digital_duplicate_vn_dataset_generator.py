import os
import random
import cv2
import shutil

import albumentations as A
from albumentations.core.composition import OneOf
from albumentations.core.composition import Compose
from albumentations.core.transforms_interface import ImageOnlyTransform
import random

import numpy as np


def rotate_bound(image, angle, bg_color=(255, 255, 255)):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # New bounding dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # Adjust rotation matrix for translation
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]

    # Warp the rotated image with white background
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=bg_color)


class RotateBound(ImageOnlyTransform):
    def __init__(self, angle_limit=10, bg_color=(255,255,255), always_apply=False, p=0.5):
        super(RotateBound, self).__init__(always_apply, p)
        self.angle_limit = angle_limit
        self.bg_color = bg_color

    def apply(self, img, **params):
        angle = random.uniform(-self.angle_limit, self.angle_limit)
        return rotate_bound(img, angle, self.bg_color)

    def get_transform_init_args_names(self):
        return ("angle_limit", "bg_color")

if __name__ == '__main__':
    based_image_files = [{
        'img_file_name': 'vn_tsr_dataset/digital/digital_table_000001/img/digital_table_000001.png',
        'annotation_file_name': 'vn_tsr_dataset/digital/digital_table_000001/annotation/content.html',
        'annotation_structure_file_name': 'vn_tsr_dataset/digital/digital_table_000001/annotation/structure.json',
    }, {
        'img_file_name': 'vn_tsr_dataset/digital/digital_table_000002/img/digital_table_000002.png',
        'annotation_file_name': 'vn_tsr_dataset/digital/digital_table_000002/annotation/content.html',
        'annotation_structure_file_name': 'vn_tsr_dataset/digital/digital_table_000002/annotation/structure.json',
    }, {
        'img_file_name': 'vn_tsr_dataset/digital/digital_table_000003/img/digital_table_000003.png',
        'annotation_file_name': 'vn_tsr_dataset/digital/digital_table_000003/annotation/content.html',
        'annotation_structure_file_name': 'vn_tsr_dataset/digital/digital_table_000003/annotation/structure.json',
    }]


    start_digital_folder_number = 4
    end_digital_folder_number = 701
    for i in range(start_digital_folder_number, end_digital_folder_number):
        six_digits_str = str(i).zfill(6)
        if os.path.exists(f'vn_tsr_dataset/digital/digital_table_{six_digits_str}'):
            shutil.rmtree(f'vn_tsr_dataset/digital/digital_table_{six_digits_str}')

        if os.path.exists(f'vn_tsr_dataset/digital/digital_table_{six_digits_str}'):
            print(f'vn_tsr_dataset/digital/digital_table_{six_digits_str} exist')
            continue


        print(f'vn_tsr_dataset/digital/digital_table_{six_digits_str} not exist, start creating a stimulated image')
        os.makedirs(f'vn_tsr_dataset/digital/digital_table_{six_digits_str}')
        os.makedirs(f'vn_tsr_dataset/digital/digital_table_{six_digits_str}/img')
        os.makedirs(f'vn_tsr_dataset/digital/digital_table_{six_digits_str}/annotation')

        random_based_image_file = random.choice(based_image_files)
        print(f"Using {random_based_image_file['img_file_name']} as the base image")

        shutil.copy(random_based_image_file['annotation_file_name'], f'vn_tsr_dataset/digital/digital_table_{six_digits_str}/annotation/content.html')
        shutil.copy(random_based_image_file['annotation_structure_file_name'], f'vn_tsr_dataset/digital/digital_table_{six_digits_str}/annotation/structure.json')


        img = cv2.imread(random_based_image_file['img_file_name'])

        augmentations = Compose([
            OneOf([
                Compose([
                RotateBound(angle_limit=10, bg_color=(255,255,255), always_apply=True)  # Force rotation
            ])
            ], p=1),  # Lower chance of any rotation/perspective

            OneOf([
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.01, 0.02), p=0.5),
                A.MotionBlur(blur_limit=1, p=0.1)
            ], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.RandomShadow(shadow_roi=(0, 0.7, 1, 1), shadow_dimension=3, p=0.2),
            A.ElasticTransform(alpha=1, sigma=50, p=1),
            A.ImageCompression(quality_lower=40, quality_upper=80, p=0.7),
        ], p=1.0)
        augmented = augmentations(image=img)['image']

        output_path = os.path.join(f'vn_tsr_dataset/digital/digital_table_{six_digits_str}/img', f"digital_table_{six_digits_str}.png")
        cv2.imwrite(output_path, augmented)
        print(f"Saved {output_path}")

        # import shutil
        # if os.path.exists(f'vn_tsr_dataset/digital/{i}'):
        #     shutil.rmtree(f'vn_tsr_dataset/digital/{i}')


