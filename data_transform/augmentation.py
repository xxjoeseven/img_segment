import cv2
import elasticdeform
import imageio
import numpy as np
import os
import random

from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm

import utility as ut
from utility.seed import set_seed

def augment_img(images, annotations, save_path, augment=True):
    """
    """
    IMAGE_SIZE = (512,512)

    for _, (x, y) in tqdm(enumerate(zip(images, annotations)), 
            total=len(images)):

            image_name = os.path.basename(x).split('.')[0]
            annotated_name = os.path.basename(y).split('.')[0]

            # Read image and annotation
            x = cv2.imread(x, cv2.IMREAD_COLOR)
            y = imageio.mimread(y)[0]

            # print(x.shape, y.shape)
            # break

            if augment == True:

                aug = HorizontalFlip(p=1.0)
                augmented = aug(image=x, annotation=y)
                x1 = augmented["image"]
                y1 = augmented["annotation"]

                aug = VerticalFlip(p=1.0)
                augmented = aug(image=x, annotation=y)
                x2 = augmented["image"]
                y2 = augmented["annotation"]

                aug = Rotate(limit=15, p=1.0)
                augmented = aug(image=x, annotation=y)
                x3 = augmented["image"]
                y3 = augmented["annotation"]

                # From Ronneberger Paper
                # smooth deformation using random displacement vectors on a
                # coarse 3 by 3 grid, 10 pixels standard deviation and bicubic
                # interpolation.

                xy4 = \
                elasticdeform.deform_random_grid([x ,y], 
                                                sigma=10, 
                                                points=3, 
                                                order=3,
                                                axis=[(0,1),(0,1)]) 

                X = [x, x1, x2, x3, xy4[0]]
                Y = [y, y1, y2, y3, xy4[1]]

            else:
                X =[x]
                Y = [y]


            number = 0

            for i, m in zip(X, Y):
                i = cv2.resize(i, IMAGE_SIZE)
                m = cv2.resize(m, IMAGE_SIZE)

                temp_img_name = f"{image_name}_{number}.png"
                temp_annotated_name = f"{annotated_name}_{number}.png"

                image_path = os.path.join(save_path, "images", temp_img_name)

                annotation_path = os.path.join(save_path,
                                            "annotated", temp_annotated_name)

                cv2.imwrite(image_path, i)
                cv2.imwrite(annotation_path, m)

                number +=1

if __name__ == "__main__":

    # Set Seed
    set_seed(77)

    # Augment Training Data 
    train_dir = r"..\dataset\training\images\\"
    train_img_path = ut.utils.get_img_list(train_dir, 'tif')

    train_mask_dir = r"..\dataset\training\\1st_manual\\"
    train_mask_path = ut.utils.get_img_list(train_mask_dir, "gif")

    ut.utils.make_dir(r"..\dataset\augmented\training\images\\")
    ut.utils.make_dir(r"..\dataset\augmented\training\annotated\\")

    augment_img(train_img_path, train_mask_path, 
            r"..\dataset\augmented\training\\", augment=True)

    # Augment Validation Data to increase number of images
    validate_dir = r"..\dataset\validate\images\\" 
    validate_img_path = ut.utils.get_img_list(validate_dir, 'tif')

    validate_mask_dir = r"..\dataset\validate\\1st_manual\\"
    validate_mask_path = ut.utils.get_img_list(validate_mask_dir, 'gif')

    ut.utils.make_dir(r"..\dataset\augmented\validate\images\\")
    ut.utils.make_dir(r"..\dataset\augmented\validate\annotated\\")

    augment_img(validate_img_path, validate_mask_path,
            r"..\dataset\augmented\validate\\", augment=True)
