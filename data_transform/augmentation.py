import cv2
import imageio
import numpy as np
import os

from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm

import utility as ut

image_path = r"..\dataset\training\images\\"
train_img_path = ut.utils.get_img_list(image_path, 'tif')

annotation_path = r"..\dataset\training\\1st_manual\\"
annotated_img_path = ut.utils.get_img_list(annotation_path, "gif")

ut.utils.make_dir(r"..\dataset\augmented\training\images\\")
ut.utils.make_dir(r"..\dataset\augmented\training\annotated\\")

def augment_img(images, annotations, save_path, augment=True):
    """
    """
    IMAGE_SIZE = (512,512)

    for index, (x, y) in tqdm(enumerate(zip(images, annotations)), 
            total=len(images)):

            image_name = os.path.basename(x).split('.')[0]
            annotated_name = os.path.basename(y).split('.')[0]

            # Read image and annotation
            x = cv2.imread(x, cv2.IMREAD_COLOR)
            y = imageio.mimread(y)[0]

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

                X = [x, x1, x2, x3]
                Y = [y, y1, y2, y3]

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
    np.random.seed(77)

    augment_img(train_img_path, annotated_img_path, 
            r"..\dataset\augmented\training\\", augment=True)    

