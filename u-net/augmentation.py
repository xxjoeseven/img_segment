import cv2
import imageio
import numpy as np
import os

from albumentations import HorizontalFlip, VerticalFlip, Rotate
from glob import glob
from tqdm import tqdm

def get_tif_list(path):
    """Search given path for .tif files

    Args: 
        path: file path of the directory with the images

    Return:
        a list of the images path.
    """

    list_img = glob(os.path.join(path, "*.tif"))

    return list_img

if __name__ == "__main__":
    np.random.seed(77)

    # change current working directory to the dataset directory
    os.chdir(r"..\dataset\training\images\\")
    data_path = os.getcwd()
    train_img_path = get_tif_list(data_path)
    print(train_img_path)
