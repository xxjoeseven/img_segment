import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, images_path, annotations_path):
        
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """
        """

        # read images
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0 # Normalize
        image = np.transpose(image, (2, 0, 1)) # change (512,512,3) to (3,512,512)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)

        # read annotations
        annotation = cv2.imread(self.annotations_path[index], cv2.IMREAD_GRAYSCALE)
        annotation = annotation/255.0 # Normalize
        annotation = np.expand_dims(annotation, axis=0) # (1, 512, 512)
        annotation = annotation.astype(np.float32)
        annotation = torch.from_numpy(annotation)

        return image, annotation

    def __len__(self):
        """
        """
        return self.n_samples
