from torch.utils.data import Dataset
# import cv2
import glob
import numpy as np
from PIL import Image
import io


class LungsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.lungs = list()

        for imageName in glob.glob(root_dir + '/*.jpg'):
            self.lungs.append(imageName)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.lungs)

    def __getitem__(self, idx):
        img = Image.open(self.lungs[idx]).convert('L')

        # img = cv2.imread(self.lungs[idx], 0)

        img = np.array(img, dtype='float32').reshape(128,128,1)

        if self.transform:
            img = self.transform(img)

        return img