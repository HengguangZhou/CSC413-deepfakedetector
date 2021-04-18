import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms, utils
import random
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import matplotlib.pyplot as plt


class PairedImagesDataset(Dataset):
    """
    The custom dataset contains real and fake images that will be paired together
    """
    def __init__(self, data_path, size=50000, transform=None):
        """
        :param data_path: Root directory of the image data
        """
        self.data_path = data_path
        self.size = size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(), ])
        else:
            self.transform = transform
        self.real_images = self.load_images(os.path.join(data_path, 'real'), True)
        self.fake_images = self.load_images(os.path.join(data_path, 'fake'), False)
      
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        img1, img2, np_label = self.get_image_pair()
        real_image = self.transform(img1)
        fake_image = self.transform(img2)
        label = torch.from_numpy(np_label)

        return real_image, fake_image, label

    def load_images(self, path, real):
        images = []
        label = 0.0
        if real:
            label = 1.0

        for filename in glob.glob(os.path.join(path, '*.jpg')):
            images.append((filename, np.array([label])))

        return images

    def get_image_pair(self):
        pooled_images = self.real_images+self.fake_images

        img1_info = random.choice(self.real_images)
        img2_info = random.choice(pooled_images)
        label = img1_info[1] == img2_info[1]
        img1 = Image.open(img1_info[0])
        img2 = Image.open(img2_info[0])

        return img1, img2, label


if __name__ == '__main__':
    transform = transforms.Compose([Resize(105),
        transforms.ToTensor(),
    ])  # Convert the numpy array to a tensor
    # transform = None
    r = PairedImagesDataset('../data/real_and_fake_face/', transform=transform)
    print(len(r))
    train_pairs_loader = DataLoader(dataset=r,
                                    batch_size=1,
                                    shuffle=True)
    itr = enumerate(train_pairs_loader)

    # set the length for training and validation set
    total_len = len(r)
    training_len = total_len // 4 * 3

    # storage for training data and validation data
    training_data = []
    validation_data = []
    for idx, data in itr:
        real, fake, label = data
        if idx < training_len:
            # train, train, train = data
            training_data.append(data)
        else:
            validation_data.append(data)
        # print(real.shape)
        # print(fake.shape)
        # print(label)
        # plt.imshow(fake[0].permute(1, 2, 0))
        # plt.show()
