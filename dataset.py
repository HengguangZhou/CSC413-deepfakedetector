import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms, utils
import random


class PairedImagesDataset(Dataset):
    """
    The custom dataset contains real and fake images that will be paired together
    """

    def __init__(self, data_path, transform):
        """
        :param data_path: Root directory of the image data
        """
        self.data_path = data_path
        self.transform = transform
        self.real_images = self.load_images(os.path.join(data_path, 'training_real_toy'))
        self.fake_images = self.load_images(os.path.join(data_path, 'training_fake_toy'))
        # self.real_images = self.load_images(os.path.join(data_path, 'training_real'))
        # self.fake_images = self.load_images(os.path.join(data_path, 'training_fake'))
        self.all_image_pairs = self.pair_images()
        # print(self.all_image_pairs)

    def __len__(self):
        return len(self.all_image_pairs)

    def __getitem__(self, idx):
        images = self.all_image_pairs[idx]
        real_image = self.transform(images[0])
        fake_image = self.transform(images[1])
        label = torch.from_numpy(images[2])

        return real_image, fake_image, label

    def load_images(self, path):
        images = []
        for filename in glob.glob(os.path.join(path, '*.jpg')):
            im = Image.open(filename)
            images.append(np.array(im))
            # print(np.array(im).shape)

        return images

    def pair_images(self):
        real_fake_pairs = []
        real_real_pairs = []

        # Randomly shuffle both lists so they can be paired
        random.shuffle(self.real_images)
        random.shuffle(self.fake_images)

        # Paired each fake image with a real image
        for i in range(len(self.fake_images)):
            choice = random.choice(self.real_images)
            label = np.array([1, 0])
            real_fake_pairs.append((choice, self.fake_images[i], label))

        # Want to pair real with real
        for i in range(len(self.real_images)):
            choice = random.choice(self.real_images)
            label = np.array([1, 1])
            real_real_pairs.append((choice, self.real_images[i], label))

        # Mix the real-fake pair and the real-real pair together
        all_pairs = real_fake_pairs + real_real_pairs
        random.shuffle(all_pairs)

        return all_pairs


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the numpy array to a tensor
    ])
    r = PairedImagesDataset('../data/real_and_fake_face/', transform)
    itr = enumerate(r)
    for idx, data in itr:
        real, fake, label = data
        # print(real.shape)
        # print(fake.shape)
        # print(label)
