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

    def __init__(self, data_path, transform=None):
        """
        :param data_path: Root directory of the image data
        """
        self.data_path = data_path
        if transform is None:
            self.transform = transforms.Compose([
        transforms.ToTensor(), ])
        else:
            self.transform = transform
        self.real_images = self.load_images(os.path.join(data_path, 'training_real_toy'))
        self.fake_images = self.load_images(os.path.join(data_path, 'training_fake_toy'))
        # self.real_images = self.load_images(os.path.join(data_path, 'training_real'))
        # self.fake_images = self.load_images(os.path.join(data_path, 'training_fake'))
        # self.all_image_pairs = self.pair_images()
        # print(self.all_image_pairs)

    def __len__(self):
        # return len(self.all_image_pairs)
        return 10000

    def __getitem__(self, idx):
        img1, img2, np_label = self.get_image_pair(idx)
        real_image = self.transform(img1)
        fake_image = self.transform(img2)
        label = torch.from_numpy(np_label)

        return real_image, fake_image, label

    def load_images(self, path):
        images = []
        for filename in glob.glob(os.path.join(path, '*.jpg')):
            im = Image.open(filename)
            images.append(im)
            # print(np.array(im).shape)

        return images

    def get_image_pair(self, index):
        random.shuffle(self.real_images)
        random.shuffle(self.fake_images)
        if index % 2 == 0:
            # Get real-fake image pair
            img1 = random.choice(self.real_images)
            img2 = random.choice(self.fake_images)
            label = np.array([1, 0])
        else:
            # Get real-real image pair
            idx1 = random.randint(0, len(self.real_images)-1)
            idx2 = random.randint(0, len(self.real_images)-1)

            # Avoid same image pair
            while idx1 == idx2:
                idx1 = random.randint(0, len(self.real_images)-1)
                idx2 = random.randint(0, len(self.real_images)-1)

            img1 = self.real_images[idx1]
            img2 = self.real_images[idx2]
            label = np.array([1, 1])

        return (img1, img2, label)


if __name__ == '__main__':
    transform = transforms.Compose([Resize(105),
        transforms.ToTensor(),
    ]) # Convert the numpy array to a tensor
    # transform = None
    r = PairedImagesDataset('../data/real_and_fake_face/', transform)
    # print(len(r))
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
        plt.imshow(fake[0].permute(1, 2, 0))
        plt.show()
