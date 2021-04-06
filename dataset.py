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
        self.real_images = self.load_images(os.path.join(data_path, 'training_real_toy'), True)
        self.fake_images = self.load_images(os.path.join(data_path, 'training_fake_toy'), False)
        self.pair_lst_index = 0
        self.pair_lst = self.populate_pair_lst()
        # self.real_images = self.load_images(os.path.join(data_path, 'training_real'))
        # self.fake_images = self.load_images(os.path.join(data_path, 'training_fake'))

    def __len__(self):
        # return len(self.all_image_pairs)
        return len(self.pair_lst)

    def __getitem__(self, idx):
        img1, img2, np_label = self.get_image_pair(idx)
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
            im = Image.open(filename)
            images.append(im)
            # print(np.array(im).shape)

        return images

    def get_image_pair(self, idx):
        img_pair_index, label = self.pair_lst[idx]
        if label: # if real
            img1 = self.real_images[img_pair_index[0]]
            img2 = self.real_images[img_pair_index[1]]
        else:
            img1 = self.real_images[img_pair_index[0]]
            img2 = self.fake_images[img_pair_index[1]]

        return img1, img2, label

    def populate_pair_lst(self):
        lst_index = []
        r = len(self.real_images)
        f = len(self.fake_images)
        # index for real - fake
        for i in range(r):
            for j in range(f):
                # format: each element has format of (index, label)
                lst_index.append(((i, j), np.array([0.0])))

        # index for real - real
        for i in range(r):
            for j in range(r):
                lst_index.append(((i, j), np.array([1.0])))

        random.shuffle(lst_index)
        return lst_index


if __name__ == '__main__':
    transform = transforms.Compose([Resize(105),
                                    transforms.ToTensor(),
                                    ])  # Convert the numpy array to a tensor
    # transform = None
    r = PairedImagesDataset('../data/real_and_fake_face/', transform)
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
