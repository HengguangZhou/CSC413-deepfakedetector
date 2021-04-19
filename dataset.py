import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from torchvision import transforms
import random
from torchvision.transforms import Resize


class PairedImagesDataset(Dataset):
    """
    The custom dataset contains real and fake images that will be paired together
    """
    def __init__(self, data_path, size=50000, transform=None, enable_fake_pairs=False):
        """
        :param data_path: Root directory of the image data
        :param size: size of the total data length
        :param transform: transform the output data into a specific data structure, default is tensor
        :param enable_fake_pairs: True if testing for cnn_pairwise, should be left as false for all other experiments
        """
        self.data_path = data_path
        self.size = size
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(), ])
        else:
            self.transform = transform
        self.enable_fake_pairs = enable_fake_pairs
        self.real_images = self.load_images(os.path.join(data_path, 'real'), True)
        self.fake_images = self.load_images(os.path.join(data_path, 'fake'), False)
      
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        :param idx: Not used since each pair is sampled randomly
        The return value are as follows: real image, unknown image, label for first image, label for second image,
        output label.
        Output label's value is determined by if user enables enable_fake_pairs, if not enabled, output label takes the
        value as the same as the label for second image, if enabled, output label outputs 1 if both labels are the same
        and 0 if both labels are different from each other
        """

        img1, img2, np_label1, np_label2 = self.get_image_pair()
        real_image = self.transform(img1)
        fake_image = self.transform(img2)
        if self.enable_fake_pairs:
            np_label = np.array([0.0])
            if np_label1 == np_label2:
                np_label = np.array([1.0])
        else:
            np_label = np_label2
        label1 = torch.from_numpy(np_label1)
        label2 = torch.from_numpy(np_label2)
        label = torch.from_numpy(np_label)

        return real_image, fake_image, label1, label2, label

    def load_images(self, path, real):
        """
        :param path: Root directory of the image data
        :param real: 1 if the path leads to real images, 0 if the path leads to fake images
        Load in all files under path and append a label associated with each file.
        Label is 1 if the image is real, and 0 if the image is fake.
        Return a list of tuples of the format (image, label)
        """
        images = []
        label = 0.0
        if real:
            label = 1.0

        for filename in glob.glob(os.path.join(path, '*.jpg')):
            images.append((filename, np.array([label])))

        return images

    def get_image_pair(self):
        """
        Randomly get a pair of real image and an unknown image.
        Real image is randomly sampled from a list of all real images
        Unknown image is randomly sampled from a pooled list where
        the pooled list consist of both real and fake images
        Return the first and second image, then their respective labels
        """
        pooled_images = self.real_images + self.fake_images
        img1_info = random.choice(self.real_images)
        if self.enable_fake_pairs:
            img1_info = random.choice(pooled_images)
        img2_info = random.choice(pooled_images)
        img1 = Image.open(img1_info[0])
        img2 = Image.open(img2_info[0])
        label1 = np.array([img1_info[1][0]])
        label2 = np.array([img2_info[1][0]])

        return img1, img2, label1, label2


if __name__ == '__main__':
    transform = transforms.Compose([Resize(105),
        transforms.ToTensor(),
    ])  # Convert the numpy array to a tensor
    # transform = None
    r = PairedImagesDataset('../data/real_and_fake_face/', transform=transform, enable_fake_pairs=False)
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
        real, fake, label1, label2, label = data
        if idx < training_len:
            # train, train, train = data
            training_data.append(data)
        else:
            validation_data.append(data)
        # print(real.shape)
        # print(fake.shape)
        print("label1:{}, label2:{}, label:{}".format(label1, label2, label))
        # plt.imshow(fake[0].permute(1, 2, 0))
        # plt.show()
