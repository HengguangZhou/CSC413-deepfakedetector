import argparse
import os
import torch
import numpy as np
from models.siamese_net import siamese
from tqdm import tqdm
from dataset import PairedImagesDataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch import nn
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='siamese')
    parser.add_argument('--real_image', type=str, required=True)
    parser.add_argument('--unknown_image', type=str, required=True)
    # parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument("--weights", type=str, default="checkpoints/")
    parser.add_argument('--input_channels', type=int, default=3)

    opts = parser.parse_args()

    real_img = Image.open(opts.real_image)
    unknown_img = Image.open(opts.unknown_image)
    transform = Compose([Resize(105),
                         ToTensor(),
                         ])

    real_img, unknown_img = transform(real_img).unsqueeze(0), transform(unknown_img).unsqueeze(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if opts.model == 'siamese':
        model = siamese(opts.input_channels)
    else:
        model = siamese(opts.input_channels)

    model.load_state_dict(torch.load(opts.weights, map_location=device), strict=False)
    model.eval()

    print(torch.sigmoid(model(real_img, unknown_img)))
