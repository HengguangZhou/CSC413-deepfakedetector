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
from models.CRFN import CRFN
from models.cnn_pairwise import CnnPairwise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='siamese')
    # parser.add_argument('--real_image', type=str, required=True)
    # parser.add_argument('--unknown_image', type=str, required=True)
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument("--weights", type=str, default="checkpoints/")
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    opts = parser.parse_args()

    # real_img = Image.open(opts.real_image)
    # unknown_img = Image.open(opts.unknown_image)
    transform = Compose([Resize(105),
                         ToTensor(),
                         ])

    # real_img, unknown_img = transform(real_img).unsqueeze(0), transform(unknown_img).unsqueeze(0)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if opts.model == 'siamese':
        model = siamese(opts.input_channels)
    elif opts.model == "CRFN":
        model = CRFN(opts.input_channels)
    elif opts.model == "CnnPairwise":
        model = CnnPairwise(opts.input_channels)

    model.load_state_dict(torch.load(opts.weights, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    test_pairs = PairedImagesDataset(data_path=opts.test_data, size=10000, transform=transform)

    test_pairs_loader = DataLoader(dataset=test_pairs,
                                  batch_size=opts.batch_size,
                                  shuffle=True)

    with tqdm(total=(len(test_pairs) - len(test_pairs) % opts.batch_size)) as t:
        val_corr = 0
        for idx, data in enumerate(test_pairs_loader):
            img1, img2, label = data
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            i1, i2, pred = model(img1, img2)
            batch_corr = int(torch.sum(torch.round(torch.sigmoid(pred)) == label))

            t.set_postfix(accuracy='{:.2f}%'.format(batch_corr / img1.shape[0] * 100))
            t.update(img1.shape[0])
            val_corr += batch_corr
        print("\nval accuracy: {:.2f}%".format(val_corr / len(test_pairs) * 100))
