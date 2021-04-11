import argparse
import os
import torch
import numpy as np
from models.siamese_net import siamese
from models.cnn_pairwise import CnnPairwise
from tqdm import tqdm
from dataset import PairedImagesDataset
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch import nn
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from loss import ContrastiveLoss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn_pairwise')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    # parser.add_argument('--train_data', type=str, default='../data/real_and_fake_face')
    # parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weights_dir", type=str, default="checkpoints/")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--margin', type=int, default=2.5)

    opts = parser.parse_args()

    if not os.path.exists(opts.weights_dir):
        os.mkdir(opts.weights_dir)

    if not os.path.exists(opts.weights_dir):
        os.mkdir(opts.weights_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if opts.model == 'siamese':
        model = siamese(opts.input_channels)
    elif opts.model == 'cnn_pairwise':
        model = CnnPairwise(opts.input_channels)
    else:
        model = CnnPairwise(opts.input_channels)

    model = model.to(device)

    criterion = ContrastiveLoss(margin=opts.margin)
    criterion1 = torch.nn.BCEWithLogitsLoss()
    val = torch.nn
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    optimizer.zero_grad()

    transform = Compose([Resize(105),
                    ToTensor(),
                    ])
    train_pairs = PairedImagesDataset(data_path=opts.train_data, size=40000, transform=transform)
    val_pairs = PairedImagesDataset(data_path=opts.val_data, size=10000, transform=transform)
    train_pairs_loader = DataLoader(dataset=train_pairs,
                                    batch_size=opts.batch_size,
                                    shuffle=True)

    val_pairs_loader = DataLoader(dataset=val_pairs,
                                  batch_size=opts.batch_size,
                                  shuffle=True)

    for epoch in range(opts.num_epochs):
        model.train()
        with tqdm(total=(len(train_pairs) - len(train_pairs) % opts.batch_size)) as t:
            t.set_description(f'train epoch: {epoch}/{opts.num_epochs - 1}')
            train_corr = 0
            los = 0
            # print(f'train len: {len(train_pairs_loader)}')
            for idx, data in enumerate(train_pairs_loader):
                img1, img2, label = data
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                i1, i2, pred = model(img1, img2)
                # print(pred)
                # print(label)
                # print("------------------------")
                if epoch < 10:
                    loss = criterion(i1, i2, label)
                else:
                    loss = criterion1(pred,  label)
                los += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_corr = int(torch.sum(torch.round(torch.sigmoid(pred)) == label))
                batch_loss = label

                t.set_postfix(loss='{:.6f}'.format(los / (idx + 1)),
                              train_accuracy='{:.2f}%'.format(batch_corr / img1.shape[0] * 100))
                t.update(img1.shape[0])
                train_corr += batch_corr
            print("\ntrain accuracy: {:.2f}%".format(train_corr / len(train_pairs) * 100))

        torch.save(model.state_dict(), os.path.join(opts.weights_dir, f'model_{opts.model}_epoch_latest.pth'))

        model.eval()

        with tqdm(total=(len(val_pairs) - len(val_pairs) % opts.batch_size)) as t:
            t.set_description(f'val epoch: {epoch}/{opts.num_epochs - 1}')
            val_corr = 0
            # print(f'val len: {len(val_pairs_loader)}')
            for idx, data in enumerate(val_pairs_loader):
                img1, img2, label = data
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                i1, i2, pred = model(img1, img2)
                batch_corr = int(torch.sum(torch.round(torch.sigmoid(pred)) == label))

                t.set_postfix(accuracy='{:.2f}%'.format(batch_corr / img1.shape[0] * 100))
                t.update(img1.shape[0])
                val_corr += batch_corr
            print("\nval accuracy: {:.2f}%".format(val_corr / len(val_pairs) * 100))
