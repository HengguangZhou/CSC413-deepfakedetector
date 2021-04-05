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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='siamese')
    parser.add_argument('--train_data', type=str, required=True)
    # parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weights_dir", type=str, default="checkpoints/")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument('--input_channels', type=int, default=3)

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
    else:
        model = siamese(opts.input_channels)

    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    val = torch.nn
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    optimizer.zero_grad()

    transform = Compose([Resize(200),
                    ToTensor(),
                    ])
    # transform = None
    data_pairs = PairedImagesDataset(data_path=opts.train_data, transform=transform)
    train_pairs, val_pairs = random_split(dataset=data_pairs,
                 lengths=[int(0.8 * len(data_pairs)),
                          int(np.ceil((1 - 0.8) * len(data_pairs)))])
    # val_pairs = PairedImagesDataset(opts.test_data)
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
            los = 0
            # print(f'train len: {len(train_pairs_loader)}')
            for idx, data in enumerate(train_pairs_loader):
                img1, img2, label = data
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                pred = model(img1, img2)
                # print(pred)
                # print(label)
                # print("------------------------")
                loss = criterion(pred, label)
                los += loss.item()
                loss.backward()
                optimizer.step()

                t.set_postfix(loss='{:.6f}'.format(los / (idx + 1)))
                t.update(img1.shape[0])

        torch.save(model.state_dict(), os.path.join(opts.weights_dir, f'model_{opts.model}_epoch_{epoch}.pth'))

        model.eval()

        with tqdm(total=(len(val_pairs) - len(val_pairs) % opts.batch_size)) as t:
            t.set_description(f'val epoch: {epoch}/{opts.num_epochs - 1}')
            val_corr = 0
            # print(f'val len: {len(val_pairs_loader)}')
            for idx, data in enumerate(val_pairs_loader):
                img1, img2, label = data
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                pred = model(img1, img2)
                batch_corr = int(torch.sum(torch.round(pred) == label))

                t.set_postfix(accuracy='{:.2f}%'.format(batch_corr / img1.shape[0] * 100))
                t.update(img1.shape[0])
                val_corr += batch_corr
            print(f"\nval accuracy: {val_corr / len(val_pairs)}")


