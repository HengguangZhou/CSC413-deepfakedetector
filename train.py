import argparse
import os
import torch
from models.siamese_net import siamese
from models.cnn_pairwise import CnnPairwise
from models.CRFN import CRFN
from tqdm import tqdm
from dataset import PairedImagesDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from loss import ContrastiveLoss
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CRFN')
    parser.add_argument('--train_data', type=str, required=True)
    parser.add_argument('--val_data', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weights_dir", type=str, default=".\\weights\\")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument('--input_channels', type=int, default=3)
    parser.add_argument('--margin', type=float, default=1.5)
    parser.add_argument('--enable_fake_pairs', type=bool, default=False)

    opts = parser.parse_args()

    writer = SummaryWriter()

    weights_path = os.path.join(opts.weights_dir,
                                opts.model)
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if opts.model == 'siamese':
        model = siamese(opts.input_channels)
    elif opts.model == 'cnn_pairwise':
        model = CnnPairwise(opts.input_channels)
    elif opts.model == 'CRFN':
        model = CRFN(opts.input_channels)

    model = model.to(device)

    contrastive = ContrastiveLoss(margin=opts.margin)
    CE = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    optimizer.zero_grad()

    transform = Compose([Resize(105),
                         ToTensor()])

    train_pairs = PairedImagesDataset(data_path=opts.train_data, size=40000,
                                      transform=transform, enable_fake_pairs=opts.enable_fake_pairs)
    val_pairs = PairedImagesDataset(data_path=opts.val_data, size=10000,
                                    transform=transform, enable_fake_pairs=opts.enable_fake_pairs)
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
            conLos = 0
            for idx, data in enumerate(train_pairs_loader):
                img1, img2, label1, label2, same_label = data
                img1, img2, label1, label2, same_label = img1.to(device), img2.to(device), label1.to(device), \
                                                    label2.to(device), same_label.to(device)

                if opts.model == 'siamese':
                    pred = model(img1, img2)
                    loss = CE(pred, label2)
                    los += loss.item()
                else:  # If the model is cnn-pairwise or CRFN
                    i1, i2, pred = model(img1, img2)
                    con = contrastive(i1, i2, same_label)
                    cro = CE(pred, label2)
                    if epoch < 10:
                        loss = con
                    else:
                        loss = cro
                    conLos += con.item()
                    los += cro.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_corr = int(torch.sum(torch.round(torch.sigmoid(pred)) == label2))

                t.update(img1.shape[0])
                train_corr += batch_corr

                t.set_postfix(loss='{:.6f}'.format(los / (idx + 1)), ConLoss='{:.6f}'.format(conLos / (idx + 1)),
                              train_accuracy='{:.2f}%'.format(batch_corr / img1.shape[0] * 100))

            print("\nConLoss: {:.2f}".format(conLos / len(train_pairs) * opts.batch_size))
            print("\nloss: {:.2f}".format(los / len(train_pairs) * opts.batch_size))
            print("\ntrain accuracy: {:.2f}%".format(train_corr / len(train_pairs) * 100))
            writer.add_scalar(f'ConLoss/train',
                              conLos / len(train_pairs) * opts.batch_size, epoch)
            writer.add_scalar(f'CELoss/train',
                              los / len(train_pairs) * opts.batch_size, epoch)
            writer.add_scalar(f'Accuracy/train',
                              train_corr / len(train_pairs), epoch)

        torch.save(model.state_dict(), os.path.join(weights_path,
                                    f"{opts.model}_latest.pth"))

        model.eval()

        with tqdm(total=(len(val_pairs) - len(val_pairs) % opts.batch_size)) as t:
            t.set_description(f'val epoch: {epoch}/{opts.num_epochs - 1}')
            val_corr = 0
            for idx, data in enumerate(val_pairs_loader):
                img1, img2, label1, label2, same_label = data
                img1, img2, label1, label2, same_label = img1.to(device), img2.to(device), \
                                                         label1.to(device), label2.to(device), same_label.to(device)
                i1, i2, pred = model(img1, img2)
                batch_corr = int(torch.sum(torch.round(torch.sigmoid(pred)) == label2))

                t.set_postfix(accuracy='{:.2f}%'.format(batch_corr / img1.shape[0] * 100))
                t.update(img1.shape[0])
                val_corr += batch_corr
            print("\nval accuracy: {:.2f}%".format(val_corr / len(val_pairs) * 100))
            writer.add_scalar(f'Accuracy/test',
                              val_corr / len(val_pairs), epoch)
