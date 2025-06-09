import multiprocessing
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import models
from dataset.data import SimCLRDataset
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from models.SimCLR import SimCLR
from models.FineTunedDino import FineTunedDino
from tqdm import tqdm
from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss 
from dataset.augment import Augmentation
from utils.utils import get_new_run_folder
import utils.utils as utils
parser = argparse.ArgumentParser(description = "CBIR implementation using SimCLR")

parser.add_argument('-i', '--img_path', help='Path to image dataset', type=str, required=True)
parser.add_argument('-c', '--corrupted_path', help='Path to corrupted file', type=str, required=True)
parser.add_argument('-b', '--batch', help='Batch size', type=int, default=32)
parser.add_argument('-e', '--epoch', help='Number of epochs', type=int, default=20)
parser.add_argument('-lr','--learning_rate', help="Learning rate",type=float,default=0.01)
parser.add_argument('-m','--model', help="SimCLR or DINO",type=str, default="SimCLR")

def main(img_path, corrupted_path, batch_size, epoch, lr, model):
    print(batch_size, epoch)
   
    mean = [0.5573, 0.5598, 0.5478]
    std = [0.2112, 0.2071, 0.2058]

    run_dir = get_new_run_folder()
    transformations = Augmentation(mean, std, 224)
    dataset = SimCLRDataset(img_path, transformations, corrupted_path)

    dataset_size = len(dataset)

    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    print(f"Train size: {len(train_set)}")
    print(f"Validation size: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size, shuffle=True, num_workers=2)
    backbone = models.resnet50(pretrained=True)

    if(model == "Dino"):
        model = FineTunedDino()
    else:
        model = SimCLR(backbone)

    optimizer = torch.optim.AdamW(model.parameters(), lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-5)

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print('Using: ',  device)

    model = model.to(device)
    base_loss =  NTXentLoss(temperature=0.5)
    loss_func = SelfSupervisedLoss(base_loss)

    #Training
    training_losses = []
    validation_losses = []

    for i in range(epoch):
        model.train()
        training_loss = 0
        validation_loss = 0
        for images in tqdm(train_loader):
            v1,v2 = images
            v1 = v1.to(device)
            v2 = v2.to(device)

            _,z1 = model(v1)
            _,z2 = model(v2)

            loss = loss_func(z1,z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        with torch.no_grad():
            model.eval()
            for images in tqdm(val_loader):
                v1,v2 = images
                v1 = v1.to(device)
                v2 = v2.to(device)

                _,z1 = model(v1)
                _,z2 = model(v2)

                loss = loss_func(z1,z2)
                
                validation_loss += loss.item()
        
        scheduler.step()

        avg_train_loss = training_loss / len(train_loader)
        avg_val_loss = validation_loss / len(val_loader)
        training_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)

        print(f"Epoch [{i+1}/{epoch}] - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
        if (i+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(run_dir, f"simclr_model_epoch_{i+1}.pth"))

    plt.figure(figsize=[8,6])
    plt.plot(training_losses,'r',linewidth=3.0)
    plt.plot(validation_losses,'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = parser.parse_args()
    model = args.model

    if utils.parse_model(model):
        print("ERROR: Model must be Dino or SimCLR")
    else:
        main(args.img_path, args.corrupted_path, args.batch, args.epoch, args.learning_rate, model)