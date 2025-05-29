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
from tqdm import tqdm
from pytorch_metric_learning.losses import SelfSupervisedLoss, NTXentLoss 
from dataset.augment import basic_transform, simclr_transform

parser = argparse.ArgumentParser(description = "CBIR implementation using SimCLR")

parser.add_argument('-i', '--img_path', help='Path to image dataset', type=str, required=True)
parser.add_argument('-c', '--corrupted_path', help='Path to corrupted file', type=str, required=True)
parser.add_argument('-b', '--batch', help='Batch size', type=int, default=32)
parser.add_argument('-e', '--epoch', help='Number of epochs', type=int, default=20)


def get_new_run_folder(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run_")]
    run_ids = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_id = max(run_ids, default=0) + 1
    run_folder = os.path.join(base_dir, f"run_{next_id}")
    os.makedirs(run_folder)
    return run_folder


def main(img_path, corrupted_path, batch_size, epoch):
    run_dir = get_new_run_folder()
    dataset = SimCLRDataset(img_path, basic_transform(224), simclr_transform(224), corrupted_path)

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    print(f"Train size: {len(train_set)}")
    print(f"Validation size: {len(val_set)}")

    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size, shuffle=True, num_workers=2)
    backbone = models.resnet50(pretrained=True)

    # Load one batch
    v1_batch, v2_batch = next(iter(train_loader))

    # Convert to CPU tensors for visualization (if needed)
    v1_batch = v1_batch.cpu()
    v2_batch = v2_batch.cpu()

    model = SimCLR(backbone)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-5)

    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print('Using: ',  device)

    model = model.to(device)
    base_loss =  NTXentLoss()
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

    main(args.img_path, args.corrupted_path, args.batch, args.epoch)
