import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset.data import SimCLRDataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torch.utils.data import random_split
from models.SimCLR import SimCLR, NTXentLoss
from tqdm import tqdm
import utils.dataset_cleaning

simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.GaussianBlur(3),
    transforms.ToTensor()
])

basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# TODO change the path of your data set here
IMG_PATH = '/Volumes/U1/Fac/M2/HighVision_Corpus_Groundtruth/historicaldataset'
dataset = SimCLRDataset(IMG_PATH, basic_transform , simclr_transform)

total_size = len(dataset)
train_size = int(0.75 * total_size)
val_size = int(0.10 * total_size)
test_size = total_size - train_size - val_size 

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])


# Parameters zone
batch_size = 32
train_loader = DataLoader(train_set, batch_size, shuffle=True,num_workers=2)
val_loader = DataLoader(val_set, batch_size, shuffle=True,num_workers=2)
test_loader = DataLoader(test_set, batch_size, shuffle=True,num_workers=2)
backbone = models.resnet50(pretrained = True)
epoch = 5


model = SimCLR(backbone)
optimizer = torch.optim.AdamW(model.parameters())

device = torch.device('mps' if torch.mps.is_available else 'cpu')
print('Using: ',  device)

model = model.to(device)
ntxent = NTXentLoss(device)
criterion = torch.nn.CrossEntropyLoss().to(device)


#Training
training_losses = []
validation_losses = []

for i in range(epoch):
    training_loss = 0
    validation_loss = 0
    for images in tqdm(train_loader):
        v1,v2 = images
        v1 = v1.to(device)
        v2 = v2.to(device)

        _,z1 = model(v1)
        _,z2 = model(v2)

        logits, labels = ntxent(z1,z2)
        loss = criterion(logits,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

    with torch.no_grad():
        for images in tqdm(val_loader):
            v1,v2 = images
            v1 = v1.to(device)
            v2 = v2.to(device)

            _,z1 = model(v1)
            _,z2 = model(v2)

            logits, labels = ntxent(z1,z2)
            loss = criterion(logits,labels)
            
            validation_loss += loss.item()
    
    avg_train_loss = training_loss / len(train_loader)
    avg_val_loss = validation_loss / len(val_loader)
    training_losses.append(avg_train_loss)
    validation_losses.append(avg_val_loss)

    print(f"Epoch [{i+1}/{epoch}] - Training Loss: {avg_train_loss:.4f} - Validation Loss: {avg_val_loss:.4f}")
    


