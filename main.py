import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import RClassifier
from train import train, valid
from utils import save, load

DEVICE = "mps"
PATH = "models/02-03.2__.pt"

# Transform to float-tensor-fy and norm 
transform = transforms.ToTensor()

train_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
valid_set = datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=32)

try:
    model, epoch = load(PATH)
    model = model.to(DEVICE)
except ValueError:
    model = RClassifier(
        t=16, 
        z_size=33, 
        conv_channels=2, 
        activation="softsign"
    ).to(DEVICE)
    epoch = 0

optim = torch.optim.Adam(params=model.parameters(), lr=0.0005)

for i in range(6):
    train(model, train_loader, optim, epoch=epoch, device=DEVICE)
    print(valid(model, valid_loader, epoch=epoch, device=DEVICE))
    epoch += 1
    save(model, epoch, PATH)