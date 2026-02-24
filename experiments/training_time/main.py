import torch
from torchvision import datasets, transforms
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader
from model import RClassifier
from utils import save
from train import train, valid
from sonification import sonify
from scipy.io import wavfile

DEVICE = "mps"
REPLICATIONS = 1
EPOCHS = 10

transform = transforms.ToTensor()

train_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
valid_set = datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=32)
model = RClassifier(
    t=20, 
    z_size=33, 
    conv_channels=2, 
    activation="softsign"
).to(DEVICE)

optim = torch.optim.Adam(params=model.parameters(), lr=0.0005)

full_z_history = torch.empty(0, model.Z_SIZE).to(DEVICE)

for epoch in range(EPOCHS):
    model.train()
    train(model, train_loader, optim, epoch=epoch, do_tqdm=False)
    loss, accuracy = valid(model, valid_loader, epoch=epoch, do_tqdm=False)
    print(loss, accuracy)
    # save(model, epoch, f"experiments/training_time/model.pt")

    model.eval()
    with torch.no_grad():
        _ = model(valid_set[0][0].to(DEVICE))
        z_history = model.get_history(layer="z").to(DEVICE)
        z_history = z_history.squeeze(1)
        z_history = (z_history - torch.min(z_history)) / (torch.max(z_history) - torch.min(z_history)) * 2000 + 50
        full_z_history = torch.cat((full_z_history, z_history), dim=0) 

wavfile.write(f"experiments/training_time/history_1.wav", 44100, sonify(full_z_history[:, :], 0.05, do_stereo=True))