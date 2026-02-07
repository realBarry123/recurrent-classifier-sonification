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

noise_functions = [
    lambda x: x, 
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.05, clip=True), 
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.1, clip=True), 
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.2, clip=True), 
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.3, clip=True), 
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.4, clip=True), 
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.6, clip=True), 
]
transform = transforms.ToTensor()

for i in range(len(noise_functions)):
    noise_function = noise_functions[i]

    train_set = datasets.FashionMNIST(
        root=".", 
        train=True, 
        download=True, 
        transform=transforms.Compose([transform, noise_function])
    )
    valid_set = datasets.FashionMNIST(
        root=".", 
        train=False, 
        download=True, 
        transform=transforms.Compose([transform, noise_function])
    )

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    valid_loader = DataLoader(dataset=valid_set, batch_size=32) 

    model = RClassifier(
        t=16, 
        z_size=33, 
        conv_channels=2, 
        activation="softsign"
    ).to(DEVICE)

    optim = torch.optim.Adam(params=model.parameters(), lr=0.0005)

    epoch = 0

    for _ in range(6):
        train(model, train_loader, optim, epoch=epoch)
        valid(model, valid_loader, epoch=epoch)
        epoch += 1
        save(model, epoch, f"experiments/noise/{str(i)}.pt")

    with torch.no_grad():
        model.eval()
        _ = model(valid_set[0][0].to(DEVICE))
        z_history = model.get_history(layer="z")
        z_history = z_history.squeeze(1)
        z_history = torch.nn.functional.sigmoid(z_history * 10) * 2000 + 50
        wavfile.write(f"experiments/noise/{str(i)}.wav", 44100, sonify(z_history[:, :], 1, do_stereo=True))
    