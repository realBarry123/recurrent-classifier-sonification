import torch
from torchvision import datasets, transforms
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader
from model import RClassifier
from utils import save, load
from train import train, valid
from sonification import sonify
from scipy.io import wavfile

DEVICE = "mps"
REPLICATIONS = 3

noise_functions = [
    lambda x: x, 
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.2, clip=True),
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.5, clip=True), 
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

    for j in range(REPLICATIONS):
        print(f"Model {str(i)}-{str(j)}:")

        model = RClassifier(
            t=16, 
            z_size=33, 
            conv_channels=2, 
            activation="softsign"
        ).to(DEVICE)

        optim = torch.optim.Adam(params=model.parameters(), lr=0.0005)
        best_loss = 10 # 10: the worst possible loss ever

        for epoch in range(10):
            train(model, train_loader, optim, epoch=epoch, do_tqdm=False)
            loss, accuracy = valid(model, valid_loader, epoch=epoch, do_tqdm=False)
            print(loss, accuracy)
            if loss < best_loss:
                best_loss = loss
                # Save our best model
                save(model, epoch, f"experiments/noise/{str(i)}-{str(j)}.pt")

        # Load back our best model
        model, _ = load(f"experiments/noise/{str(i)}-{str(j)}.pt")
        model = model.to(DEVICE)

        with torch.no_grad():
            model.eval()
            _ = model(valid_set[0][0].to(DEVICE))
            z_history = model.get_history(layer="z")
            z_history = z_history.squeeze(1)
            z_history = (z_history - torch.min(z_history)) / (torch.max(z_history) - torch.min(z_history)) * 2000 + 50
            wavfile.write(f"experiments/noise/{str(i)}-{str(j)}.wav", 44100, sonify(z_history[:, :], 1, do_stereo=True))
    