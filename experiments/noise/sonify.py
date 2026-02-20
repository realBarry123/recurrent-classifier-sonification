import torch
from torchvision import datasets, transforms
import torchvision.transforms.v2 as transforms_v2
from torch.utils.data import DataLoader
from utils import load
from sonification import sonify
from scipy.io import wavfile

REPLICATIONS = 3
DEVICE = "mps"

transform = transforms.ToTensor()

noise_functions = [
    lambda x: x, 
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.2, clip=True),
    transforms_v2.GaussianNoise(mean=0.0, sigma=0.5, clip=True), 
]


for i in range(len(noise_functions)):
    valid_set = datasets.FashionMNIST(
        root=".", 
        train=False, 
        download=True, 
        transform=transforms.Compose([transform, noise_functions[i]])
    )

    valid_loader = DataLoader(dataset=valid_set, batch_size=32) 

    for j in range(REPLICATIONS):
    
        model, _ = load(f"experiments/noise/{str(i)}-{str(j)}.pt")
        model = model.to(DEVICE)

        with torch.no_grad():
            model.eval()
            _ = model(valid_set[0][0].to(DEVICE))
            z_history = model.get_history(layer="z")
            z_history = z_history.squeeze(1)
            z_history = (z_history - torch.min(z_history)) / (torch.max(z_history) - torch.min(z_history)) * 2000 + 50
            wavfile.write(f"experiments/noise/{str(i)}-{str(j)}.wav", 44100, sonify(z_history[:, :], 1, do_stereo=True))
