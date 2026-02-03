import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import RClassifier
from utils import save, load

DEVICE = "mps"
PATH = "model.pt"

transform = transforms.ToTensor()

train_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
valid_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=32)

mse_loss = torch.nn.MSELoss()

def train(model, epoch):
    for x, y in tqdm(train_loader, desc=f"E{epoch} Train"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        model.train()
        y = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_hat = model(x)
        loss = mse_loss(y, y_hat)

        optim.zero_grad()
        loss.backward()
        optim.step()
        #print(loss.item())

def valid(model, epoch):
    correct = 0
    total = 0
    for x, y in tqdm(valid_loader, desc=f"E{epoch} Valid"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        model.eval()
        with torch.no_grad():
            out = model(x)
            y_hat = out.argmax(dim=1)
            correct += (y == y_hat).sum().item()
            total += y.shape[0]
    return correct / total

try:
    model, epoch = load(PATH)
    model = model.to(DEVICE)
except ValueError:
    model = RClassifier(
        t=10, 
        z_size=33, 
        conv_channels=2, 
        activation="softsign"
    ).to(DEVICE)
    epoch = 0

optim = torch.optim.Adam(params=model.parameters(), lr=0.0005)

for i in range(6):
    train(model, epoch)
    print(valid(model, epoch))
    epoch += 1
    save(model, epoch, PATH)