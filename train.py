import torch, tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import RClassifier

DEVICE = "mps"
PATH = "model.pt"

transform = transforms.ToTensor()

train_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
valid_set = datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)

train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=32)

def load(path):
    try: 
        state_dict, configs, epoch = torch.load(path)
    except: 
        raise ValueError("failed to load model from path " + path)
    model = RClassifier(**configs)
    model.load_state_dict(state_dict)
    return model, epoch

def save(model: torch.nn.Module, epoch: int, path: str):
    torch.save([model.state_dict(), model.configs, epoch], path)

mse_loss = torch.nn.MSELoss()

def train(model):
    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        model.train()
        y = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_hat, _ = model(x)
        loss = mse_loss(y, y_hat)

        optim.zero_grad()
        loss.backward()
        optim.step()
        #print(loss.item())

def valid(model):
    correct = 0
    total = 0
    for x, y in valid_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        model.eval()
        out, _ = model(x)
        y_hat = out.argmax(dim=1)
        correct += (y == y_hat).sum().item()
        total += y.shape[0]
    return correct / total

try:
    model = load(PATH).to(DEVICE)
except ValueError:
    model = RClassifier(t=10, z_size=32, conv_channels=2, activation="softsign").to(DEVICE)

optim = torch.optim.Adam(params=model.parameters(), lr=0.0005)

for i in range(6):
    train(model)
    print(valid(model))
    save(model, 0, PATH)

torch.save(model.state_dict(), PATH)