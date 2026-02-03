import torch
from tqdm import tqdm

def train(model, loader, optim, epoch=0, device="mps"):
    mse_loss = torch.nn.MSELoss()
    for x, y in tqdm(loader, desc=f"E{epoch} Train"):
        x = x.to(device)
        y = y.to(device)
        model.train()
        y = torch.nn.functional.one_hot(y, num_classes=10).float()
        y_hat = model(x)
        loss = mse_loss(y, y_hat)

        optim.zero_grad()
        loss.backward()
        optim.step()
        #print(loss.item())

def valid(model, loader, epoch=0, device="mps"):
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc=f"E{epoch} Valid"):
        x = x.to(device)
        y = y.to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
            y_hat = out.argmax(dim=1)
            correct += (y == y_hat).sum().item()
            total += y.shape[0]
    return correct / total