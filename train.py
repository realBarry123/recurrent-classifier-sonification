import torch
from tqdm import tqdm

def train(model, loader, optim, epoch=0, device="mps", do_tqdm=True):
    mse_loss = torch.nn.MSELoss()

    iterator = loader
    if do_tqdm:
        iterator = tqdm(iterator, desc=f"E{epoch} Train")

    for x, y in iterator:
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

def valid(model, loader, epoch=0, device="mps", do_tqdm=True):
    correct = 0
    total = 0

    iterator = loader
    if do_tqdm:
        iterator = tqdm(iterator, desc=f"E{epoch} Valid")

    for x, y in iterator:
        x = x.to(device)
        y = y.to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
            y_hat = out.argmax(dim=1)
            correct += (y == y_hat).sum().item()
            total += y.shape[0]
    return correct / total