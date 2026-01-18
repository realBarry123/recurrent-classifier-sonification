import torch
from model import RClassifier

def load(path):
    try: 
        state_dict, configs, epoch = torch.load(path)
    except: 
        raise ValueError("failed to load model from path " + path)
    model = RClassifier(**configs)
    model.load_state_dict(state_dict)
    return model, epoch

def save(model, epoch, path):
    torch.save([model.state_dict(), model.configs, epoch], path)


import matplotlib.pyplot as plt
from matplotlib import animation

def plot_history(history):
    fig, ax = plt.subplots()
    bar_plot = plt.bar(range(0, len(history[0])), history[0], width=0.8, bottom=None)

    def update(frame):
        for i, b in enumerate(bar_plot):
            b.set_height(frame[i])

    game = animation.FuncAnimation(fig, update, frames=history)
    plt.show()

def interpolate(history, scale_factor):
    T, V = history.shape
    history = history.permute(1, 0).view(1, V, T)
    history = torch.nn.functional.interpolate(history, scale_factor=scale_factor, mode='linear', align_corners=False)
    history = history.squeeze().permute(1, 0)
    return history
