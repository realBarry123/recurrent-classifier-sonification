import torch
from torchsummary import summary
DEVICE = "mps"

class RClassifier(torch.nn.Module):
    def __init__(self, t, z_size, conv_channels, activation: str):
        super(RClassifier, self).__init__()

        self.T = t
        self.Z_SIZE = z_size
        self.CONV_CHANNELS = conv_channels
        self.activation = activation

        self.configs = {
            "T": self.T,
            "Z_SIZE": self.Z_SIZE,
            "CONV_CHANNELS": self.CONV_CHANNELS,
            "activation": self.activation
        }

        self.conv = torch.nn.Conv2d(
            in_channels=1,
            out_channels=self.CONV_CHANNELS,
            kernel_size=5
        )

        self.pool = torch.nn.MaxPool2d(
            kernel_size=4
        )

        self.z_linear = torch.nn.Linear(
            in_features=self.CONV_CHANNELS*36+self.Z_SIZE, # 576
            out_features=self.Z_SIZE
        )
        self.out_linear = torch.nn.Linear(
            in_features=self.Z_SIZE,
            out_features=10,
        )
        if activation == "softsign":
            self.r_activation = torch.nn.Softsign()
        else:
            self.r_activation = torch.nn.Tanh()
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)

        x = x.view(-1, self.CONV_CHANNELS*36)

        z = torch.zeros((x.shape[0], self.Z_SIZE)).to(DEVICE)

        z_history = []

        for i in range(self.T):
            z = torch.cat((x, z), dim=1)
            z = self.z_linear(z)
            z = self.r_activation(z) # bounded (or not) activation function
            z_history.append(z)
        z = self.out_linear(z)
        # z = self.softmax(z) no need, CELoss does it for you
        return z, z_history

if __name__ == "__main__":
    model = RClassifier(t=10, z_size=15, conv_channels=1, activation="softsign")
    #model(torch.randn(32, 1, 28, 28))
    summary(model, (1, 28, 28), batch_size=32)