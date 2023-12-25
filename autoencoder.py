from torchvision import datasets
from torchvision import transforms
import torch
import numpy as np
# import pandas as pd

class AE(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def learn(data_loader, input_size, device, epochs=80):
    model = AE(input_size=input_size).double().to(device)
    model.train()

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    loss_df = []

    for epoch in range(epochs):
        losses = []

        for items in data_loader:
            items = items[0].cuda()
            optimizer.zero_grad()

            reconstructed = model(items)
            loss = loss_function(reconstructed, items)

            loss.backward()

            optimizer.step()

            losses.append(loss.detach().cpu().numpy().item())

        losses = np.array(losses)

        loss_df.append({
            'epoch': epoch + 1,
            'loss': losses.mean()
        })

        print(f'{epoch + 1:03}, {losses.mean():.5f}')

    #loss_df = pd.DataFrame(loss_df)
    #loss_df.index = loss_df['epoch']
    #loss_df = loss_df.drop(columns=['epoch'])

    return model

def predict(m, y_true, device):
    y_pred = m(torch.from_numpy(y_true).to(device)).detach().cpu()
    y_true = torch.from_numpy(y_true)
    return y_true, y_pred
