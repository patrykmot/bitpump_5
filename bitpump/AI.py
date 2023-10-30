import bitpump as bit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd


class AIModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #       x = x.to(torch.float32)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        checkpoint = torch.load(file_name)
        # self.load_state_dict(checkpoint['model_state_dict'])
        self.load_state_dict(checkpoint)
        self.eval()


def train(model: AIModel, data_in: pd.DataFrame, data_target: pd.DataFrame, lr: float, max_error: float):
    optimizer: optim.Adam = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    loos_fn: nn.MSELoss = nn.MSELoss()

    error = 10000000
    epoch = 0
    while error > max_error and epoch < 10000:
        error = 0
        for input, target in zip(data_in.iloc, data_target.iloc):
            input_tensor = torch.tensor(input.values)
            target_tensor = torch.tensor(target.values)
            output = model(input_tensor)
            loos: torch.Tensor = loos_fn(output, target_tensor)
            error += loos.item()
            loos.backward()
            optimizer.step()
        epoch += 1
        error /= len(data_in)
        if epoch % 10 == 0:
            print(f"epoch = {epoch} Error = {error}", flush=True)



