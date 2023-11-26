import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

import Utils
from Freezer import Freezer


class AIModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, freezer: Freezer, hidden_size2: int = 0,
                 use_relu: bool = True,
                 device_str: str = "cpu"): #device_str: str = "cuda"
        super().__init__()
        self.device = torch.device(device_str)
        super().to(self.device)
        self._use_relu = use_relu
        self._two_hidden_layers: bool = hidden_size2 > 0
        self._linear1 = nn.Linear(input_size, hidden_size, device=self.device)
        out_layer_input_size = hidden_size
        if self._two_hidden_layers:
            self._linear2 = nn.Linear(hidden_size, hidden_size2)
            out_layer_input_size = hidden_size2
        self.linear_out = nn.Linear(out_layer_input_size, output_size, device=self.device)
        self.loos_fn: nn.MSELoss = nn.MSELoss()
        self.freezer = freezer
        self.error = 99999999.0
        print(
            f"Create AIModel with input size = {input_size}, hidden size = {hidden_size}, output size = {output_size}")

        # Try to load neural network
        self.load()

    def forward(self, x):
        if self._use_relu:
            x = F.relu(self._linear1(x))
        else:
            x = F.sigmoid(self._linear1(x))
        if self._two_hidden_layers:
            if self._use_relu:
                x = F.relu(self._linear2(x))
            else:
                x = F.sigmoid(self._linear2(x))
        x = self.linear_out(x)
        return x

    def save(self):
        file_path, err = self.freezer.get_model_data()
        print(f"Saving model into file {file_path} with error = {self.error}")
        torch.save(self.state_dict(), file_path)
        self.freezer.save_model_data(self.error)

    def load(self):
        file_path, error_saved = self.freezer.get_model_data()
        if Utils.is_file_exist(file_path):
            print(f"Loading model from file {file_path}")
            checkpoint = torch.load(file_path)
            self.load_state_dict(checkpoint)
            self.eval()
            self.error = error_saved

    def calculate_error(self, data_in: pd.DataFrame, data_target: pd.DataFrame):
        error_tmp = 0
        data_in = data_in.astype(dtype='float32')
        data_target = data_target.astype(dtype='float32')
        for input, target in zip(data_in.iloc, data_target.iloc):
            input_tensor = torch.tensor(input.values, device=self.device)
            target_tensor = torch.tensor(target.values, device=self.device)
            output = self(input_tensor)
            loos: torch.Tensor = self.loos_fn(output, target_tensor)
            error_tmp += loos.item()
        error_tmp = error_tmp / len(input.index)
        return error_tmp

    def train_me(self, data_in: pd.DataFrame, data_target: pd.DataFrame, lr: float, max_error: float,
              max_epoch: int = 10000):
        assert len(data_in.index) == len(data_target.index)
        super().train(True)
        data_in = data_in.astype(dtype='float32')
        data_target = data_target.astype(dtype='float32')
        print(f"Starting training with data in head = \n{data_in.head()} , data target head = \n{data_target.head()}")
        print(
            f"Starting training with data in describe = \n{data_in.describe()} , data target describe = \n{data_target.describe()}")
        optimizer: optim.Adam = optim.Adam(self.parameters(), lr=lr)
        optimizer.zero_grad()

        error = 10000000
        epoch = 0
        while error > max_error and epoch < max_epoch:
            error = 0
            for input, target in zip(data_in.iloc, data_target.iloc):
                input_tensor = torch.tensor(input.values, device=self.device)
                target_tensor = torch.tensor(target.values, device=self.device)
                optimizer.zero_grad()
                output = self(input_tensor)
                loos: torch.Tensor = self.loos_fn(output, target_tensor)
                error += loos.item()
                loos.backward()
                optimizer.step()
            epoch += 1
            error /= len(data_in.index)  # is this correct len?
            self.save_model_if_needed(error)
            if epoch % 10 == 0:
                print(f"epoch = {epoch} Error = {error}", flush=True)

            # Set the model to evaluation mode, disabling dropout and using population
            # statistics for batch normalization.
            # self().eval()

    def save_model_if_needed(self, training_error: float):
        if training_error < self.error:
            self.error = training_error
            self.save()
