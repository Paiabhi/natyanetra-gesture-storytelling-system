import torch
import torch.nn as nn
import numpy as np

MUDRAS = [
    "Pataka",
    "Tripataka",
    "Ardhapataka",
    "Kartarimukha",
    "Mayura",
    "Ardhachandra"
]

class MudraMLP(nn.Module):
    def __init__(self, input_size=63, num_classes=6):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class GestureClassifier:

    def __init__(self, model_path):

        self.device = torch.device("cpu")

        self.model = MudraMLP()

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.eval()

    def predict(self, landmarks):

        data = np.array(
            landmarks,
            dtype=np.float32
        )

        tensor = torch.tensor(
            data,
            dtype=torch.float32
        ).unsqueeze(0)

        with torch.no_grad():

            output = self.model(tensor)

            pred = torch.argmax(
                output,
                dim=1
            ).item()

        return MUDRAS[pred]
