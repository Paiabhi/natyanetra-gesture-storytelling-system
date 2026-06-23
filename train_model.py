import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class MudraMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(63,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,6)
        )

    def forward(self,x):

        return self.layers(x)


df = pd.read_csv(
    "landmarks.csv",
    header=None
)

X = df.iloc[:,:-1].values

y = df.iloc[:,-1].values

encoder = LabelEncoder()

y = encoder.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

model = MudraMLP()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

for epoch in range(100):

    optimizer.zero_grad()

    outputs = model(X_train)

    loss = criterion(
        outputs,
        y_train
    )

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:

        print(
            f"Epoch {epoch}: {loss.item()}"
        )

torch.save(
    model.state_dict(),
    "mudra_model.pt"
)

print("Model Saved")
