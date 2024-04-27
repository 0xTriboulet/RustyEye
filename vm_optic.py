import copy

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
import torch.optim as optim


class RustyOptic(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.5):
        super(RustyOptic, self).__init__()
        self.input_layer = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer after the first linear transformation
        self.hidden_layer1 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer after the second linear transformation
        self.output_layer = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.input_layer(x))
        x = self.dropout1(x)  # Apply dropout after activation
        x = self.relu(self.hidden_layer1(x))
        x = self.dropout2(x)  # Apply dropout after activation
        x = self.output_layer(x)
        return self.sigmoid(x)


if __name__ == "__main__":
    data = pd.read_csv('./process_info_aggregate.csv')

    X = data.iloc[:, 0:5].values
    y = data.iloc[:, 5:].values

    # convert pandas DataFrame (X) and numpy array (y) into PyTorch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.20, shuffle=True)

    n_epochs = 2000
    batch_size = 1

    loss_fn = nn.BCELoss()
    model = RustyOptic(input_dim=X.shape[1], dropout_rate=0.40)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_acc = 0
    best_weights = None
    best_loss = 0

    # Calculate the number of batches
    n_batches = int(np.ceil(len(X_train) / batch_size))

    for epoch in range(n_epochs):

        model.train()
        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            ce = loss_fn(y_pred, y_test)
            acc = ((y_pred > 0.5).float() == y_test).float().mean()

        ce = float(ce)
        acc = float(acc)

        if acc > best_acc or (acc == 1.0 and ce < best_loss):
            best_acc = acc
            best_loss = ce
            best_weights = copy.deepcopy(model.state_dict())
            print(f"Epoch {epoch} validation: BCELoss={ce}, Accuracy={acc}, Best Accuracy={best_acc}")

        model.load_state_dict(best_weights)

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save('vm_detect_model.pt')  # Save


