from typing import List, Tuple, Dict, Optional
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    if name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.01)
    if name in ('silu', 'swish'):
        return nn.SiLU()
    if name == 'gelu':
        return nn.GELU()
    return nn.ReLU()


def build_mlp_torch(layer_sizes: List[int], activation: str = 'relu', dropout: float = 0.0) -> nn.Module:
    """Build a simple fully-connected MLP in PyTorch.

    layer_sizes: list like [784, 256, 128, 47]
    activation: one of 'relu','leakyrelu','silu','gelu'
    dropout: dropout probability (0 means no dropout)
    """
    layers: List[nn.Module] = []
    act = _get_activation(activation)
    for i in range(1, len(layer_sizes)):
        layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        if i < len(layer_sizes) - 1:
            layers.append(copy.deepcopy(act))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


class MLPTorch(nn.Module):
    def __init__(self, layer_sizes: List[int], activation: str = 'relu', dropout: float = 0.0):
        super().__init__()
        self.net = build_mlp_torch(layer_sizes, activation, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _to_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def get_dataloaders(X_train, y_train, X_val, y_val, batch_size: int = 128):
    train_loader = _to_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, batch_size, shuffle=False)
    return train_loader, val_loader


def train_torch(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    learning_rate: float = 0.01,
    batch_size: int = 128,
    optimizer: str = 'sgd',
    l2: float = 0.0,
    scheduler_type: str = 'exponential',
    decay_rate: float = 0.05,
    final_lr: float = 1e-5,
    patience: int = 10,
    device: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    train_loader = _to_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _to_loader(X_val, y_val, batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    if optimizer.lower() == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2)

    if scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda ep: math.exp(-decay_rate * ep))
    elif scheduler_type == 'linear':
        def linear_lambda(ep, T=epochs):
            return max(final_lr / learning_rate, 1.0 - (ep / max(1, T - 1)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda ep: linear_lambda(ep, epochs))
    else:
        scheduler = None

    best_val = float('inf')
    epochs_no_improve = 0
    best_state = None

    train_losses, val_losses = [], []

    for ep in range(epochs):
        model.train()
        running = 0.0
        count = 0
        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += loss.item() * Xb.size(0)
            count += Xb.size(0)
        epoch_train_loss = running / max(1, count)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_v = 0.0
        count_v = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device)
                yv = yv.to(device)
                logits = model(Xv)
                loss_v = criterion(logits, yv)
                running_v += loss_v.item() * Xv.size(0)
                count_v += Xv.size(0)
        epoch_val_loss = running_v / max(1, count_v)
        val_losses.append(epoch_val_loss)

        if scheduler is not None:
            scheduler.step()

        # early stopping
        if epoch_val_loss < best_val - 1e-8:
            best_val = epoch_val_loss
            epochs_no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses


def predict_torch(model: nn.Module, X: np.ndarray, batch_size: int = 256, device: Optional[str] = None) -> np.ndarray:
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    loader = _to_loader(X, np.zeros(X.shape[0], dtype=np.int64), batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0)


def evaluate_noise_robustness(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    noise_levels: List[float] = [0.0, 0.01, 0.02, 0.05],
    batch_size: int = 256,
) -> Dict[float, float]:
    """Return accuracy for each gaussian noise std fraction of max (255).
    noise_levels are relative std (e.g., 0.01 => std = 0.01*255)
    """
    results = {}
    for nl in noise_levels:
        if nl <= 0:
            Xn = X_test.copy()
        else:
            std = nl * 255.0
            noise = np.random.normal(0, std, size=X_test.shape)
            Xn = X_test.astype(np.float32) + noise
            Xn = np.clip(Xn, 0, 255).astype(np.uint8)

        Xn_flat = Xn.reshape(Xn.shape[0], -1).astype(np.float32)
        preds = predict_torch(model, Xn_flat, batch_size=batch_size)
        acc = float((preds == y_test).mean())
        results[nl] = acc
    return results


def add_noise_numpy(X: np.ndarray, std_fraction: float) -> np.ndarray:
    if std_fraction <= 0:
        return X.copy()
    std = std_fraction * 255.0
    noise = np.random.normal(0, std, size=X.shape)
    Xn = X.astype(np.float32) + noise
    return np.clip(Xn, 0, 255).astype(np.uint8)
