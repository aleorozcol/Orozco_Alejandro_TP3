import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import itertools
import numpy as np
import time

try:
    import cupy as cp
except ImportError:  # pragma: no cover - CuPy is optional outside the notebook runtime
    cp = None

# Configurar el dispositivo para usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Corriendo en: {device}")


def evaluate_model_numpy(model, X_test, y_test, num_classes=47):
    """Evalúa un modelo implementado en NumPy o CuPy (M0, M1)."""

    usa_cupy = cp is not None and isinstance(model.params["W1"], cp.ndarray)

    if usa_cupy:
        X_eval = cp.asarray(X_test)
        y_eval = cp.asarray(y_test)
        probs = model.forward(X_eval)
        y_pred = cp.argmax(probs, axis=1)
        acc = float(cp.mean(y_eval == y_pred).get())
        y_pred_np = cp.asnumpy(y_pred)
        y_true_np = cp.asnumpy(y_eval)
    else:
        probs = model.forward(X_test)
        y_pred = np.argmax(probs, axis=1)
        acc = float(np.mean(y_test == y_pred))
        y_pred_np = y_pred
        y_true_np = y_test

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true_np, y_pred_np):
        cm[t, p] += 1

    f1_scores = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return acc, float(np.mean(f1_scores))


def evaluate_model_pytorch(model, X_test, y_test, num_classes=47):
    """Evalúa un model implementado en PyTorch (M2, M3)."""

    model.eval()
    model_device = next(model.parameters()).device

    X_t = torch.tensor(X_test, dtype=torch.float32).to(model_device)

    with torch.no_grad():
        outputs = model(X_t)
        _, predicciones = torch.max(outputs, 1)

    y_pred = predicciones.cpu().numpy()
    acc = float(np.mean(y_test == y_pred))

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1

    f1_scores = []
    for i in range(num_classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    return acc, float(np.mean(f1_scores))

class MLP_PyTorch(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP_PyTorch, self).__init__()
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # ReLU en todas las capas ocultas
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                
        # nn.Sequential empaqueta todo en un flujo secuencial hacia adelante
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # retorna directamente los logits (sin Softmax al final)
        return self.network(x)

def train_pytorch(X_train, y_train, X_val, y_val, config, epochs=100, patience=10):
    
    # transformamos los arrays de NumPy a Tensores de PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # TensorDataset: Empaqueta las imágenes y sus etiquetas juntas para que no se desordenen
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    # DataLoader: hace un np.random.permutation y un bucle for saltando con slicing para armar los mini-batches
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    if 'activacion' in config and 'dropout_rate' in config:
        # para M3
        model = MLP_Advanced_PyTorch(
            config['arquitectura'], 
            activacion=config['activacion'], 
            dropout_rate=config['dropout_rate']
        ).to(device)
    else:
        # para M2
        # .to(device) envía todas las matrices de pesos a la placa de video (si hay una) o al procesador
        model = MLP_PyTorch(config['arquitectura']).to(device)

    # funcion de error
    # ya lleva el Softmax incorporado adentro mediante una fórmula matemática más estable (LogSoftmax)
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2'])
        
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(-0.05))
    
    hist_loss_train, hist_loss_val = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print(f"\nIniciando el entrenamiento... (Modelo: {model.__class__.__name__})")
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # forward
            outputs = model(batch_X)
            # compute loss
            loss = criterion(outputs, batch_y)
            
            # Limpieza de gradientes
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update_params
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
        loss_epoch_train = running_loss / len(X_train)
        hist_loss_train.append(loss_epoch_train)
        
        # validation
        model.eval() # apaga el Dropout para que la red use el 100% de sus neuronas
        val_running_loss = 0.0
        
        with torch.no_grad(): 
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_running_loss += loss.item() * batch_X.size(0)
                
        loss_epoch_val = val_running_loss / len(X_val)
        hist_loss_val.append(loss_epoch_val)
        
        # actualizamos la tasa de aprendizaje
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Época {epoch:03d} | Train Loss: {loss_epoch_train:.4f} | Val Loss: {loss_epoch_val:.4f}")
            
        # Early Stopping
        if loss_epoch_val < best_val_loss:
            best_val_loss = loss_epoch_val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"¡Early stopping activado en la época {epoch}!")
                break
                
    return model, hist_loss_train, hist_loss_val

class MLP_Advanced_PyTorch(nn.Module):
    def __init__(self, layer_sizes, activacion='relu', dropout_rate=0.0):
        super(MLP_Advanced_PyTorch, self).__init__()
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            # Capa Lineal (Pesos y Sesgos)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # si NO es la capa de salida, agregamos Activación y Dropout
            if i < len(layer_sizes) - 2:

                if activacion == 'relu':
                    layers.append(nn.ReLU())
                elif activacion == 'leaky_relu':
                    layers.append(nn.LeakyReLU(negative_slope=0.01))
                elif activacion == 'silu': 
                    layers.append(nn.SiLU())
                elif activacion == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Activación {activacion} no soportada.")

                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))
                    
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def grid_search_pytorch_m3(X_train, y_train, X_val, y_val, base_config):
    param_grid_m3 = {
        'arquitectura': [
            [784, 256, 128, 64, 47],           
            [784, 512, 256, 128, 47]    
        ],
        'activacion': ['leaky_relu', 'silu', 'gelu'],
        'dropout_rate': [0.2, 0.4],
        'l2': [0.0, 0.0001]      
    }
    
    keys = param_grid_m3.keys()
    values = (param_grid_m3[key] for key in keys)
    comb = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    best_val_loss = float('inf')
    best_config_m3 = None
    
    print(f"Iniciando búsqueda de M3 con {len(comb)} combinaciones ...")
    
    for i, config in enumerate(comb):
        print(f"\n[{i+1}/{len(comb)}] Probando: Arq={config['arquitectura']}, Act={config['activacion']}, Drop={config['dropout_rate']}")
        
        config_actual = base_config.copy()
        config_actual.update(config)
        
        t_start = time.time()
        
        modelo, train_loss, val_loss = train_pytorch( 
            X_train, y_train, X_val, y_val, 
            config=config_actual, 
            epochs=40, 
            patience=5  # para acelerar el grid search
        )
        
        t_total = time.time() - t_start
        min_val_loss = min(val_loss)
        
        print(f"--> Val Loss: {min_val_loss:.4f} | Tiempo: {t_total:.1f}s")
        
        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_config_m3 = config_actual
            print("🏆 ¡NUEVO MEJOR MODELO!")
            
        del modelo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print("\n🥇 Configuración Ganadora M3")
    for k, v in best_config_m3.items():
        print(f"  {k}: {v}")
        
    return best_config_m3
