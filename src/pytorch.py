import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time

# Configurar el dispositivo para usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Corriendo en: {device}")

class MLP_PyTorch(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP_PyTorch, self).__init__()
        layers = []
        
        # Construimos las capas dinámicamente según la lista de arquitectura
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            # Aplicamos ReLU en todas las capas ocultas (todas menos la última)
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
                
        # nn.Sequential empaqueta todo en un flujo secuencial hacia adelante
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Retorna directamente los logits (sin Softmax al final)
        return self.network(x)


def train_m2_pytorch(X_train, y_train, X_val, y_val, config, epochs=100):
    # Transformamos los arrays de NumPy a Tensores de PyTorch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # Creamos los DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = MLP_PyTorch(config['arquitectura']).to(device)

    criterion = nn.CrossEntropyLoss()
    

    if config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], weight_decay=config['l2'])
        
    # 6. Configuramos el Scheduler Exponencial igual que en M1
    # decay_rate = 0.05 -> factor multiplicativo = exp(-0.05) ≈ 0.9512
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=np.exp(-0.05))
    
    hist_loss_train, hist_loss_val = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    
    print("\nIniciando el entrenamiento de M2...")
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Pasada hacia adelante
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Limpieza de gradientes, retropropagación y actualización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
            
        loss_epoch_train = running_loss / len(X_train)
        hist_loss_train.append(loss_epoch_train)
        
        # --- Fase de Validación ---
        model.eval() # Activa el modo evaluación (desactiva gradientes internamente)
        val_running_loss = 0.0
        
        with torch.no_grad(): # Bloque que ahorra memoria al no calcular gradientes
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_running_loss += loss.item() * batch_X.size(0)
                
        loss_epoch_val = val_running_loss / len(X_val)
        hist_loss_val.append(loss_epoch_val)
        
        # Actualizamos la tasa de aprendizaje al final de la época
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

class MLP_Avanzado_PyTorch(nn.Module):
    def __init__(self, layer_sizes, activacion='relu', dropout_rate=0.0):
        super(MLP_Avanzado_PyTorch, self).__init__()
        layers = []
        
        for i in range(len(layer_sizes) - 1):
            # 1. Capa Lineal (Pesos y Sesgos)
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Si NO es la capa de salida, agregamos Activación y Dropout
            if i < len(layer_sizes) - 2:
                # 2. Selección de la función de activación
                if activacion == 'relu':
                    layers.append(nn.ReLU())
                elif activacion == 'leaky_relu':
                    layers.append(nn.LeakyReLU(negative_slope=0.01))
                elif activacion == 'silu': # También conocida como Swish
                    layers.append(nn.SiLU())
                elif activacion == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Activación {activacion} no soportada.")
                
                # 3. Capa de Dropout (Solo se aplica si la tasa es mayor a 0)
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))
                    
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

import itertools
import time

def grid_search_pytorch_m3(X_train, y_train, X_val, y_val, base_config):
    # Grilla de exploración para M3
    param_grid_m3 = {
        'arquitectura': [
            [784, 256, 128, 47],        # Ancha (2 ocultas)
            [784, 256, 128, 64, 47],    # Más profunda (3 ocultas)
            [784, 512, 256, 128, 47]    # Muy profunda y ancha
        ],
        'activacion': ['leaky_relu', 'silu', 'gelu'],
        'dropout_rate': [0.1, 0.3]      # Probamos Dropout suave y medio
    }
    
    keys = param_grid_m3.keys()
    values = (param_grid_m3[key] for key in keys)
    combinaciones = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    mejor_val_loss = float('inf')
    mejor_config_m3 = None
    resultados_m3 = []
    
    print(f"Iniciando búsqueda de M3 con {len(combinaciones)} combinaciones (GPU habilitada si existe)...")
    
    for i, config in enumerate(combinaciones):
        print(f"\n[{i+1}/{len(combinaciones)}] Probando: Arq={config['arquitectura']}, Act={config['activacion']}, Drop={config['dropout_rate']}")
        
        # Mezclamos la configuración base (batch, optimizer) con la nueva que estamos probando
        config_actual = base_config.copy()
        config_actual.update(config)
        
        t_start = time.time()
        
        # OJO: Asumiendo que tu función de entrenamiento ahora instancia MLP_Avanzado_PyTorch
        # pasándole config_actual['activacion'] y config_actual['dropout_rate']
        modelo, train_loss, val_loss = train_m2_pytorch( # o como se llame tu función
            X_train, y_train, X_val, y_val, config=config_actual, epochs=40
        )
        
        t_total = time.time() - t_start
        min_val_loss = min(val_loss)
        
        print(f"--> Val Loss: {min_val_loss:.4f} | Tiempo: {t_total:.1f}s")
        
        if min_val_loss < mejor_val_loss:
            mejor_val_loss = min_val_loss
            mejor_config_m3 = config_actual
            print("⭐ ¡NUEVO MEJOR MODELO (M3)!")
            
    print("\n🏆 CONFIGURACIÓN GANADORA M3 🏆")
    for k, v in mejor_config_m3.items():
        print(f"  {k}: {v}")
        
    return mejor_config_m3

# Ejecución: Asume que M1_config tiene {'optimizer': 'adam', 'learning_rate': 0.001, 'batch_size': 256, 'l2': 0.0}
# M3_config = grid_search_pytorch_m3(X_train_flat, y_train, X_val_flat, y_val, M1_config)