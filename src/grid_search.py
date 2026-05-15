import itertools
import time
import pandas as pd
from MLP_advanced import MLP_Advanced


def _as_options(value):
    if isinstance(value, (list, tuple, range, set)):
        return list(value)
    return [value]

def grid_search(X_train, y_train, X_val, y_val, epochs=40, patience=5):
    # 1. Definimos el diccionario con las opciones a explorar
    param_grid = {
        'arquitectura': [
            [784, 256, 128, 47],       # Más ancha (más nodos)
            [784, 128, 64, 32, 47]     # Más profunda (más capas)
        ],
        'optimizer': ['sgd', 'adam'],
        'learning_rate': 0.01,
        'batch_size': [128, 256],
        'l2': [0.0, 0.01]       # Con y sin regularización
    }

    # 2. Magia de itertools para generar todas las combinaciones posibles
    keys = param_grid.keys()
    values = (_as_options(param_grid[key]) for key in keys)
    combinaciones = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    total_configs = len(combinaciones)
    print(f"Iniciando Grid Search. Total de configuraciones a probar: {total_configs}")
    print("-" * 60)
    
    mejor_val_loss = float('inf')
    mejor_config = None
    historial_resultados = []

    # 3. Iteramos sobre cada configuración
    for i, config in enumerate(combinaciones):
        print(f"\n[{i+1}/{total_configs}] Probando: Opt={config['optimizer']}, LR={config['learning_rate']}, L2={config['l2']}, Batch={config['batch_size']}, Arq={config['arquitectura']}")
        
        start_time = time.time()
        
        # Instanciamos un modelo fresco para esta prueba
        modelo = MLP_Advanced(config['arquitectura'])
        
        # Entrenamos con los parámetros de esta configuración
        _, val_loss_history = modelo.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size'],
            optimizer=config['optimizer'],
            l2=config['l2'],
            patience=patience,
            scheduler_type='exponential',
            decay_rate=0.05,
            final_lr=1e-5
        )
        
        tiempo_ejecucion = time.time() - start_time
        
        # El menor error logrado en validación es nuestro indicador de éxito
        min_val_loss = float(min(val_loss_history))
        
        historial_resultados.append({
            'config': config,
            'val_loss': float(min_val_loss),
            'tiempo': tiempo_ejecucion
        })
        
        print(f"--> Val Loss: {min_val_loss:.4f} | Tiempo: {tiempo_ejecucion:.1f} seg")
        
        # Actualizamos el podio si encontramos un modelo mejor
        if min_val_loss < mejor_val_loss:
            mejor_val_loss = min_val_loss
            mejor_config = config
            print("⭐ ¡NUEVO MEJOR MODELO (M1) ENCONTRADO!")


    print("\n" + "="*100)
    print("🏆 BÚSQUEDA FINALIZADA 🏆")
    print("="*100)
    print(f"Mejor Validation Loss: {mejor_val_loss:.4f}")
    print("\nConfiguración ganadora (M1):")
    for k, v in mejor_config.items():
        print(f"  - {k}: {v}")
    
    # ========================================================================
    # RANKING TOP-5
    # ========================================================================
    print("\n" + "="*100)
    print("TOP-5 CONFIGURACIONES")
    print("="*100)
    
    # Convertir a DataFrame para análisis más fácil
    df_resultados = pd.DataFrame([
        {
            'Ranking': i+1,
            'Val Loss': r['val_loss'],
            'Tiempo (s)': r['tiempo'],
            'Optimizer': r['config']['optimizer'],
            'Learning Rate': r['config']['learning_rate'],
            'Batch Size': r['config']['batch_size'],
            'L2': r['config']['l2'],
            'Arquitectura': str(r['config']['arquitectura'])
        }
        for r in historial_resultados
    ])
    
    # Ordenar por val_loss (ascendente)
    df_top5 = df_resultados.nsmallest(5, 'Val Loss').reset_index(drop=True)
    df_top5['Ranking'] = range(1, len(df_top5) + 1)
    
    print(df_top5.to_string(index=False))
    
    print("\n" + "="*100)
    print(f"Total de configuraciones exploradas: {total_configs}")
    print(f"Tiempo total de búsqueda: {sum([r['tiempo'] for r in historial_resultados]):.1f} seg")
    print("="*100)
        
    return mejor_config, historial_resultados, df_top5