import itertools
import time
import pandas as pd
from MLP_advanced import MLP_Advanced


def _format_architecture(architecture):
    return " -> ".join(map(str, architecture))


def _print_section(title, width=88):
    line = "=" * width
    print(f"\n{line}")
    print(title.center(width))
    print(line)

def grid_search(X_train, y_train, X_val, y_val, epochs=40, patience=5):

    param_grid = {
        'arquitectura': [
            [784, 256, 128, 47],       
            [784, 256, 128, 64, 47],
            [784, 512, 256, 128, 47]    
        ],
        'optimizer': ['adam'],
        'learning_rate': [0.001],
        'batch_size': [128, 256],
        'l2': [0.0, 0.001, 0.0001]
    }

    keys = param_grid.keys()
    values = (param_grid[key] for key in keys)
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    
    total_configs = len(configs)
    print(f"Iniciando Grid Search. Total de configuraciones a probar: {total_configs}")
    print("-" * 60)
    
    best_val_loss = float('inf')
    best_config = None
    hist_results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{total_configs}] Probando: Opt={config['optimizer']}, LR={config['learning_rate']}, L2={config['l2']}, Batch={config['batch_size']}, Arq={config['arquitectura']}")
        
        start_time = time.time()

        model = MLP_Advanced(config['arquitectura'])

        _, val_loss_history = model.train(
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
        
        time_model = time.time() - start_time

        min_val_loss = float(min(val_loss_history))
        
        hist_results.append({
            'config': config,
            'val_loss': float(min_val_loss),
            'tiempo': time_model
        })
        
        print(f"--> Val Loss: {min_val_loss:.4f} | Tiempo: {time_model:.1f} seg")

        if min_val_loss < best_val_loss:
            best_val_loss = min_val_loss
            best_config = config
            print("🏆 ¡NUEVO MEJOR MODELO!")


    _print_section("🥇 Configuración Ganadora M1")
    print(f"Mejor validation loss: {best_val_loss:.4f}")
    print("Configuración ganadora:")
    print(f"  - Arquitectura: {_format_architecture(best_config['arquitectura'])}")
    print(f"  - Optimizer: {best_config['optimizer']}")
    print(f"  - Learning rate: {best_config['learning_rate']}")
    print(f"  - Batch size: {best_config['batch_size']}")
    print(f"  - L2: {best_config['l2']}")

    _print_section("TOP 5 CONFIGURACIONES")

    df_results = pd.DataFrame([
        {
            'Ranking': i + 1,
            'Val Loss': r['val_loss'],
            'Tiempo (s)': r['tiempo'],
            'Optimizer': r['config']['optimizer'],
            'Learning Rate': r['config']['learning_rate'],
            'Batch Size': r['config']['batch_size'],
            'L2': r['config']['l2'],
            'Arquitectura': _format_architecture(r['config']['arquitectura'])
        }
        for r in hist_results
    ])
    
    df_top5 = df_results.nsmallest(5, 'Val Loss').reset_index(drop=True)
    df_top5['Ranking'] = range(1, len(df_top5) + 1)
    df_top5['Val Loss'] = df_top5['Val Loss'].map(lambda x: f"{x:.4f}")
    df_top5['Tiempo (s)'] = df_top5['Tiempo (s)'].map(lambda x: f"{x:.1f}")
    
    print(df_top5.to_string(index=False))
    
    _print_section("RESUMEN DE EJECUCIÓN")
    print(f"Configuraciones exploradas: {total_configs}")
    print(f"Tiempo total de búsqueda: {sum(r['tiempo'] for r in hist_results):.1f} seg")
    print("=" * 88)
        
    return best_config, hist_results, df_top5