import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

try:
    import cupy as cp
except ImportError:  # pragma: no cover - optional GPU dependency
    cp = None


def _to_numpy(array):
    """Convert CuPy arrays to NumPy arrays when needed."""

    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)

def predict(model, X):
    """
    Realiza el forward pass y devuelve las clases predichas.
    """
    A_out = model.forward(X)
    # np.argmax nos devuelve el índice (la clase) con la mayor probabilidad
    if cp is not None and isinstance(A_out, cp.ndarray):
        return cp.asnumpy(cp.argmax(A_out, axis=1))
    return np.argmax(A_out, axis=1)

def compute_accuracy(y_true, y_pred):
    """
    Calcula el Accuracy: (Predicciones Correctas) / (Total de Predicciones)
    """
    return np.mean(y_true == y_pred)

def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    Construye la Matriz de Confusión NxN.
    Las filas son las clases reales y las columnas las predichas.
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_class, pred_class in zip(y_true, y_pred):
        cm[true_class, pred_class] += 1
    return cm

def compute_f1_macro(cm):
    """
    Calcula el F1-Score Macro a partir de la matriz de confusión.
    F1 Macro es el promedio de los F1-Scores de cada clase.
    """
    num_classes = cm.shape[0]
    f1_scores = []
    
    for i in range(num_classes):
        # Verdaderos Positivos (Diagonal)
        TP = cm[i, i]
        # Falsos Positivos (Suma de la columna menos TP)
        FP = np.sum(cm[:, i]) - TP
        # Falsos Negativos (Suma de la fila menos TP)
        FN = np.sum(cm[i, :]) - TP
        
        # Precisión y Exhaustividad (Recall) con manejo de división por cero
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        
        # F1-Score para la clase i
        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
            
        f1_scores.append(f1)
        
    # El F1 Macro es el promedio simple de todos los F1 de clase
    return np.mean(f1_scores)


def build_metrics_report(model, X_train, y_train, X_val, y_val, num_classes, model_name="Modelo"):
    """Build a pandas report with loss, accuracy and macro F1 for train and validation."""

    y_pred_train = predict(model, X_train)
    y_pred_val = predict(model, X_val)

    y_train_one_hot = np.eye(num_classes)[_to_numpy(y_train).astype(int)]
    y_val_one_hot = np.eye(num_classes)[_to_numpy(y_val).astype(int)]

    loss_train = model.compute_loss(y_train_one_hot, model.forward(X_train))
    loss_val = model.compute_loss(y_val_one_hot, model.forward(X_val))

    acc_train = compute_accuracy(y_train, y_pred_train)
    acc_val = compute_accuracy(y_val, y_pred_val)

    cm_train = compute_confusion_matrix(y_train, y_pred_train, num_classes)
    cm_val = compute_confusion_matrix(y_val, y_pred_val, num_classes)

    f1_train = compute_f1_macro(cm_train)
    f1_val = compute_f1_macro(cm_val)

    return pd.DataFrame(
        {
            "Métrica": ["Cross-Entropy", "Accuracy", "F1-Score Macro"],
            "Train": [loss_train, acc_train, f1_train],
            "Validation": [loss_val, acc_val, f1_val],
            "Modelo": [model_name, model_name, model_name],
        }
    )


def compare_training_improvements(
    model_factory,
    experiments,
    X_train,
    y_train,
    X_val,
    y_val,
    reference_metrics,
    epochs=20,
    learning_rate=0.1,
    num_classes=47,
    reference_name="M0",
    return_histories=False,
):
    """Train one improvement at a time and compare each run against a precomputed baseline."""

    y_train_np = _to_numpy(y_train).astype(int)
    y_val_np = _to_numpy(y_val).astype(int)
    rows = []
    histories = []

    baseline_row = {
        "Experimento": reference_name,
        "Scheduler": reference_metrics.get("Scheduler", "-"),
        "Batch Size": reference_metrics.get("Batch Size", "Full"),
        "Optimizer": reference_metrics.get("Optimizer", "sgd"),
        "L2": reference_metrics.get("L2", 0.0),
        "Patience": reference_metrics.get("Patience", "-"),
        "Épocas ejecutadas": reference_metrics.get("Épocas ejecutadas", np.nan),
        "Tiempo (s)": reference_metrics.get("Tiempo (s)", np.nan),
        "Train Loss": reference_metrics.get("Train Loss", np.nan),
        "Val Loss": reference_metrics.get("Val Loss", np.nan),
        "Train Acc": reference_metrics.get("Train Acc", np.nan),
        "Val Acc": reference_metrics.get("Val Acc", np.nan),
        "Train F1": reference_metrics.get("Train F1", np.nan),
        "Val F1": reference_metrics.get("Val F1", np.nan),
    }
    rows.append(baseline_row)

    for experiment in experiments:
        model = model_factory()
        start_time = time.time()
        train_history, val_history = model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=experiment.get("epochs", epochs),
            learning_rate=experiment.get("learning_rate", learning_rate),
            scheduler_type=experiment.get("scheduler_type", "exponential"),
            decay_rate=experiment.get("decay_rate", 0.05),
            final_lr=experiment.get("final_lr", 1e-5),
            batch_size=experiment.get("batch_size"),
            patience=experiment.get("patience", 5),
            l2=experiment.get("l2", 0.0),
            optimizer=experiment.get("optimizer", "sgd"),
        )
        elapsed_time = time.time() - start_time

        y_pred_train = predict(model, X_train)
        y_pred_val = predict(model, X_val)

        cm_train = compute_confusion_matrix(y_train_np, y_pred_train, num_classes)
        cm_val = compute_confusion_matrix(y_val_np, y_pred_val, num_classes)

        rows.append(
            {
                "Experimento": experiment.get("name", "Experimento"),
                "Scheduler": experiment.get("scheduler_type", "exponential"),
                "Batch Size": "Full" if experiment.get("batch_size") is None else experiment.get("batch_size"),
                "Optimizer": experiment.get("optimizer", "sgd"),
                "L2": experiment.get("l2", 0.0),
                "Patience": experiment.get("patience", 5),
                "Épocas ejecutadas": len(train_history),
                "Tiempo (s)": elapsed_time,
                "Train Loss": float(train_history[-1]),
                "Val Loss": float(val_history[-1]),
                "Train Acc": compute_accuracy(y_train_np, y_pred_train),
                "Val Acc": compute_accuracy(y_val_np, y_pred_val),
                "Train F1": compute_f1_macro(cm_train),
                "Val F1": compute_f1_macro(cm_val),
            }
        )

        histories.append(
            {
                "name": experiment.get("name", "Experimento"),
                "train_loss": [float(value) for value in train_history],
                "val_loss": [float(value) for value in val_history],
            }
        )

    report = pd.DataFrame(rows)
    reference_time = baseline_row.get("Tiempo (s)")
    reference_val_loss = baseline_row.get("Val Loss")

    if pd.notna(reference_time):
        report["ΔTiempo vs M0 (%)"] = 100 * (report["Tiempo (s)"] - reference_time) / reference_time
    else:
        report["ΔTiempo vs M0 (%)"] = np.nan

    if pd.notna(reference_val_loss):
        report["ΔVal Loss vs M0 (%)"] = 100 * (report["Val Loss"] - reference_val_loss) / reference_val_loss
    else:
        report["ΔVal Loss vs M0 (%)"] = np.nan

    if return_histories:
        return report, histories

    return report
