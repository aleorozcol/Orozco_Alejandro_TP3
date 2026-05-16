import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def predict(model, X):
    """
    Realiza el forward pass y devuelve las clases predichas.
    """
    A_out = model.forward(X)
    # np.argmax nos devuelve el índice (la clase) con la mayor probabilidad
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

    y_train_one_hot = np.eye(num_classes)[y_train.astype(int)]
    y_val_one_hot = np.eye(num_classes)[y_val.astype(int)]

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
