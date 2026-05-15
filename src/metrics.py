import numpy as np
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
