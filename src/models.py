import numpy as np

def to_one_hot(y, num_classes=47):
    """
    Convierte un array unidimensional de etiquetas enteras a una matriz one-hot.
    """
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

class MLP:

    def __init__(self, layer_sizes):
        """
        layer_sizes: lista con la cantidad de nodos por capa.
        """

        self.num_layers = len(layer_sizes) # capas totales
        self.params = {} # diccionario para los pesos W y bias b

        for i in range(1, self.num_layers):
            # usamos inicialización de He (multiplicar por sqrt(2/n)) 
            # mantiene la varianza de las activaciones al propagar hacia adelante con ReLU, evitando que las activaciones se apaguen o exploten en capas profundas
            self.params[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_sizes[i]))
            
    def relu(self, Z):
        # devuelve el mismo valor si es positivo, o 0 si es negativo
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        # restamos np.max(Z) a todo el vector antes de hacer la exponencial. 
        # elegir c = max z centra los valores ≤ 0, garantizando que al menos un término sea 1 y los demás sean ≤ 1, evitando overflow y mejorando precisión numérica
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
   
     
    def forward(self, X):
        """
        X: matriz de entrada de tamaño (batch_size, input_size)
        """
        # guardamos los valores intermedios que vamos a usar en el backpropagation
        self.cache = {'A0': X}
        A = X
        
        # iteramos por las capas ocultas
        for i in range(1, self.num_layers - 1):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']

            Z = A @ W + b
            A = self.relu(Z)

            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
            
        # Capa de salida 
        L = self.num_layers - 1
        W_out = self.params[f'W{L}']
        b_out = self.params[f'b{L}']
        
        Z_out = A @ W_out + b_out
        A_out = self.softmax(Z_out)

        self.cache[f'Z{L}'] = Z_out
        self.cache[f'A{L}'] = A_out
        
        return A_out # probabilidades

    def compute_loss(self, Y_true_one_hot, Y_pred):
        """
        Calcula la función de costo Cross-Entropy.
        """
        m = Y_true_one_hot.shape[0]
        # evita que hagamos logaritmo de 0
        epsilon = 1e-15
        loss = -np.sum(Y_true_one_hot * np.log(Y_pred + epsilon)) / m
        return loss

    def backward(self, Y_true_one_hot):
        m = Y_true_one_hot.shape[0]
        self.grads = {}
        
        # capa de salida
        L = self.num_layers - 1
        
        # 1. Gradiente de la capa de salida (Softmax + Cross-Entropy)
        # Recuperamos la predicción del caché guardado en el forward pass
        A_out = self.cache[f'A{L}']
        # La derivada de Softmax + Cross-Entropy se simplifica a (Predicción - Real)
        dZ = A_out - Y_true_one_hot
        
        # Calculamos gradientes de pesos (dW) y sesgos (db) para la última capa
        A_prev = self.cache[f'A{L-1}']
        self.grads[f'dW{L}'] = (A_prev.T @ dZ) / m
        self.grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # 2. Propagación del error hacia atrás para las capas ocultas
        for i in range(L - 1, 0, -1):
            W_next = self.params[f'W{i+1}']
            
            # Derivada de ReLU: es 1 si Z > 0, y 0 si es negativo
            Z_curr = self.cache[f'Z{i}']
            d_relu = np.where(Z_curr > 0, 1.0, 0.0)
            
            # Regla de la cadena para calcular el error en la capa actual
            dZ = (dZ @ W_next.T) * d_relu
            
            # Guardamos los gradientes de esta capa
            A_prev = self.cache[f'A{i-1}']
            self.grads[f'dW{i}'] = (A_prev.T @ dZ) / m
            self.grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m

    def update_params(self, learning_rate):

        for i in range(1, self.num_layers):
            self.params[f'W{i}'] -= learning_rate * self.grads[f'dW{i}']
            self.params[f'b{i}'] -= learning_rate * self.grads[f'db{i}']

    def train(self, X_train, y_train, X_val, y_val,  epochs, learning_rate):

        num_classes = self.params[f'W{self.num_layers-1}'].shape[1]

        y_train_one_hot = to_one_hot(y_train, num_classes=num_classes)
        y_val_one_hot = to_one_hot(y_val, num_classes=num_classes)
        
        hist_loss = []
        hist_loss_val = []
        
        for epoch in range(epochs):

            Y_pred = self.forward(X_train)
            
            loss_train = self.compute_loss(y_train_one_hot, Y_pred)
            hist_loss.append(loss_train)
            
            self.backward(y_train_one_hot)
            
            self.update_params(learning_rate)

            # Forward pass (Validation): usamos la red actualizada pero no hacemos backward
            Y_pred_val = self.forward(X_val)
            loss_val = self.compute_loss(y_val_one_hot, Y_pred_val)
            hist_loss_val.append(loss_val)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Época {epoch:03d} | Train Loss: {loss_train:.4f} | Val Loss: {loss_val:.4f}")
                
        return hist_loss
    