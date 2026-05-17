import cupy as np

def to_one_hot(y, num_classes=47):
    """
    Convierte un array unidimensional de etiquetas enteras a una matriz one-hot.
    """
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

class MLP_Advanced:

    def __init__(self, architecture):
        """
        architecture: lista con la cantidad de nodos por capa.
        """

        self.num_layers = len(architecture) # capas totales
        self.params = {} # diccionario para los pesos W y bias b

        # ADAM
        self.v = {} # Primer momento (Velocidad / Momentum)
        self.s = {} # Segundo momento (Velocidad al cuadrado / RMSprop)
        self.t = 0  # Contador de pasos de tiempo

        for i in range(1, self.num_layers):
            # inicialización He (multiplicar por sqrt(2/n)) 
            # mantiene la varianza de las activaciones al propagar hacia adelante con ReLU, evitando que las activaciones se apaguen o exploten en capas profundas
            self.params[f'W{i}'] = np.random.randn(architecture[i-1], architecture[i]) * np.sqrt(2. / architecture[i-1])
            self.params[f'b{i}'] = np.zeros((1, architecture[i]))

            self.v[f'dW{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.v[f'db{i}'] = np.zeros_like(self.params[f'b{i}'])
            self.s[f'dW{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.s[f'db{i}'] = np.zeros_like(self.params[f'b{i}'])
            
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
        self.save = {'A0': X}
        A = X
        
        # iteramos por las capas ocultas
        for i in range(1, self.num_layers - 1):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']

            Z = A @ W + b
            A = self.relu(Z)

            self.save[f'Z{i}'] = Z
            self.save[f'A{i}'] = A
            
        # Capa de salida 
        L = self.num_layers - 1
        W_out = self.params[f'W{L}']
        b_out = self.params[f'b{L}']
        
        Z_out = A @ W_out + b_out
        A_out = self.softmax(Z_out)

        self.save[f'Z{L}'] = Z_out
        self.save[f'A{L}'] = A_out
        
        return A_out # probabilidades

    def compute_loss(self, Y_true_one_hot, Y_pred, l2=0):
        """
        Calcula la función de costo Cross-Entropy.
        """
        m = Y_true_one_hot.shape[0]
        # evita que hagamos logaritmo de 0
        epsilon = 1e-20
        loss = -np.sum(Y_true_one_hot * np.log(Y_pred + epsilon)) / m

        # l2
        l2_cost = 0
        if l2 > 0:
            for i in range(1, self.num_layers):
                # suma de los cuadrados de todos los pesos
                l2_cost += np.sum(np.square(self.params[f'W{i}']))
            l2_cost = (l2 / (2*m)) * l2_cost
        return loss + l2_cost

    def backward(self, Y_true_one_hot, l2=0):
        m = Y_true_one_hot.shape[0]
        self.grads = {}
        
        # capa de salida
        L = self.num_layers - 1
        
        # Gradiente de la capa de salida (Softmax + Cross-Entropy)
        A_out = self.save[f'A{L}']
        # La derivada de Softmax + Cross-Entropy se simplifica a (Predicción - Real)
        dZ = A_out - Y_true_one_hot
        W_curr = self.params[f'W{L}']
        
        # Calculamos gradientes de pesos (dW) y sesgos (db) para la última capa
        A_prev = self.save[f'A{L-1}']
        self.grads[f'dW{L}'] = ((A_prev.T @ dZ) / m) + ((l2 / m) * W_curr) # le sumamos la penalidad 
        self.grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # propagación del error 
        for i in range(L - 1, 0, -1):
            W_next = self.params[f'W{i+1}']
            W_curr = self.params[f'W{i}']
            Z_curr = self.save[f'Z{i}']

            # derivada de la relu: pone 1 donde la Z original era positiva y 0 donde era negativa
            d_relu = np.where(Z_curr > 0, 1.0, 0.0) 
            
            # regla de la cadena para calcular el error en la capa actual
            dZ = (dZ @ W_next.T) * d_relu

            A_prev = self.save[f'A{i-1}']
            self.grads[f'dW{i}'] = ((A_prev.T @ dZ) / m) + ((l2 / m) * W_curr)
            self.grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m

    def update_params(self, learning_rate, optimizer='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8):
        # los beta indican que tanto import
        self.t += 1

        for i in range(1, self.num_layers):
            if optimizer == 'adam':

                # promedio móvil de los gradientes
                self.v[f'dW{i}'] = beta1 * self.v[f'dW{i}'] + (1 - beta1) * self.grads[f'dW{i}']
                self.v[f'db{i}'] = beta1 * self.v[f'db{i}'] + (1 - beta1) * self.grads[f'db{i}']
                
                # promedio móvil de los gradientes al cuadrado
                self.s[f'dW{i}'] = beta2 * self.s[f'dW{i}'] + (1 - beta2) * np.square(self.grads[f'dW{i}'])
                self.s[f'db{i}'] = beta2 * self.s[f'db{i}'] + (1 - beta2) * np.square(self.grads[f'db{i}'])
                
                # bias correction
                v_dW_corr = self.v[f'dW{i}'] / (1 - beta1**self.t)
                v_db_corr = self.v[f'db{i}'] / (1 - beta1**self.t)
                s_dW_corr = self.s[f'dW{i}'] / (1 - beta2**self.t)
                s_db_corr = self.s[f'db{i}'] / (1 - beta2**self.t)
                
                # actualización final de los pesos
                self.params[f'W{i}'] -= learning_rate * v_dW_corr / (np.sqrt(s_dW_corr) + epsilon)
                self.params[f'b{i}'] -= learning_rate * v_db_corr / (np.sqrt(s_db_corr) + epsilon)
            else:
                self.params[f'W{i}'] -= learning_rate * self.grads[f'dW{i}']
                self.params[f'b{i}'] -= learning_rate * self.grads[f'db{i}']

    def train(self, X_train, y_train, X_val, y_val,  epochs, learning_rate, scheduler_type='exponential', decay_rate=0.05, final_lr=1e-5, batch_size=None, patience=5, l2=0, optimizer='sgd', verbose=True):

        num_classes = self.params[f'W{self.num_layers-1}'].shape[1]

        y_train_one_hot = to_one_hot(y_train, num_classes=num_classes)
        y_val_one_hot = to_one_hot(y_val, num_classes=num_classes)
        
        hist_loss_train = []
        hist_loss_val = []

        m_train = X_train.shape[0]

        # no mini-batches
        if batch_size is None:
            batch_size = m_train
        # early stopping
        best_val_loss = float('inf')
        no_improve = 0
        
        for epoch in range(epochs):

            # rate scheduling
            if scheduler_type == 'exponential':
                current_lr = learning_rate * np.exp(-decay_rate * epoch)
            elif scheduler_type == 'linear':
                # bajamos linealmente desde el LR inicial hasta el final_lr
                # max() asegura que nunca baje más de final_lr (eso es la "saturación")
                progress = epoch /max(1, epochs - 1)
                current_lr = max(final_lr, learning_rate - (learning_rate - final_lr) * progress)

            # mini-batches
            permutation = np.random.permutation(m_train) # para que la red no memorice y podamos generalizar mejor
            X_shuffled = X_train[permutation, :]
            Y_shuffled = y_train_one_hot[permutation, :]

            epoch_loss = 0

            for i in range(0, m_train, batch_size):
                X_batch = X_shuffled[i: i+batch_size, :]
                Y_batch = Y_shuffled[i: i+batch_size, :]

                Y_pred_batch = self.forward(X_batch)
                batch_loss = self.compute_loss(Y_batch, Y_pred_batch, l2)

                # acumulamos el costo ponderado para sacar el promedio final
                epoch_loss += batch_loss * X_batch.shape[0]

                self.backward(Y_batch, l2)
                self.update_params(current_lr, optimizer)
            
            # promediamos el costo de la época
            epoch_loss_train = epoch_loss / m_train
            hist_loss_train.append(epoch_loss_train)
            
            # validation
            Y_pred_val = self.forward(X_val)
            loss_val = self.compute_loss(y_val_one_hot, Y_pred_val, l2) 
            hist_loss_val.append(loss_val)
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                scheduler_label = 'Exp' if scheduler_type == 'exponential' else 'Lin'
                print(f"Época {epoch:03d} | [{scheduler_label}] LR: {current_lr:.5f} | Train Loss: {epoch_loss_train:.4f} | Val Loss: {loss_val:.4f}")
            
            # early stopping
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                no_improve = 0 # reset
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print(f"\n¡Early stopping en la época {epoch}!")
                    break                    
                
        return hist_loss_train, hist_loss_val 
