import cupy as np

class MLP_Advanced:

    def __init__(self, layer_sizes):
        """
        layer_sizes: lista con la cantidad de nodos por capa.
        """

        self.num_layers = len(layer_sizes) # capas totales
        self.params = {} # diccionario para los pesos W y bias b
        self.v = {} # primer momento ADAM
        self.s = {} # segundo ,momento ADAM
        self.t = 0 # contador ADAM

        for i in range(1, self.num_layers):

            self.params[f'W{i}'] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * np.sqrt(2. / layer_sizes[i-1])
            self.params[f'b{i}'] = np.zeros((1, layer_sizes[i]))

            # Inicializamos los momentos de Adam en 0
            self.v[f'dW{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.v[f'db{i}'] = np.zeros_like(self.params[f'b{i}'])
            self.s[f'dW{i}'] = np.zeros_like(self.params[f'W{i}'])
            self.s[f'db{i}'] = np.zeros_like(self.params[f'b{i}'])            
            
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
   
     
    def forward(self, X):

        self.cache = {'A0': X}
        A = X

        for i in range(1, self.num_layers - 1):
            W = self.params[f'W{i}']
            b = self.params[f'b{i}']

            Z = A @ W + b
            A = self.relu(Z)

            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A

        L = self.num_layers - 1
        W_out = self.params[f'W{L}']
        b_out = self.params[f'b{L}']
        
        Z_out = A @ W_out + b_out
        A_out = self.softmax(Z_out)

        self.cache[f'Z{L}'] = Z_out
        self.cache[f'A{L}'] = A_out
        
        return A_out 

    def compute_loss(self, Y_true_one_hot, Y_pred, l2=0):
        m = Y_true_one_hot.shape[0]
        epsilon = 1e-15
        loss = -np.sum(Y_true_one_hot * np.log(Y_pred + epsilon)) / m

        l2_cost = 0
        if l2 > 0:
            for i in range(1, self.num_layers):
                l2_cost += np.sum(np.square(self.params[f'W{i}']))
            l2_cost = (l2 / (2 * m)) * l2_cost
            
        return loss + l2_cost

    def backward(self, Y_true_one_hot, l2=0):
        m = Y_true_one_hot.shape[0]
        self.grads = {}
        L = self.num_layers - 1
        
        A_out = self.cache[f'A{L}']
        dZ = A_out - Y_true_one_hot
        
        A_prev = self.cache[f'A{L-1}']
        W_curr = self.params[f'W{L}']
        self.grads[f'dW{L}'] = ((A_prev.T @ dZ) / m) + ((l2 / m) * W_curr)
        self.grads[f'db{L}'] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # 2. Propagación del error hacia atrás para las capas ocultas
        for i in range(L - 1, 0, -1):
            W_next = self.params[f'W{i+1}']
            Z_curr = self.cache[f'Z{i}']
            d_relu = np.where(Z_curr > 0, 1.0, 0.0)

            dZ = (dZ @ W_next.T) * d_relu

            A_prev = self.cache[f'A{i-1}']
            W_curr = self.params[f'W{i}']
            self.grads[f'dW{i}'] = ((A_prev.T @ dZ) / m) + ((l2 / m) * W_curr)
            self.grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m

    def update_params(self, learning_rate, optimizer='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8):

        self.t += 1

        for i in range(1, self.num_layers):
            if optimizer == 'adam':
                # --- MEJORA: Optimizador Adam ---
                # 1. Momento de primer orden (Promedio móvil de los gradientes)
                self.v[f'dW{i}'] = beta1 * self.v[f'dW{i}'] + (1 - beta1) * self.grads[f'dW{i}']
                self.v[f'db{i}'] = beta1 * self.v[f'db{i}'] + (1 - beta1) * self.grads[f'db{i}']
                
                # 2. Momento de segundo orden (Promedio móvil de los gradientes al cuadrado)
                self.s[f'dW{i}'] = beta2 * self.s[f'dW{i}'] + (1 - beta2) * np.square(self.grads[f'dW{i}'])
                self.s[f'db{i}'] = beta2 * self.s[f'db{i}'] + (1 - beta2) * np.square(self.grads[f'db{i}'])
                
                # 3. Corrección de sesgo
                v_dW_corr = self.v[f'dW{i}'] / (1 - beta1**self.t)
                v_db_corr = self.v[f'db{i}'] / (1 - beta1**self.t)
                s_dW_corr = self.s[f'dW{i}'] / (1 - beta2**self.t)
                s_db_corr = self.s[f'db{i}'] / (1 - beta2**self.t)
                
                # 4. Actualización
                self.params[f'W{i}'] -= learning_rate * v_dW_corr / (np.sqrt(s_dW_corr) + epsilon)
                self.params[f'b{i}'] -= learning_rate * v_db_corr / (np.sqrt(s_db_corr) + epsilon)
            else:
                self.params[f'W{i}'] -= learning_rate * self.grads[f'dW{i}']
                self.params[f'b{i}'] -= learning_rate * self.grads[f'db{i}']

    def train(self, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size=None, optimizer='sgd', l2=0, decay_rate=0.05, patience=10, scheduler_type='exponential', final_lr=1e-5):
        
        num_classes = self.params[f'W{self.num_layers-1}'].shape[1]
        y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
        y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
        
        y_val_one_hot = np.zeros((y_val.shape[0], num_classes))
        y_val_one_hot[np.arange(y_val.shape[0]), y_val] = 1
        
        m_train = X_train.shape[0]
        historial_loss_train, historial_loss_val = [], []
        
        # --- MEJORA: Early Stopping ---
        best_val_loss = float('inf')
        epochs_no_improve = 0
        
        # Si no se define batch_size, usamos todo el dataset (Batch Gradient Descent)
        if batch_size is None:
            batch_size = m_train
            
        for epoch in range(epochs):
            # --- MEJORA: Rate Scheduling (Lineal o Exponencial) ---
            if scheduler_type == 'exponential':
                # Scheduling Exponencial: LR(t) = LR_0 * exp(-decay_rate * t)
                current_lr = learning_rate * np.exp(-decay_rate * epoch)
            elif scheduler_type == 'linear':
                # Scheduling Lineal con saturación: LR(t) = max(final_lr, LR_0 - (LR_0 - final_lr) * t/T)
                # Baja linealmente desde learning_rate hasta final_lr a lo largo de todas las épocas
                current_lr = max(final_lr, learning_rate - (learning_rate - final_lr) * (epoch / max(1, epochs - 1)))
            else:
                raise ValueError(f"scheduler_type '{scheduler_type}' no reconocido. Use 'exponential' o 'linear'.")
            
            # --- MEJORA: Mini-batches ---
            # Mezclamos los datos al inicio de cada época
            permutation = np.random.permutation(m_train)
            X_shuffled = X_train[permutation, :]
            Y_shuffled = y_train_one_hot[permutation, :]
            
            epoch_loss = 0
            num_batches = int(np.ceil(m_train / batch_size))
            
            for i in range(0, m_train, batch_size):
                X_batch = X_shuffled[i:i + batch_size, :]
                Y_batch = Y_shuffled[i:i + batch_size, :]
                
                Y_pred_batch = self.forward(X_batch)
                batch_loss = self.compute_loss(Y_batch, Y_pred_batch, l2)
                epoch_loss += batch_loss * X_batch.shape[0] # Ponderamos por el tamaño del batch
                
                self.backward(Y_batch, l2)
                self.update_params(current_lr, optimizer)
                
            # Promediamos el costo de la época
            epoch_loss_train = epoch_loss / m_train
            historial_loss_train.append(epoch_loss_train)
            
            # Validación
            Y_pred_val = self.forward(X_val)
            loss_val = self.compute_loss(y_val_one_hot, Y_pred_val, l2) # Evaluamos L2 también en validación
            historial_loss_val.append(loss_val)
            
            if epoch % 5 == 0 or epoch == epochs - 1:
                scheduler_label = 'Exp' if scheduler_type == 'exponential' else 'Lin'
                print(f"Época {epoch:03d} | [{scheduler_label}] LR: {current_lr:.5f} | Train Loss: {epoch_loss_train:.4f} | Val Loss: {loss_val:.4f}")
            
            # Early Stopping check
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                epochs_no_improve = 0
                # Acá podrías guardar los mejores pesos si quisieras
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\n¡Early stopping activado en la época {epoch}! La validación no mejoró en {patience} épocas.")
                    break
                    
        return historial_loss_train, historial_loss_val
    

