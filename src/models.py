import numpy as np

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
   