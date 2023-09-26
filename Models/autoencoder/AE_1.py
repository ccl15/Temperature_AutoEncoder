import tensorflow as tf
from tensorflow.keras import layers

class HiddenLayer(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.bn = layers.BatchNormalization()
        self.fc = layers.Dense(filters, activation='relu')
    def __call__(self, x, training):
        return self.fc(self.bn(x, training=training))
        
class Model(tf.keras.Model):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_layer = layers.InputLayer((input_size,))
        self.hidden_layers = [HiddenLayer(filters) for filters in hidden_size]
        self.output_layer = layers.Dense(input_size)
            
    def __call__(self, x, training=False):
        x = self.input_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x, training=training)
        x = self.output_layer(x)
        return x
