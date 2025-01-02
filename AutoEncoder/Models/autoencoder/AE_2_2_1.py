import tensorflow as tf
from tensorflow.keras import layers
       

class Model(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()

        self.ae_model = [
            layers.Conv1D(1*filters  , kernel_size=5, strides=2, activation='relu', padding='same'),
            layers.Conv1D(2*filters  , kernel_size=5, strides=2, activation='relu', padding='same'),
            layers.Conv1DTranspose(filters  , kernel_size=5, strides=2, activation='relu', padding='same'),
            layers.Conv1DTranspose(1 , kernel_size=5, strides=2, padding='same')
        ]

    def __call__(self, x, training=False):
        x = tf.expand_dims(x, -1)
        for layer in self.ae_model:
            x = layer(x)
        return x
    