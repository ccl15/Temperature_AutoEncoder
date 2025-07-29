import tensorflow as tf
from tensorflow.keras import layers
       

class Model(tf.keras.Model):
    def __init__(self, filters, ks=3, act='leaky', **kwargs):
        super().__init__()
        if act == 'leaky':
            act = layers.LeakyReLU()

        self.ae_model = [
            layers.Conv1D(1*filters  , kernel_size=ks, strides=2, activation=act, padding='same'),
            layers.Conv1D(2*filters  , kernel_size=ks, strides=2, activation=act, padding='same'),
            layers.Conv1DTranspose(filters  , kernel_size=ks, strides=2, activation=act, padding='same'),
            layers.Conv1DTranspose(2 , kernel_size=ks, strides=2, padding='same')
        ]

    def __call__(self, x, training=False):
        for layer in self.ae_model:
            x = layer(x)
        return x
    