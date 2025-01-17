import tensorflow as tf
from tensorflow.keras import layers
'''
For 10 minute Temp and RH
'''


class Model(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()

        self.ae_model = [
            layers.Conv1D(1*filters  , kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv1D(2*filters  , kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv1DTranspose(1*filters  , kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv1DTranspose(2 , kernel_size=3, strides=2, padding='same') # set output channel number !
        ]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 16, 2], dtype=tf.float32)])
    def __call__(self, x, **kwargs):
        for layer in self.ae_model:
            x = layer(x)
        return x
