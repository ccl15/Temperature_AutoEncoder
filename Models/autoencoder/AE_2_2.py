import tensorflow as tf
from tensorflow.keras import layers
'''
This is convolutional autoencoder
padding valid and stride 1
'''        

class Model(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.encoder = [
                layers.InputLayer(),
                layers.Conv1D(filters  , kernel_size=3, activation='relu', padding='valid'),
                layers.BatchNormalization(),
                layers.Conv1D(filters*2, kernel_size=3, activation='relu', padding='valid'),
                layers.BatchNormalization(),
            ]
        
        self.decoder = [
                layers.Conv1DTranspose(filters, kernel_size=3, padding='valid', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv1DTranspose(1 , kernel_size=3,  padding='valid'),
            ]


    def __call__(self, x, training=False):
        x = tf.expand_dims(x, -1)
        for coder in [self.encoder, self.decoder]:
            for layer in coder:
                if isinstance(layer, layers.BatchNormalization):
                    x = layer(x, training=training)
                else:
                    x = layer(x)
        return x
