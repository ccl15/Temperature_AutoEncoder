import tensorflow as tf
from tensorflow.keras import layers
'''
This is convolutional autoencoder
'''        

class Model(tf.keras.Model):
    def __init__(self, filters):
        super().__init__()
        self.encoder = [
                layers.InputLayer(input_shape=(32, 1)),
                layers.Conv1D(filters  , kernel_size=3, strides=2, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(2*filters, kernel_size=3, strides=2, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dense(4*filters),
            ]
        
        self.decoder = [
                layers.Dense(units=16*filters, activation='relu'),
                layers.Reshape(target_shape=(8, filters*2)),
                layers.Conv1DTranspose(filters, kernel_size=3, strides=2, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv1DTranspose(1 , kernel_size=3, strides=2, padding='same'),
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
