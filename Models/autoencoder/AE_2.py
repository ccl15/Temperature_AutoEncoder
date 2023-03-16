import tensorflow as tf
from tensorflow.keras import layers
'''
This is convolutional autoencoder
'''        

class Model(tf.keras.Model):
    def __init__(self, latent_dim, filters):
        super().__init__()
        self.encoder = [
                layers.InputLayer(input_shape=(72, 1)),
                layers.Conv1D(filters  , kernel_size=3, strides=2, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Conv1D(filters*2, kernel_size=3, strides=2, activation='relu', padding='same'),
                layers.BatchNormalization(),
                layers.Flatten(),
                layers.Dense(latent_dim),
            ]
        
        self.decoder = [
                layers.Dense(units=filters*36, activation='relu'),
                layers.Reshape(target_shape=(18, filters*2)),
                layers.Conv1DTranspose(filters, kernel_size=3, strides=2, padding='same', activation='relu'),
                layers.BatchNormalization(),
                layers.Conv1DTranspose(1 , kernel_size=3, strides=1, padding='same', activation='relu'),
                layers.BatchNormalization(),
                tf.keras.layers.Conv1D(1, kernel_size=3, strides=1, padding='same'),
            ]


    def __call__(self, x, bn_flag=False):
        for coder in [self.encoder, self.decoder]:
            for layer in coder:
                if isinstance(layer, layers.BatchNormalization):
                    x = layer(x, training=bn_flag)
                else:
                    x = layer(x)