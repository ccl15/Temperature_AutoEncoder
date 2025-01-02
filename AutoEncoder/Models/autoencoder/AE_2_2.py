import tensorflow as tf
from tensorflow.keras import layers
'''
base on AE_2_0. Kernel=5
'''        

class Model(tf.keras.Model):
    def __init__(self, levels, filters):
        super().__init__()

        ae_model = []
        for level in range(1,levels+1):
            ae_model.append(layers.Conv1D(level*filters  , kernel_size=5, strides=2, activation='relu', padding='same'))
            ae_model.append(layers.BatchNormalization())
        for level in range(levels-1,0,-1):
            ae_model.append(layers.Conv1DTranspose(filters  , kernel_size=5, strides=2, activation='relu', padding='same'))
            ae_model.append(layers.BatchNormalization())
        ae_model.append(layers.Conv1DTranspose(1 , kernel_size=5, strides=2, padding='same'))  
        self.ae_model = ae_model

    def __call__(self, x, training=False):
        x = tf.expand_dims(x, -1)
        for layer in self.ae_model:
            if isinstance(layer, layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x
    