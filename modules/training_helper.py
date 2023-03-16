import h5py
import tensorflow as tf
import random

def evaluate_loss(model, dataset, loss_func):
    avg_loss = tf.keras.metrics.Mean(dtype=tf.float32)
    for batch_index, data_in in dataset.enumerate():
        data_out = model(data_in, training=False)
        loss = loss_func(data_in, data_out)
        avg_loss.update_state(loss)
    return avg_loss.result()

    
def get_tf_datasets(data_file, batch_size, shuffle_buffer):
    with h5py.File(data_file, 'r') as f:
        data = f['good/temp'][:]
    n_train = int(len(data)*0.75)
    
    # Shuffle and create dataset
    random.shuffle(data)
    data = tf.data.Dataset.from_tensor_slices(data)
    
    # Split into train/valid
    dataset = {
        'train': data.take(n_train).shuffle(shuffle_buffer).batch(batch_size).prefetch(1),
        'valid': data.skip(n_train).shuffle(shuffle_buffer).batch(batch_size).prefetch(1)
        }
    return dataset
