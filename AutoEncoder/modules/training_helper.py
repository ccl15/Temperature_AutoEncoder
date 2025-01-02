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

def get_tfr_datasets(data_file, shuffle_buffer, batch_size):
    # load data
    def _parse_example(example_string):
        feature_description = {
            'temp': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(example_string, feature_description)

        temp = tf.io.decode_raw(features['temp'], tf.float32)
        return temp

    raw_dataset = tf.data.TFRecordDataset(data_file)
    dataset = raw_dataset.map(_parse_example)

    # split to train/valid
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    count = sum(1 for _ in dataset)
    train_size = int(count*0.7)
    ds_for_model ={
        'train' : dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE),
        'valid' : dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    }
    return ds_for_model

