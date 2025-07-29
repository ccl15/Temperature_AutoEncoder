import h5py
import numpy as np
import tensorflow as tf

    
def get_h5py_datasets(file, station, batch_size, shuffle_buffer):
    # read file
    with h5py.File(file, 'r') as f:
        if station == 'ALL':
            all_data = []
            for sid in f.keys():
                data1 = f[sid]['data'][:].astype('float32')
                if data1.shape[0] > 0:
                    all_data.append(data1)
            data = np.vstack(all_data)
        else:
            data = f[station]['data'][:].astype('float32')
    
    # numbers
    print(f'data shape {data.shape}')
    n_train = int(len(data)*0.75)

    # Shuffle and counting
    np.random.shuffle(data)
    data = tf.data.Dataset.from_tensor_slices(data)

    # Split into train/valid
    dataset = {
        'train': data.take(n_train).shuffle(shuffle_buffer).batch(batch_size),
        'valid': data.skip(n_train).shuffle(shuffle_buffer).batch(batch_size)
        }
    return dataset


def get_tfr_datasets(data_file, shuffle_buffer, batch_size, input_shp=None):
    # load data
    def _parse_example(example_string):
        feature_description = {
            'temp': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(example_string, feature_description)

        if input_shp:
            temp = tf.reshape(tf.io.decode_raw(features['temp'], tf.float32), input_shp)
        else:
            temp = tf.io.decode_raw(features['temp'], tf.float32)
        return temp

    raw_dataset = tf.data.TFRecordDataset(data_file)
    dataset = raw_dataset.map(_parse_example)

    # Shuffle and counting
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    count = sum(1 for _ in dataset)
    train_size = int(count*0.7)

    # split to train/valid
    ds_for_model ={
        'train' : dataset.take(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE),
        'valid' : dataset.skip(train_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    }
    return ds_for_model

