import argparse, os, h5py, importlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from modules.ymal_reader import parse_exp_settings
import numpy as np
from pathlib import Path
import tensorflow as tf

def create_model(model_name, model_setting, weight_path):
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(model_setting['filters'])
    model.load_weights(weight_path).expect_partial()
    return model

def read_tfr(data_file,length, n):
    # load data
    def _parse_example(example_string):
        feature_description = {
            'temp': tf.io.FixedLenFeature([], tf.string),
            'time': tf.io.FixedLenFeature([], tf.string),
        }
        features = tf.io.parse_single_example(example_string, feature_description)

        temp = tf.io.decode_raw(features['temp'], tf.float32)
        return temp[-length:], features['time']

    raw_dataset = tf.data.TFRecordDataset(data_file)
    dataset = raw_dataset.map(_parse_example).shuffle(10000)
    return dataset.take(n)


def main(exp_path, only_this_sub, length, station):
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    # information
    sub_exp_settings = parse_exp_settings(exp_path, only_this_sub)[0]
    exp_name = sub_exp_settings['experiment_name']
    sub_exp_name = sub_exp_settings['sub_exp_name']

    # load model
    model_save_path = f'Models/saved_weight/{exp_name}/{sub_exp_name}/AE'
    model = create_model(sub_exp_settings['model_name'], sub_exp_settings['model_setting'], model_save_path) 
    save_folder = f'output/{exp_name}/{sub_exp_name}'
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # load test data
    '''
    for station in sta_list:
        data_file = f'data/2ds_46/{station}_H24_{yr}.h5'
        with h5py.File(data_file, 'r') as f:
            for phase in phaselist:
                test_data = f[phase]['temp'][:]
    '''
    #obs, times = zip(*read_tfr('data/6minute/466900.tfr',length, 10))
    ds = np.load(f'{save_folder}/{station}.npz')
    obs = ds['obs']
    predict = ds['pred']
    dates = ds['date']
    bad_pred = []
    bad_obs = []
    for dt in [-3,-10]:
        obsp = obs.copy()
        obsp[:,-1] += dt
        bad_pred.append(np.squeeze(model(obsp)))
        bad_obs.append(obsp)
    '''
    # For convolution AE --------------
    predict = np.squeeze(model(obs)) 
    # For Fully-connected AE ------------------------
    predict = []
    for obs1 in obs:
        obs1 = obs1[np.newaxis,:]
        predict.append(np.squeeze(model(obs1)))
    
    # -----------------------        
    dates = []
    for time in times:
        dates.append(time.numpy().decode())
    '''
    np.savez(f'{save_folder}/{station}_bad.npz', obs = bad_obs, pred = bad_pred,  date = dates)
    print('Save done')        


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('exp_path')
    # parser.add_argument('sub_exp')
    # parser.add_argument('station')
    # args = parser.parse_args()

    # main(args.exp_path, args.sub_exp, args.station)
    main('experiments/minute_p.yml', 'M221_lev2_f32_len16_2e4_p2p', 16, '466900')

