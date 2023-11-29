import argparse, os, h5py, importlib
from modules.ymal_reader import parse_exp_settings
import numpy as np
from pathlib import Path


def create_model(model_name, model_setting, weight_path):
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(model_setting['filters'])
    model.load_weights(weight_path).expect_partial()
    return model
    

def main(exp_path, name_list):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    
    exp_settings = parse_exp_settings(exp_path)[0]
    exp_name = exp_settings['experiment_name']
    model_name = exp_settings['model_name']
    model_setting = exp_settings['model_setting']
    
    save_folder = f'output/{exp_name}/'  #!!!!!!!
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    with open(name_list, 'r') as f:
        for l in f:
            sid = l.strip().split()[0]

            # load model
            weight_path = f'Models/saved_weight/{exp_name}/M221_{sid}_full/AE'   #!!!!!!
            model = create_model(model_name, model_setting, weight_path) 
            
            # load test data
            data_file = f'data/7processed/{sid}.npz'   #!!!!!
            ds = np.load(data_file)
            test_data = ds['temp']
            predict = np.squeeze(model(test_data))
            np.save(f'{save_folder}/{sid}_1.npy', predict)  #!!!!
            print(sid, 'saved')


if __name__ == '__main__':
    main('experiments/minute_bk.yml', 'data/list_re.txt')
