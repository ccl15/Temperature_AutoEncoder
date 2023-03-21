import argparse, os, h5py, importlib
from modules.ymal_reader import parse_exp_settings
import numpy as np
from pathlib import Path


def create_model(model_name, model_setting, weight_path):
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(**model_setting)
    model.load_weights(weight_path).expect_partial()
    return model
    

def main(exp_path, only_this_sub, station, yr='22'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    exp_list = parse_exp_settings(exp_path, only_this_sub)
    for sub_exp_settings in exp_list:
        # load model
        exp_name = sub_exp_settings['experiment_name']
        sub_exp_name = sub_exp_settings['sub_exp_name']
        model_save_path = f'Models/saved_weight/{exp_name}/{sub_exp_name}/AE'
        model = create_model(sub_exp_settings['model_name'], sub_exp_settings['model_setting'], model_save_path)
        
        # load test data
        data_file = f'data/station_ds/{station}_H72_{yr}.h5'
        with h5py.File(data_file, 'r') as f:
            test_data = f['bad/temp'][:]
        # output 
        predict = np.squeeze(model.predict(test_data))
            
        save_folder = f'output/{exp_name}/{sub_exp_name}'
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        np.save(f'{save_folder}/{station}_H72_{yr}.npy', predict)
    


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('exp_path')
    # parser.add_argument('sub_exp')
    # parser.add_argument('station')
    # args = parser.parse_args()

    # main(args.exp_path, args.sub_exp, args.station)
    main('experiments/AE_2_0.yml', 'H72_466920_1em3', '466920')

