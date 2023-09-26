import argparse, os, h5py, importlib
from modules.ymal_reader import parse_exp_settings
import numpy as np
from pathlib import Path


def create_model(model_name, model_setting, weight_path):
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(model_setting['filters'])
    model.load_weights(weight_path).expect_partial()
    return model
    
def main(exp_path, sub_exp):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    
    # load model
    exp_settings = parse_exp_settings(exp_path, sub_exp)[0]
    exp_name = exp_settings['experiment_name']
    weight_path = f'Models/saved_weight/{exp_name}/{sub_exp}/AE'
    model = create_model(exp_settings['model_name'], exp_settings['model_setting'], weight_path) 

    # save path
    save_folder = f'output/{exp_name}/{sub_exp}'  #!!!!!!!
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # stations
    stations = Path('data/7processed').glob('*.npz')
    for data_file in stations:
        ds = np.load(data_file)
        test_data = ds['temp']

        predict = np.squeeze(model(test_data))
        np.save(f'{save_folder}/{data_file.stem}.npy', predict)


if __name__ == '__main__':
    main('experiments/minute_p.yml','M221_lev2_f32_len16_2e4_p2p')
