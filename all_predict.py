import argparse, os, h5py, importlib
from modules.ymal_reader import parse_exp_settings
import numpy as np
from pathlib import Path


def create_model(model_name, model_setting, weight_path):
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(model_setting['filters'])
    model.load_weights(weight_path).expect_partial()
    return model
    

def main(exp_path, phaselist=['good', 'bad'], only_this_sub=''):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''


    exp_settings = parse_exp_settings(exp_path, only_this_sub)[0]
    exp_name = exp_settings['experiment_name']
    model_name = exp_settings['model_name']
    model_setting = exp_settings['model_setting']

    subexp_save_paths = Path(f'Models/saved_weight/{exp_name}').glob('./AE_*')
    for weight_save_path in subexp_save_paths:
        # load model
        model = create_model(model_name, model_setting, f'{weight_save_path}/AE') 
        
        subexp_name = weight_save_path.name
        save_folder = f'output/{exp_name}/{subexp_name}'
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        # load test data
        station = subexp_name[3:]
        if station == '467270':
            data_file = 'data/station_ds/467270_H24_20t22.h5'
        else:
            data_file = f'data/station_ds/{station}_H24_18t21.h5'
        with h5py.File(data_file, 'r') as f:
            for phase in phaselist:
                test_data = f[phase]['temp'][:]
                predict = np.squeeze(model(test_data))
                np.save(f'{save_folder}/{station}_18t21_{phase}.npy', predict)
        print(save_folder, station, 'saved')


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('exp_path')
    # parser.add_argument('sub_exp')
    # parser.add_argument('station')
    # args = parser.parse_args()

    # main(args.exp_path, args.sub_exp, args.station)
    main('experiments/AE_2_s24.yml')
