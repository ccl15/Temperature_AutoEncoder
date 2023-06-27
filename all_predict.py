import argparse, os, h5py, importlib
from modules.ymal_reader import parse_exp_settings
import numpy as np
from pathlib import Path


def create_model(model_name, model_setting, weight_path):
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(model_setting['filters'])
    model.load_weights(weight_path).expect_partial()
    return model
    

def main(exp_path, name_list, phaselist=['good', 'bad']):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    
    exp_settings = parse_exp_settings(exp_path)[0]
    exp_name = exp_settings['experiment_name']
    model_name = exp_settings['model_name']
    model_setting = exp_settings['model_setting']

    with open(name_list, 'r') as f:
        sids = [l.strip() for l in f]

    for sid in sids:
        # load model
        weight_path = f'Models/saved_weight/{exp_name}/AE_{sid}/AE'   #!!!!!!
        model = create_model(model_name, model_setting, weight_path) 
        
        save_folder = f'output/{exp_name}/AE_{sid}'  #!!!!!!!
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        # load test data
        data_file = f'data/2ds_46/{sid}_H24_18t21.h5'   #!!!!!
        with h5py.File(data_file, 'r') as f:
            for phase in phaselist:
                test_data = f[phase]['temp'][:]
                predict = np.squeeze(model(test_data))
                np.save(f'{save_folder}/{sid}_18t21_{phase}.npy', predict)  #!!!!
        print(sid, 'saved')


if __name__ == '__main__':
    main('experiments/AE_2_s24.yml', 'data/list_46_re.txt', ['good'])
