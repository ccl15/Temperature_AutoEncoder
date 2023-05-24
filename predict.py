import argparse, os, h5py, importlib
from modules.ymal_reader import parse_exp_settings
import numpy as np
from pathlib import Path


def create_model(model_name, model_setting, weight_path):
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(model_setting['filters'])
    model.load_weights(weight_path).expect_partial()
    return model
    

def main(exp_path, yr, phaselist=['good', 'bad'], only_this_sub=''):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    exp_list = parse_exp_settings(exp_path, only_this_sub)
    for sub_exp_settings in exp_list:
        # information
        exp_name = sub_exp_settings['experiment_name']
        sub_exp_name = sub_exp_settings['sub_exp_name']
        sta_list = [sub_exp_name[3:]] #!!!! 3

        # load model
        model_save_path = f'Models/saved_weight/{exp_name}/{sub_exp_name}/AE'
        model = create_model(sub_exp_settings['model_name'], sub_exp_settings['model_setting'], model_save_path) 
        save_folder = f'output/{exp_name}/{sub_exp_name}'
        Path(save_folder).mkdir(parents=True, exist_ok=True)
        
        # load test data
        for station in sta_list:
            data_file = f'data/station_ds/{station}_H72_{yr}.h5'
            with h5py.File(data_file, 'r') as f:
                for phase in phaselist:
                    test_data = f[phase]['temp'][:]
                    predict = np.squeeze(model(test_data))
                    np.save(f'{save_folder}/{station}_{yr}_{phase}.npy', predict)
            print(save_folder, station, 'saved')
        


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('exp_path')
    # parser.add_argument('sub_exp')
    # parser.add_argument('station')
    # args = parser.parse_args()

    # main(args.exp_path, args.sub_exp, args.station)
    #sta_list =[466880]
    main('experiments/AE_2_s.yml', '20t22',
          only_this_sub='AE_467270') 
#         sta_list=sta_list)
