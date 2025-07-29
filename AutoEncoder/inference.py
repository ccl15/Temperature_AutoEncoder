import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from modules.ymal_reader import parse_exp_settings
import numpy as np
#import tensorflow as tf 
import importlib
import h5py
from tqdm import tqdm
from pathlib import Path
import json


def stn_high():
    with open("/home/ccl/Resources/StnTown_Sta_20250520.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    elevation_dict = {item["ID"]: item["Elev"] for item in data["Item"]}
    return elevation_dict
    

#%% calculate
def MAE(x, y, ax=""):
    if ax == "":
        return np.mean(np.abs(x-y))
    else:
        return np.mean(np.abs(x-y), axis=int(ax))

def create_model(exp_settings, sub_exp):
    # load setting
    model_settings = exp_settings['model_setting']
    model_name = model_settings['name']
    weight_path = f'Models/saved/{exp}/{sub_exp}/AE'

    # load model weight
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(**model_settings)
    model.load_weights(weight_path).expect_partial()
    return model


def Do_inference(sub_exp, data_file, save_name, H_limt=None):
    exp_settings = parse_exp_settings(exp_path, sub_exp)[0]
    model = create_model(exp_settings, sub_exp)

    data = h5py.File(data_file, 'r')

    #with h5py.File(f'{folder}/cases_infer.h5', 'w') as fo:
    with h5py.File(f'{save_name}', 'w') as fo:
        for sid in tqdm(data.keys()):
               
            # good data error
            data1 = data[sid]['data'][:].astype('float32')
            if data1.shape[0] == 0:
                continue
            else:
                infe = model(data1).numpy()
                Terr_cases = MAE(data1[:,-3:,0], infe[:,-3:,0], ax=1)

            grp = fo.create_group(sid)
            grp.create_dataset('Pass', data = infe)
            grp.create_dataset('PassMAE', data = Terr_cases)            
            
            # bad data error
            if 'test_data' in data[sid]:
                data2 = data[sid]['test_data'][:].astype('float32')
                if data2.shape[0] == 0:
                    continue
                infe = model(data2).numpy()
                terr_cases = MAE(data2[:, -3:, 0], infe[:, -3:, 0], ax=1)
                
            grp.create_dataset('Warm', data = infe)
            grp.create_dataset('WarmMAE', data = terr_cases)


#%% main 
if __name__ == '__main__':
    exp_path = 'experiments/AE_10min.yml' #!!!!
    exp = 'AE_10min'
    
    data_file = '../data10/2_input/y2022.h5'
    sub_exp = 'Pre_M30_f32k3'
    folder = f'verify/{exp}/{sub_exp}'
    Path(folder).mkdir(parents=True, exist_ok=True)
    Do_inference(sub_exp, data_file, f'{folder}/cases_2022.h5')
    '''
    #
    sub_exp = 'AE30f32_H15'
    data_file = '../data10/2_input/y2022_h15.h5'
    save_name = f'verify/{exp}/LH_fs/y2022_H15.h5'
    Do_inference(sub_exp, data_file, save_name)
    
    sub_exp = 'AE30f32_L15'
    data_file = '../data10/2_input/y2022_l15.h5'
    save_name = f'verify/{exp}/LH_fs/y2022_L15.h5'
    Do_inference(sub_exp, data_file, save_name)
    
    exp_path = 'experiments/AE_hr16.yml' #!!!!
    exp = 'AE_HR16'

    Do_inference('Pre_AE30_f16_mse',
                '../dataHR/2_input/All_y2022.h5', 
                f'verify/{exp}/Pre_AE30_f16_mse/y2022_infer.h5')
    Do_inference('AE30f16_H2000_1e4',
                '../dataHR/2_input/y2022_h2000.h5', 
                f'verify/{exp}/LH_fs/y2022_h2.h5')
    '''