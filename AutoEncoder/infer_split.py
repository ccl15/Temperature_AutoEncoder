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


    
#%% calculate
def MAE(x, y, ax=""):
    if ax == "":
        return np.mean(np.abs(x-y))
    else:
        return np.mean(np.abs(x-y), axis=int(ax))

def create_model(exp_settings, sub_exp):
    # load setting
    exp = exp_settings['experiment_name']
    model_settings = exp_settings['model_setting']
    model_name = model_settings['name']
    weight_path = f'Models/saved/{exp}/{sub_exp}/AE'

    # load model weight
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(**model_settings)
    model.load_weights(weight_path).expect_partial()
    return model

def stn_high():
    with open("/home/ccl/Resources/StnTown_Sta_20250520.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    elevation_dict = {item["ID"]: item["Elev"] for item in data["Item"]}
    return elevation_dict


#%%
def Do_inference_one(sid, subexp, data_file, save_name):
    exp_settings = parse_exp_settings(exp_path, subexp)[0]
    model = create_model(exp_settings, subexp)

    data = h5py.File(data_file, 'r')
    infe = model(data[sid]['data'][:].astype('float32')).numpy()
    Terr_cases = MAE(data[sid]['data'][:,-3:,0], infe[:,-3:,0], ax=1)

    with h5py.File(f'{save_name}.h5', 'w') as fo:
        grp = fo.create_group(sid)
        grp.create_dataset('Pass', data = infe)
        grp.create_dataset('PassMAE', data = Terr_cases)         
    print(f'{save_name} saved.')

def Do_inference_list(split_high, sub_list, data_file, save_name):
    exp_settings = parse_exp_settings(exp_path, sub_list[0])[0]
    models = [create_model(exp_settings, sub) for sub in sub_list]

    data = h5py.File(data_file, 'r')

    #with h5py.File(f'{folder}/cases_infer.h5', 'w') as fo:
    with h5py.File(f'{save_name}', 'w') as fo:
        for sid in tqdm(data.keys()):

            # Add station high select
            if sid in elevation_dict:
                sta_high = elevation_dict[sid]
            else: continue

            # chosen model
            if sta_high < split_high:
                model = models[0]
            else:
                model = models[1]
        
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
#%% main 
if __name__ == '__main__':
    elevation_dict = stn_high()

    # 10 min
    '''
    exp_path = 'experiments/AE_10min.yml' #!!!!
    exp = 'AE_10min'
    
    Do_inference_one('C0I520', 'AE30f16_C0I520', 
                     '../dataHR/2_input/All_y2022.h5',
                     f'verify/{exp}/LH_fs/C0I520_2022')
    yr = 2024
    Do_inference_list(
        [1500],
        ['AE30f32_L15','AE30f32_H20_1em5'],
        f'../data10/2_input/y{yr}.h5', 
        f'verify/{exp}/LH_fs/y{yr}_m20.h5'
    )
    '''
    # HR
    exp_path = 'experiments/AE_hr16_copy.yml' #!!!!
    #pre_yr = 24
    for test_yr in [23]:
        Do_inference_list(1500,
                    ['AE30f16_Low2', 'AE30f16_H2000_1e4'],
                    f'../dataHR/2_input/All_y20{test_yr}.h5', 
                    f'verify/AE_HR16//Trans_2023/{test_yr}_LH.h5')
    
    