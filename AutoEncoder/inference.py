import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from modules.ymal_reader import parse_exp_settings
import numpy as np
import pandas as pd
#import tensorflow as tf 
import importlib
import h5py
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from tqdm import tqdm
from pathlib import Path


#%% calculate
def MAE(x, y, ax=""):
    if ax == "":
        return np.mean(np.abs(x-y))
    else:
        return np.mean(np.abs(x-y), axis=int(ax))

def create_model(exp_path, sub_exp_name):
    # load setting
    exp_settings = parse_exp_settings(exp_path, sub_exp_name)[0]
    model_settings = exp_settings['model_setting']
    model_name = model_settings['name']
    weight_path = f'Models/saved/AE_10min/{sub_exp_name}/AE'

    # load model weight
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    model = model_class.Model(**model_settings)
    model.load_weights(weight_path).expect_partial()
    return model


#%% plot result
def plot_T_RH(x, y, t, sid, fname):
    plt.figure()
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    Tnode = np.arange(-15,1)

    axs[0].plot(Tnode, x[:,0], '', label='Observation')
    axs[0].plot(Tnode, y[:,0], '', label='Inference')
    axs[0].set_ylabel('Temp (oC)')
    axs[0].set_title(f'[{sid}] {t}', loc='left')
    axs[0].set_title(f'L3MAE:{MAE(x,y):.2f}', loc='right')
    axs[0].legend()

    axs[1].plot(Tnode, x[:,1]*100, '', label='Observation')
    axs[1].plot(Tnode, y[:,1]*100, '', label='Inference')
    axs[1].set_ylabel('RH (%)')

    plt.subplots_adjust(hspace=0.05)
    plt.savefig(f'{fname}.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_station_cases(sid):
    data = f[sid]['test_data'].astype('float32')
    time = f[sid]['test_time']#[::10]
    infe = model(data).numpy()

    # plot
    for i in tqdm(range(len(time))):
        x = data[i,...]
        y = infe[i,...]
        t = time[i].decode()
        plot_T_RH(x, y, t, f'{folder}/{sid}_W{t}')


def stations_mae_txt(f):
    Error_set= {}
    # inference all station
    for sid in tqdm(f.keys()):
        # good data error
        data1 = f[sid]['data'][:].astype('float32')
        if data1.shape[0] == 0:
            continue
        else:
            infe = model(data1).numpy()
            Terr_all = MAE(data1[:,-3:,0], infe[:,-3:,0], ax=1)
            T_mean = np.mean(Terr_all)
            T_std = np.std(Terr_all)
            RHerr = MAE(data1[:,-3:, 1], infe[:,-3:, 1])

        # bad data error
        data2 = f[sid]['test_data'][:].astype('float32')
        if data2.shape[0] == 0: 
            t_mean = t_std = t_RHerr = np.nan
        else:
            infe = model(data2).numpy()
            t_all = MAE(data2[:, -3:, 0], infe[:, -3:, 0], ax=1)
            t_mean = np.mean(t_all)
            t_std = np.std(t_all)
            t_RHerr = MAE(data2[:, -3:, 1], infe[:, -3:, 1])
        Error_set[sid] = f'{T_mean:.3f} {T_std:.3f} {RHerr:.3f} {t_mean:.3f} {t_std:.3f} {t_RHerr:.3f}'
        print(Error_set[sid])
    # save to txt
    #fname = f'verify/{sub_exp}/mae_mean.txt'
    #with open(fname, 'w') as f:
    #    f.write(f'sid [G] Tmean, Tstd, RH_m;[B] Tmean, Tstd, RH_m\n')
    #    for key, value in Error_set.items():
    #        f.write(f'{key} {value}\n')


def save_cases_mae(f):
    with h5py.File(f'{folder}/cases_mae.h5', 'w') as fo:
        for sid in f.keys():
            # good data error
            data1 = f[sid]['data'][:].astype('float32')
            if data1.shape[0] == 0:
                continue
            else:
                infe = model(data1).numpy()
                Terr_cases = MAE(data1[:,-3:,0], infe[:,-3:,0], ax=1)

            # bad data error
            data2 = f[sid]['test_data'][:].astype('float32')
            if data2.shape[0] == 0: 
                terr_cases = []
            else:
                infe = model(data2).numpy()
                terr_cases = MAE(data2[:, -3:, 0], infe[:, -3:, 0], ax=1)

            grp = fo.create_group(sid)
            grp.create_dataset('PassMAE', data = Terr_cases)
            grp.create_dataset('WarmMAE', data = terr_cases)
#%% main 
if __name__ == '__main__':
    # load data    
    data_file = '../data/9_input_10min/All_y2024.h5'
    f = h5py.File(data_file, 'r')

    # settings    
    exp_path = 'experiments/AE_10min.yml' #!!!!
    sub_list = ['Pre_M30_f32k3']

    for sub_exp in sub_list:
        # load model 
        model = create_model(exp_path, sub_exp)

        #output folder
        folder = f'verify/{sub_exp}'
        Path(folder).mkdir(parents=True, exist_ok=True)
        
        # main
        #stations_mae_txt(f)
        save_cases_mae(f)
        print(sub_exp, 'done')