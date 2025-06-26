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
from inference import plot_T_RH


#%% calculate
def MAE(x, y, ax=""):
    if ax == "":
        return np.mean(np.abs(x-y))
    else:
        return np.mean(np.abs(x-y), axis=int(ax))


class CreateModel():
    def __init__(self, exp_path):
        # load setting
        exp_settings = parse_exp_settings(exp_path)[0]
        self.subexp_template = exp_settings['sub_exp_name']
        
        # load model 
        model_settings = exp_settings['model_setting']
        model_name = model_settings['name']
        model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
        self.model = model_class.Model(**model_settings)
    
    def baseModel(self, expname):
        path = f'Models/saved/AE_10min/{expname}/AE'
        self.model.load_weights(path).expect_partial()
        return self.model
    
    def __call__(self, sid):
        # load model weight
        subexp_name = self.subexp_template.replace("sid", sid)
        weight_path = f'Models/saved/AE_10min/{subexp_name}/AE'
        self.model.load_weights(weight_path).expect_partial() 
        return self.model    
    

#%% result
def save_cases_mae(ds, fout):
    lines = open(namefile, 'r').readlines()
    for line in lines:
        sid = line.split()[0]
        # load model 
        model = modelcreator(sid)
        
        # good data error
        data1 = ds[sid]['data'][:].astype('float32')
        if data1.shape[0] == 0:
            continue
        else:
            infe = model(data1).numpy()
            PassMAE = MAE(data1[:,-3:,0], infe[:,-3:,0], ax=1)

        # bad data error
        data2 = ds[sid]['test_data'][:].astype('float32')
        if data2.shape[0] == 0: 
            terr_cases = []
        else:
            infe = model(data2).numpy()
            WarnMAE = MAE(data2[:, -3:, 0], infe[:, -3:, 0], ax=1)

        # save 
        np.savez(fout / f'{sid}.npz', PassMAE=PassMAE, WarnMAE=WarnMAE)

def verse_fig(ds, sid):
    data = ds[sid]['data'][:].astype('float32')

    # base model 
    infer_base = baseModel(data).numpy()
    mae_b = MAE(data[:, -3:, 0], infer_base[:, -3:, 0], ax=1)
    ava = mae_b > 0.1 
    data = data[ava,...]
    print(len(data), 'avaliable')
    infer_base = infer_base[ava,...]
    
    # transfer model
    model = modelcreator(sid)
    infer_trans = model(data).numpy()

    for i in np.random.choice(data.shape[0], 5):
        x = data[i,...]
        y1 = infer_base[i,...]
        y2 = infer_trans[i,...]
        t = ds[sid]['time'][i].decode()

        plot_T_RH(x, y1, t, sid, f'{folder}/{sid}_{t}_b')
        plot_T_RH(x, y2, t, sid, f'{folder}/{sid}_{t}_t')



#%% main 
if __name__ == '__main__':
    # settings    
    folder = Path('verify/Pre_M30_f32k3') #!!!!
    namefile = folder / 'namelist.txt'
    exp_path = 'experiments/transfer_bk.yml' #!!!!
    modelcreator = CreateModel(exp_path)
    
    baseModel = modelcreator.baseModel('Pre_M30_f32k3')

    # load data and set out file
    data_file = '../data/9_input_10min/All_y2024.h5'
    ds = h5py.File(data_file, 'r')

    verse_fig(ds, '467550')
    verse_fig(ds, 'C0H9A0')
    #save_cases_mae(ds, folder/'mae_t')