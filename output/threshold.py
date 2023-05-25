# calculate thershold of all station
import numpy as np
import h5py


def _load_staion_pass(sid):
    exp_dir = 'AE_station_24'
    yr = '20t22' if sid == '467270' else '18t21'

    pred_path = f'{exp_dir}/AE_{sid}/{sid}_18t21_good.npy'
    pred_g = np.load(pred_path)

    obs_path = f'../data/station_ds/{sid}_H24_{yr}.h5'
    with h5py.File(obs_path, 'r') as f:
        obs_g = f['good/temp'][:]
        if len(pred_g) != len(obs_g):
            print(sid, 'data did not fit!')
    return pred_g, obs_g


def create_all_staion_th_file():
    n = 3
    
    with open(f'threshold/HR24_th_list_{n}.txt', 'w') as fout:
        fout.write('ID      mean   std\n')
        with open('../data/StaList.txt', 'r') as fin:
            # skip first line of fin
            #fin.readline()
            # calculate threshold for each station
            for line in fin:
                sid = line.split()[0]
                pred, obs = _load_staion_pass(sid)
                errors = [np.mean(abs(yt[-n:]-yp[-n:])) for yt, yp in zip(obs, pred)]
                mean = np.mean(errors)
                std = np.std(errors)
                fout.write(f'{sid} {mean:5.3f} {std:5.3f}\n')

create_all_staion_th_file()

#%% 
def _load_staion_both(sid):
    pred_path = f'{exp_dir}/{sid}_18t21_good.npy'
    pred_g = np.load(pred_path)
    pred_path = f'{exp_dir}/{sid}_18t21_bad.npy'
    pred_b = np.load(pred_path)
    
    obs_path = f'../data/station_ds/{sid}_H24_18t21.h5'
    with h5py.File(obs_path, 'r') as f:
        obs_g = f['good/temp'][:]
        obs_b = f['bad/temp'][:]
        time_b = f['bad/time'][:]
    return pred_g, obs_g, pred_b, obs_b, time_b


def one_station_th_and_noPass(sid, n):
    with open(f'AE_station_24/threshold/th{n}_3std_{sid}.txt', 'w') as fout:
        pred_g, obs_g, pred_b, obs_b, time_b = _load_staion_both(sid)
        errors = [np.mean(abs(yt[-n:]-yp[-n:])) for yt, yp in zip(obs_g, pred_g)]
        # get th
        mean = np.mean(errors)
        std = np.std(errors)
        th = mean + 3*std
        
        # get time which MAE > th
        case = len(time_b)
        fout.write('ID      mean   std    th cases\n')
        fout.write(f'{sid} {mean:5.3f} {std:5.3f} {th:5.3f} {case}\n')
        fout.write('-------------------------------------\n')
        num = 0 
        for i in range(case):
            e = np.mean(abs(pred_b[i][-n:]-obs_b[i][-n:]))
            if e > th:
                ts = time_b[i].decode('utf-8')
                fout.write(f'{ts} {e:5.3f}\n')
                num += 1
        recall = f'{1-num/case:4.2f}'
        fout.write(f'recall: {recall}\n')
        print(sid, recall)
                

#for station in stations:
#    exp_dir = f'AE_station_24/AE_{station}'
#    one_station_th_and_noPass(station, n=3)
