import argparse, os
import importlib
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


def _get_threshold(sid):
    th_file = 'th_24h3.txt'
    with open(th_file, 'r') as f:
        next(f)
        for l in f:
            arr = l.strip().split()
            if arr[0] == sid:
                return float(arr[3])
    print(f'{sid} not found in threshold file.')
    return None


def _load_24hr_temp(sid, time_check):
    temp24 = []
    t24 = datetime.strptime(time_check, '%Y%m%d%H')
    t1 =  t24 - timedelta(hours=23)
    while t1 <= t24:
        data_path = '/NAS-DS1515P/users1/T1/DATA/QC/Temp/'   #!!!!! modify input path
        yymm = t1.strftime('%Y-%m')
        tstr = t1.strftime('%Y%m%d%H')
        fn = Path(f'{data_path}/{yymm}/bias_{tstr}00.txt')
        if not fn.exists():
            print(f'File not found. Datetime: {tstr}')
            return None
        
        with open(fn, 'r') as f:
            line = next((l for l in f if l.startswith(sid)), None)
            if line:
                temp24.append(float(line.split()[1]))
            else:
                print(f'Station {sid} not in file {fn.name}.')
                return None
        t1 += timedelta(hours=1)
    return np.array(temp24)[np.newaxis, ...]


def _create_model(model_name, weight_path):
    model = importlib.import_module(model_name).Model()
    model.load_weights(weight_path).expect_partial()
    return model


def AE_check(sid, t24):
    obs = _load_24hr_temp(sid, t24)
    threshold = _get_threshold(sid)

    if (obs is not None) and (threshold is not None):
        # create model and predict
        model = _create_model('AE_2_0', f'./model/{sid}/AE')
        pred = np.squeeze(model(obs))

        # calculate last 3 hours error
        mae = np.mean(np.abs(pred[-3:] - np.squeeze(obs)[-3:]))

        # check threshold
        result = 'P' if mae <= threshold else 'A'
        print(f'AE success. {sid} {t24} MAE:{mae:.2f}, th:{threshold:.2f}, result: {result}')
    else:
        print('AE fail')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sid', type=str)
    parser.add_argument('time_check', help='yyyymmddhh (e.g. 2021090123)', type=str)
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

    AE_check(args.sid, args.time_check)
