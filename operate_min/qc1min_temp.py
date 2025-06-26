# Standard library imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
#import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from datetime import datetime, timedelta
from module.data import data_loader, get_timeserise, ScodeConverter

# Configure logging
def setup_logger(filename):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / filename
    
    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    return logging.getLogger()


def daily_AE(date, logger):
    # load data
    grouped = data_loader(f'{data_folder}/{date[:4]}/{date}')

    # start AE check
    for code, group in grouped:
        # Convert Station Code to SID
        sid = convertor(code)
        if sid is None:
            print(f"Station code {code} not found.")
            continue
        
        # load AE model
        model_path = f"tf_model/{sid}"
        if Path(model_path).exists():
            model = tf.saved_model.load(f"tf_model/{sid}").signatures["serving"]
        else:
            print(f"{sid} model not exists.")
            continue

        # get time serise for model input
        test_temp, test_time = get_timeserise(group)
        if len(test_temp) == 0:
            print(f'{sid} no data')
            continue
        
        # inference
        infer_temp = model(input_data = test_temp)['output_0'][:,:,0]
        

        # check last 3 min MAE, get time result over 0.1
        mae = np.mean(np.abs(infer_temp[:,-3:] - test_temp[:,-3:]), axis=1)
        no_pass = mae > 0.1
        NP_times = test_time[no_pass]

        if len(NP_times) > 0:
            logger.error(f'{sid}, P:{len(test_time)}, NP:{NP_times}')
            # save data
            NP_temp  = test_temp[no_pass]
            NP_infer = infer_temp[no_pass]
            np.savez(f'logs/{date}_{sid}', time=NP_times, temp=NP_temp, infer=NP_infer)
        else:
            print(f'{sid}, P:{len(test_time)}')


#%% 
if __name__ == '__main__':
    #parser = argparse.ArgumentParser()
    #parser.add_argument('date', type=str, help='date to check. (yyyymmdd)')
    #args = parser.parse_args()

    # setting
    data_folder = '/NAS-Lswinhoii/T1/RawData/PRI/m_pri/'
    # set station code covertor to get id    
    convertor = ScodeConverter()
    
    # main ----------------------------------------------
    #target_date = args.date
    t0 = '20250101'
    t1 = '20250615'
    
    # Setup logger
    logger = setup_logger('2025_M01M06')
    #logger.info(f"Starting QC check for date: {target_date}")
    
    # start main code
    while t0 <= t1:
        try:
            daily_AE(t0, logger)
            print(f"QC check completed for date: {t0}")
        except Exception as e:
            print(f"Error during QC check: {str(e)}")

        t0 = (datetime.strptime(t0, '%Y%m%d') + timedelta(days=1)).strftime('%Y%m%d')
