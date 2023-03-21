import argparse, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pathlib import Path
from modules.ymal_reader import parse_exp_settings
from modules.training_helper import get_tf_datasets
from modules.model_trainer import train_model
import tensorflow as tf
import importlib 


def environment_setting(GPU, GPU_limit):
    if GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = GPU
        
    # tf.get_logger().setLevel('ERROR')
    # restrict the memory usage
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_virtual_device_configuration(
            gpu,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_limit)]
        )

def create_model(model_name, model_setting):
    print(f'Create model {model_name}')
    model_class = importlib.import_module(f'Models.autoencoder.{model_name}')
    return model_class.Model(**model_setting)
    

def main(exp_path, omit_completed):
    # parse yaml to get experiment settings 
    exp_list = parse_exp_settings(exp_path)
    
    for sub_exp_settings in exp_list:
        exp_name = sub_exp_settings['experiment_name']
        sub_exp_name = sub_exp_settings['sub_exp_name']
        log_path = f'logs/{exp_name}/{sub_exp_name}'
        
        print(f'Executing sub-experiment: {sub_exp_name} ...')
        if omit_completed and os.path.isdir(log_path):
            print('Sub-experiment already done before, skipped ~~~~')
            continue

        # log and model saved path setting
        summary_writer = tf.summary.create_file_writer(log_path)
        model_save_path = f'Models/saved_weight/{exp_name}/{sub_exp_name}'
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        
        # load data and create model. 
        datasets = get_tf_datasets(**sub_exp_settings['data'])
        model = create_model(sub_exp_settings['model_name'], sub_exp_settings['model_setting'])
        
        # training.
        train_model(
            model,
            datasets,
            summary_writer,
            model_save_path,
            **sub_exp_settings['train_setting'],
        )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_path', help='name of the experiment setting')
    parser.add_argument('--GPU_limit', type=int, default=20000)
    parser.add_argument('-GPU', type=str, default='')
    parser.add_argument('--omit_completed', action='store_true')
    args = parser.parse_args()
    
    environment_setting(args.GPU, args.GPU_limit)
    main(args.exp_path, args.omit_completed)
