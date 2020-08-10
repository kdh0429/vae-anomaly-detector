"""
Main experiment
"""
import json
import os
import argparse
import torch
from torch.utils.data import DataLoader
from configparser import ConfigParser
from datetime import datetime

from vae.vae import VAE
from utils.data import CollisionDataset
from constants import MODELS

import wandb

def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser(description='VAE collision detection')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument(
        '--globals', type=str, default='./configs/globals.ini', 
        help="Path to the configuration file containing the global variables "
             "e.g. the paths to the data etc. See configs/globals.ini for an "
             "example."
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help="Id of the model configuration file. If this argument is not null, "
             "the system will look for the configuration file "
             "./configs/{args.model}/{args.model}{args.config}.ini"
    )
    parser.add_argument(
        '--restore', type=str, default=None, 
        help="Path to a model checkpoint containing trained parameters. " 
             "If provided, the model will load the trained parameters before "
             "resuming training or making a prediction. By default, models are "
             "saved in ./checkpoints/<args.model><args.config>/<date>/"
    )
    parser.add_argument(
        '--n_epochs', type=int, default=None,
        help="Maximum number of training iterations."
    )
    return parser.parse_args()


def load_config(args):
    """
    Load .INI configuration files
    """
    config = ConfigParser()

    # Load global variable (e.g. paths)
    config.read(args.globals)

    # Path to the directory containing the model configurations
    model_config_dir = os.path.join(config['paths']['configs_directory'], '{}/'.format(args.model))

    # Load default model configuration
    default_model_config_filename = '{}.ini'.format(args.model)
    default_model_config_path = os.path.join(model_config_dir, default_model_config_filename)
    config.read(default_model_config_path)

    if args.config:
        model_config_filename = '{}{}.ini'.format(args.model, args.config)
        model_config_path = os.path.join(model_config_dir, model_config_filename)
        config.read(model_config_path)

    config.set('model', 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if args.n_epochs is not None:
        config.set('training', 'n_epochs', str(args.n_epochs))
    return config


def run(config, trainloader, validatonloader, testcollisionloader, testfreeloader, devloader=None):
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M')
    checkpoint_directory = os.path.join(
        config['paths']['checkpoints_directory'],
        '{}{}/'.format(config['model']['name'], config['model']['config_id']),
        current_time)
    os.makedirs(checkpoint_directory, exist_ok=True)

    input_dim = trainloader.dataset.input_dim_
    vae = VAE(input_dim, config, checkpoint_directory)
    vae.to(config['model']['device'])
    vae.fit(trainloader, validatonloader)
    vae.test(trainloader, testcollisionloader,testfreeloader)


if __name__ == '__main__':
    args = argparser()
    config = load_config(args)
    
    if config.getboolean("log", "wandb") is True:
        wandb.init(project="Anomaly Detection", tensorboard=False)
        wandb_config_dict = dict()
        for section in config.sections():
            for key, value in config[section].items():
                wandb_config_dict[key] = value
        wandb.config.update(wandb_config_dict)

    # Get data path
    data_dir = config.get("paths", "data_directory")
    train_data_file_name = config.get("paths", "train_data_file_name")
    train_csv_path = os.path.join(data_dir, train_data_file_name)
    train_data = CollisionDataset(
        train_csv_path)
    trainloader = DataLoader(
        train_data,
        batch_size=config.getint("training", "batch_size"),
        shuffle=True,
        drop_last=True,
        num_workers=8,
        pin_memory=True)

    validation_data_file_name = config.get("paths", "validation_data_file_name")
    validation_csv_path = os.path.join(data_dir, validation_data_file_name)
    validation_data = CollisionDataset(
        validation_csv_path)
    validationloader = DataLoader(
        validation_data,
        batch_size=validation_data.__len__(),
        shuffle=False,
        num_workers=1,
        pin_memory=False)

    test_collision_data_file_name = config.get("paths", "test_collision_data_file_name")
    test_collision_csv_path = os.path.join(data_dir, test_collision_data_file_name)
    test_collision_data = CollisionDataset(
        test_collision_csv_path)
    testcollisionloader = DataLoader(
        test_collision_data,
        batch_size=config.getint("training", "batch_size"),
        shuffle=False,
        num_workers=8,
        pin_memory=False)

    test_free_data_file_name = config.get("paths", "test_free_data_file_name")
    test_free_csv_path = os.path.join(data_dir, test_free_data_file_name)
    test_free_data = CollisionDataset(
        test_free_csv_path)
    testfreeloader = DataLoader(
        test_free_data,
        batch_size=config.getint("training", "batch_size"),
        shuffle=False,
        num_workers=8,
        pin_memory=False)

    run(config, trainloader, validationloader, testcollisionloader, testfreeloader)