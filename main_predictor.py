import argparse
import joblib
import json
import os
import torch
import numpy as np
import timeit
import datetime
from collections import namedtuple

from utils.Utils import check_existing_trained_model, load_or_train_model, get_data, perform_predictions_on_test_data
from utils.Config import Config
from modelsNN.modelsNN import DeepRBP

def main_predictor(args):
    path_data = args.path_data

    # Load the config of this experiment
    config = Config(
                    args.max_node, args.num_hidden_layers, args.learning_rate, args.batch_size, args.epochs, args.same_num_nodes, args.node_divition_factor,
                    args.activation_layer, args.sel_optimizer, args.model_name, args.source_train, args.train_tumor_types, args.test_tumor_types
                    )
    print('[main] The config of this experiment was loaded... -> DONE\n')
    print(f'configuration: {config.get_config()}')
    
    ### Check if there is an already trained model with these settings
    path_save_files = path_data + '/trained_models/' + f'model_{config.max_node}N_{config.num_hidden_layers}HL_{config.node_divition_factor}f' 
    is_model_available = check_existing_trained_model(args.use_trained_model, path_save_files, config)
    print(f'is_model_available: {is_model_available}')
    print('[main] Check if there is an already trained model with these settings ... -> DONE\n')
    
    # Get the DeepRBP predictor model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DeepRBP(n_inputs=1282, n_outputs=11462, config=config, device=device)
    model.to(device)
    print('[main] 2) Get the model ... -> DONE\n')
    
    # Train the DeepRBP predictor model if there is not an already trained model
    print('[main] Train the DeepRBP predictor model if there is not an already trained model')
    model, data_scale = load_or_train_model(is_model_available, model, path_save_files, device, path_data, config, optimizer=args.sel_optimizer, save_model=args.save_model)
    
    # Do predictions on test data TCGA and GTEX
    data_test = get_data(config, path_data, set_mode='test')
    results_tcga, results_gtex = perform_predictions_on_test_data(config, data_test, data_scale, model, device, path_data, path_result=args.path_result, plot_results=True)
    results_tcga.to_csv("results_tcga.csv", index=False)
    results_gtex.to_csv("results_gtex.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepRBP predictor parameters')
    parser.add_argument('--sel_optimizer', type=str, default='adamW', help='Optimizer to be used.')
    parser.add_argument('--source_train', type=str, default='TCGA', help='Data source to train.')
    parser.add_argument('--max_node', type=int, default=1024, help='Number of nodes in first hidden layer')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--activation_layer', type=str, default='relu', help='Activation layer function to be used in the hidden layers.')
    parser.add_argument('--node_divition_factor', type=int, default=8, help='The value by which the number of nodes is divided from one hidden layer to the next.')
    parser.add_argument('--same_num_nodes', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use the same number of nodes in each hidden layer.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--model_name', type=str, default='deepRBP', help='Project name.')
    parser.add_argument('--train_tumor_types', type=str, default='all', help='List of tumor types to train.')
    parser.add_argument('--test_tumor_types',  type=str, default='all', help='List of tumor types to test.')
    parser.add_argument('--path_result', type=str, default='/scratch/jsanchoz/ML4BM-Lab/DeepRBP/results', help='Path for the results directory')
    parser.add_argument('--path_data', type=str, default='/scratch/jsanchoz/data', help='Path for the data directory')
    parser.add_argument('--use_trained_model', type=lambda x: x.lower() == 'true', default=True, help='Use a pre-trained model if available.')      
    parser.add_argument('--save_model', type=lambda x: x.lower() == 'true', default=False, help='Whether to save or not the created model.')      
    parser.add_argument('--plot_results', type=lambda x: x.lower() == 'true', default=False, help='Whether to plot the isoform abundance percentage and the plot real vs pred scat.')
    args = parser.parse_args()
    main_predictor(args) 
    
      