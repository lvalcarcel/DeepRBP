# create_data.py
import os
import sys
import timeit 
import datetime
import argparse

ruta_utils = '/scratch/jsanchoz/ML4BM-Lab/DeepRBP/utils'
os.environ['PYTHONPATH'] = ruta_utils + ':' + os.environ.get('PYTHONPATH', '')
print('utils')
import utils.Utils as utils
from utils import Generate_Datasets as gd

def create_data(args, path_data):    
    utils.check_create_new_directory(path_data+'/splitted_datasets')
    print('[create_data] Creating the Training and Test folders and dividing the samples per tumor_type')
    gd.create_datasets(path_data=path_data, chunksize=args.chunksize, select_genes = args.select_genes)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Datasets')
    parser.add_argument('--chunksize', type=int, help='Number of samples to process per chunk', required=True)
    parser.add_argument('--select_genes', type=str, help='Genes selected for modelling', default=None)
    parser.add_argument('--path_data', type=str, default='/scratch/jsanchoz/ML4BM-Lab/DeepRBP/data', help='Path for the data directory')
    args = parser.parse_args()
    
    print(f'path_data is: {path_data}')
    start_time = timeit.default_timer()
    
    create_data(args, path_data)
    duration = datetime.timedelta(seconds=timeit.default_timer() - start_time)
    print('[create_data] Create the Data ... -> DONE')
    print('[create_data] The time of execution is:', str(duration))