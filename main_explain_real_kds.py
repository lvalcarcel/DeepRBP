import argparse
import pandas as pd
import torch
from utils.Utils import load_model, generate_data, evaluate_model, prepare_data, check_data_exists

def main(readme, path_data, experiment, rbp_interest, getBM, path_model, path_result):  
    print('[main] 1) Create Data \n')
    print('[main] 1.1) Check if Data is already created ... \n')
    generate_data_flag = check_data_exists(path_data, experiment)
    print('[main] 1.1) Check if Data is already created ... -> DONE')
    
    condition1 = experiment+'_control'
    condition2 = f'{experiment}_{rbp_interest}_kd'
    df_filereport = pd.read_csv(path_data+f'/{readme}.txt', delimiter='\t')
    
    if generate_data_flag:
        generate_data(df_filereport, condition1, condition2, path_data, path_model)
    
    print('[main] 2) Get data processed (real control and kd data) ... -> DONE \n')
    data_control, data_kd = prepare_data(path_model, f'{path_data}/datasets', condition1, condition2)
    df_rbps_control = data_control.df_scaled_rbps
    df_labels_control = data_control.df_labels
    df_gns_control = data_control.df_gns
    df_rbps_kd = data_kd.df_scaled_rbps
    df_labels_kd = data_kd.df_labels
    df_gns_kd = data_kd.df_gns
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('[main] The device selected is:', device)
    model = load_model(path_model,df_rbps_control, df_labels_control, device)

    print('[main] 3) Load the Trained Model ... -> DONE \n')
    print('[main] 3.1) Evaluate the model prediction ...')
    evaluate_model(model, df_rbps_control, df_gns_control, df_labels_control, df_rbps_kd, df_gns_kd, df_labels_kd, device, condition1, condition2)
    print('[main] 3.1) Evaluate the model prediction in (real control and kd data)... -> DONE')
  
    print('[main] 4) Perform explainability from control data with DeepLIFT ...')
    path = path_result+f'/explainability_cell_line_kds/{path_model.rsplit("/", 1)[-1]}/{experiment}'

    df_deeplift_scores_GxRBP = perform_deeplift_pipeline(df_scaled_test, test_labels, test_gn, model, path_save, getBM, select_reference='knockout', method='tstat')


    print('[main] 4) Perform explainability from control data ... -> DONE')



if __name__ == '__main__':
parser = argparse.ArgumentParser(description='DeepRBP explain in real kds parameters')
parser.add_argument('--readme_txt', type=str, default='filereport_read_run_PRJEB39343_tsv', help='readme with sample experiment description')
parser.add_argument('--path_data', type=str, default='/scratch/jsanchoz/validation_real_kds/prjeb39343_gencode_23', help='Path to data')
parser.add_argument('--experiment', type=str, default='HFE', help='Name of the experiment')
parser.add_argument('--rbp_interest', type=str, default='MBNL1', help='Name of the RBP of interest')
parser.add_argument('--path_model', type=str, default='/scratch/jsanchoz/ML4BM-Lab/DeepRBP/data/trained_models/model_1024N_2HL_8f', help='Path to model')
parser.add_argument('--path_result', type=str, default='/scratch/jsanchoz/ML4BM-Lab/DeepRBP/results', help='Path to result')
args = parser.parse_args()

    # Load the getBM set
    getBM = pd.read_csv(f"{args.path_model.rsplit('/', 2)[0]}/processed/getBM_reduced.csv", index_col=0)
    # Reorder getBM (in the last model version the data is returned with the transcript values sorted)
    getBM = getBM.sort_values(by='Transcript_ID').reset_index(drop=True)
    # Call main
    main(readme=args.readme_txt, path_data=args.path_data, experiment=args.experiment, rbp_interest=args.rbp_interest, getBM=getBM, path_model=args.path_model, path_result=args.path_result)

 