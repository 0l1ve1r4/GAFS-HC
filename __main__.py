from src.__main__ import __main__
from src.Dataframe import Dataframe
import os
import yaml

# ===============================================================================================================================
# Main file init to run the genetic algorithm, read the config.json file and change the variables there to run the algorithm    #
# ===============================================================================================================================

if __name__ == "__main__":    
    with open("./config.yaml", "r") as FILE:
        config = yaml.safe_load(FILE)
        
    datasets_path = './data'
    dirs = os.listdir(datasets_path)    
    
    if config['remove_program_junk']:
        for d in dirs: 
            try:
                file_type = d.split('_')[-2]
                if file_type == "fold":
                    os.remove(f'{datasets_path}/{d}')
            except:
                pass
                
    datasets = config['dataset_path']
    for dataset in datasets:
        
        if config["NNwGA"]['clear_training_data']:
            os.system("rm -rf ./generated-files/training_data.txt")
            os.system("touch ./generated-files/training_data.txt")
        
        for index in range(config['num_executions']):    
                __main__(config, dataset)
        
