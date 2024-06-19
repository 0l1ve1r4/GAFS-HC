import os
import ctypes
import pandas as pd

def call_nbayes(training_dataset:str, test_dataset:str, result_file:str = './result.arff', mlnp:str = 'n', usf:str = 'n') -> float:
    """Call nbayes function from nbayes.so, read docs/GMNB_2009_Silla.pdf for more information.
    
    `Args:`
        - mlnp (char 'y' or 'n'): Mandatory Leaf Node Prediction
        - usf (char 'y' or 'n'): Usefulness
        - training_dataset (str): path
        - test_dataset (str): path
        - result_file (str): path
    """

    if not os.path.exists('./src/nbayes.so'):
        raise Exception("Error: nbayes.so not found. Please compile the call_nbayes.cpp file.")
    
    if os.name == 'nt':
        raise Exception("Error: nbayes.so is not compatible with Windows, try to compile to nbayes.dll.")

    nbayes_dll = ctypes.CDLL('./src/nbayes.so')

    nbayes_dll.call_nbayes.argtypes = [ctypes.c_char, ctypes.c_char, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    nbayes_dll.call_nbayes.restype = ctypes.c_float

    # Use 'b' to create bytes-like objects 
    mlnp = bytes(mlnp, 'utf-8')
    usf = bytes(usf, 'utf-8')
    training_dataset = bytes(training_dataset, 'utf-8')
    test_dataset = bytes(test_dataset, 'utf-8')
    result_file = bytes(result_file, 'utf-8')
    
    result = float(nbayes_dll.call_nbayes(mlnp, usf, training_dataset, test_dataset, result_file))
    
    return result




def cross_validation_5_fold(train_files: list, test_files: list, mlnp:str = 'n', usf:str = 'n') -> float:
    """Perform 5-fold cross-validation using call_nbayes function."""
    
    if len(train_files) != 5 or len(test_files) != 5:
        raise Exception("Error: You must provide exactly 5 training files, 5 test files, and 5 result files.")
    
    results = []
    for i in range(5):
        result = call_nbayes(train_files[i], test_files[i], "./", mlnp, usf)
        results.append(result)
    
    return ( sum(results) / 5 )
    
if __name__ == '__main__':
    
    try:
        df = pd.read_csv('results/GMNB_2009_Silla.csv')
    except Exception as e:
        print(e)
        df = pd.DataFrame(columns=['Dataset', 'HF', 'MLNP', 'USF', '5-fold'])
    
    train_files = []
    test_files = []
    results = []
    
    files = ['data/CellCycle_single.arff', 'data/Church_single.arff', 'data/Derisi_single.arff', 'data/Eisen_single.arff',
             'data/Expr_single.arff', 'data/Gasch1_single.arff', 'data/Gasch2_single.arff',
             'data/Sequence_single.arff', 'data/SPO_single.arff'
             ]
    
    
    for f in files:
        name = f.split('/')[-1].split('_')[0]
        folder = 'data/'

        train_files = []
        test_files = []

        for i in range(5):
            train_files.append(f'{folder}/{name}_train_fold_{i}.arff')
            test_files.append(f'{folder}/{name}_test_fold_{i}.arff')
        
    
        x = cross_validation_5_fold(train_files, test_files)
        df = pd.concat([df, pd.DataFrame([[name, x, 'n', 'n', 'True']], columns=['Dataset', 'HF', 'MLNP', 'USF', '5-fold'])], ignore_index=True)
        
    df = df.to_csv('results/GMNB_2009_Silla.csv', index=False)
