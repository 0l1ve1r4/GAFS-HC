import os
from src.utils import Utils
from src.cpp_converter import call_nbayes
from src.dataset import Dataset
import multiprocessing

class CrossValidation:

    def __init__(self, filepath: str, train_files: list, test_files: list) -> None:
        """Initialize the CrossValidation class."""
        
        self.filepath = filepath
        self.folder = filepath.split('/')[0]
        
        self.train_files = train_files
        self.test_files = test_files
        
        self.utils = Utils()
        self.num_folds = 5
        
        os.makedirs(f"{self.folder}/temp", exist_ok=True)

    def save_best_chromosome(self, best_chromosome: list) -> float:
        
        train_paths, test_paths = self.chromosome_to_file(best_chromosome, 0)
        fitness = self.cross_validation_5_fold(train_paths, test_paths)
        
        os.makedirs(f"{self.folder}/best_chromosome", exist_ok=True)
        for i in range(len(train_paths)):
            os.system(f"cp {train_paths[i]} {self.folder}/best_chromosome/train_{i}.arff")
            os.system(f"cp {test_paths[i]} {self.folder}/best_chromosome/test_{i}.arff")
        
        os.system(f"rm -rf {self.folder}/temp")
        return fitness
        
    def cross_validation_5_fold(self, train_files: list, test_files: list, mlnp:str = 'n', usf:str = 'n') -> float:
        """Perform 5-fold cross-validation using call_nbayes function."""
        
        if len(train_files) != 5 or len(test_files) != 5:
            raise Exception("Error: You must provide exactly 5 training files, 5 test files, and 5 result files.")
        
        results = []
        for i in range(5):
            result = call_nbayes(train_files[i], test_files[i], "./", mlnp, usf)
            results.append(result)
        
        return float( sum(results) / 5 )
    
    def evaluate_population(self, multiprocessing_index, chromosome: list) -> list:
        """Evaluate the population using the 5-fold cross-validation. (Linear)"""
        
        train_paths, test_paths = self.chromosome_to_file(chromosome, multiprocessing_index)
        
        return self.cross_validation_5_fold(train_paths, test_paths)

    def evaluate_population_parallel(self, population: list):
        """Evaluate the population using multiprocessing.""" 
        
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.starmap(self.evaluate_population, enumerate(population))
        return results
    
    def chromosome_to_file(self, chromosome: list, multiprocessing_index: int) -> None:
        """Given a binary chromosome, select the attributes and objects from the dataset and save them in a new file."""
        
        chromossome_train_files = []
        chromossome_test_files = []
        
        for i in range(5):
            train_dataset = Dataset(self.train_files[i])
            test_dataset = Dataset(self.test_files[i])
            
            train_dataset.dataset_dict['attributes'] =  self.select_attributes(train_dataset.dataset_dict['attributes'], chromosome)
            test_dataset.dataset_dict['attributes'] = self.select_attributes(test_dataset.dataset_dict['attributes'], chromosome)
            
            train_dataset.dataset_dict['data'] = self.select_objects(train_dataset.dataset_dict['data'], chromosome)
            test_dataset.dataset_dict['data'] = self.select_objects(test_dataset.dataset_dict['data'], chromosome)
                        
            train_dataset.save_dataset(f"{self.folder}/temp/train_{multiprocessing_index}_{i}.arff")
            test_dataset.save_dataset(f"{self.folder}/temp/test_{multiprocessing_index}_{i}.arff")
            
            chromossome_train_files.append(f"{self.folder}/temp/train_{multiprocessing_index}_{i}.arff") 
            chromossome_test_files.append(f"{self.folder}/temp/test_{multiprocessing_index}_{i}.arff")

        return chromossome_train_files, chromossome_test_files

    def select_attributes(self, attributes: list, chromosome: list) -> list:
        """Select the attributes from the list of attributes based on the chromosome."""
        
        selected_attributes = []
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                selected_attributes.append(attributes[i])
        # @attribute class
        selected_attributes.append(attributes[-1]) 
        
        return selected_attributes
    
    def select_objects(self, objects: list, chromosome: list) -> list:
        """Select the objects from the list of objects based on the chromosome."""
        
        selected_objects = []
        
        for obj in objects:
            selected_obj = []
            for i in range(len(chromosome)):
                if chromosome[i] == 1:
                    selected_obj.append(obj[i])
            selected_obj.append(obj[-1])
            selected_objects.append(selected_obj)
        
        return selected_objects
