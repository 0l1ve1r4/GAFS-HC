import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold
from src.utils import Utils
from src.dataset import Dataset

class Preprocessing: 

    def __init__(self) -> None:
        self.utils = Utils()

    def split_dataset(self, original_path: str, test_path: str, train_path: str) -> None:
        ' Split the dataset into 5 train and 5 test datasets, each one with 80% and 20% of the original dataset, respectively.'
        
        original_dataset = Dataset(original_path) # Create dataset object
        data = original_dataset.dataset_dict['data'] # Get dataset data
        num_folds = 5
            
        df = pd.DataFrame()
        df['data'] = [data[i][:-1] for i in range(len(data))] # Get data without classes
        df['class'] = [data[i][-1] for i in range(len(data))] # Get classes
        
        skf = StratifiedKFold(n_splits=num_folds)
        
        train_objects = []
        train_classes = []
        test_objects = [] 
        test_classes = []
          
        for train_index, test_index in skf.split(df['data'], df['class']):
  
            train_data = df['data'][train_index]
            train_class = df['class'][train_index]
            test_data = df['data'][test_index]
            test_class = df['class'][test_index]
            
            train_objects.append(train_data.values.tolist())
            test_objects.append(test_data.values.tolist())
            
            train_classes.append(train_class.values.tolist())
            test_classes.append(test_class.values.tolist())
        
        for i in range(1):
            for j in range(len(train_objects[i])):
                train_objects[i][j].append(train_classes[i][j])
                
            for j in range(len(test_objects[i])):
                test_objects[i][j].append(test_classes[i][j])
                
        for i in range(5):
            for j in range(len(train_objects[i])):
                if len(train_objects[i][j]) != len(train_objects[0][0]):
                    self.utils.debug(f"Error in train_objects[{i}][{j}]", "error")
                    exit()
        
        for fold in range(num_folds):
            original_dataset.dataset_dict['data'] = train_objects[fold]
            original_dataset.save_dataset(f"{train_path}_fold_{fold}.arff")
            original_dataset.dataset_dict['data'] = test_objects[fold]
            original_dataset.save_dataset(f"{test_path}_fold_{fold}.arff")            
        
    def discretize_data(self, dataset_path: str, output_path: str, num_partitions = 20) -> None:
        """
        Discretizes the data in the given dataset and saves the discretized dataset to the specified output path.
        """
        dataset = Dataset(dataset_path) 
        attributes = dataset.dataset_attributes
        
        data = dataset.dataset_objects
        
        classes = [data[i][-1] for i in range(len(data))] 
        data = np.array([data[i][:-1] for i in range(len(data))]) 
        data = data.astype(float)
        
        kbins = KBinsDiscretizer(n_bins=num_partitions, encode='ordinal', strategy='quantile')
        data_discretized = kbins.fit_transform(data)
        data_discretized = data_discretized.astype(int)
        
        data = data_discretized.tolist()
        variance_per_feature = [f'{i}' for i in range(num_partitions)]

        for i in range(len(data_discretized)):
            data[i].append(classes[i])  

        for j in range(len(attributes)):
            if attributes[j][0].upper() == 'CLASS':
                break
            attributes[j] = (attributes[j][0], variance_per_feature)
            
        dataset.dataset_dict['attributes'] = attributes
        dataset.dataset_dict['data'] = data 
        dataset.save_dataset(output_path)
        
        self.remove_root(output_path)


    def minimum_classes(self, dataset_path: str, output_path: str, minimum = 10) -> None:
        
        """ Remove classes with less than 10 instances or classes with only 'R' (root)
        if a class is removed, the function will be called again, until all classes have more than 10 instances."""

        self.utils.debug(f"Removing classes with less than {minimum} instances", "info")

        
        dataset = Dataset(dataset_path) # Create dataset object
        
        df = pd.DataFrame() # Objects, class. eg. [[1, 1, 2, ..., 'R.1.1']]
        df_attributes = pd.DataFrame() # @ATTRIBUTE class, eg. [['R.1.1', 'R.1.2', 'R.1.3']]
        
        # Class in the in the final of each row of @DATA
        df['class'] = [dataset.dataset_objects[i][-1] for i in range(len(dataset.dataset_objects))]
        df_attributes['attribute_class'] = (dataset.dataset_attributes[-1])[1]
        
        filtered_attributes = set()
        ordered_list = (dataset.dataset_attributes[-1])[1] 
        
        isrecursion = True
        while isrecursion:
            isrecursion = False
            
            # Get the leaf nodes to remove the classes with less than 10 instances first
            df_attributes['attribute_class'] = sorted(df_attributes['attribute_class'], key=len, reverse=True) 
            leaf_node = len(df_attributes['attribute_class'][0])

            for i in range(len(df_attributes)):
                
                # Check if isn´t a blank space or a class with only 'R'
                attribute_i = df_attributes['attribute_class'][i]
                
                for fixed_attribute in ordered_list:
                    if fixed_attribute == attribute_i:
                        index = ordered_list.index(fixed_attribute)
                                
                isvalid = any(letter.isnumeric() for letter in attribute_i)
                
                if not isvalid or (attribute_i, index) in filtered_attributes:
                    continue
                
                if len(attribute_i) < leaf_node and isrecursion:
                    i = 0
                    leaf_node = len(df_attributes['attribute_class'][0])
                    continue
                    
                count = 0
                                
                for j in range(len(df)):                                            
                        if attribute_i == df['class'][j]:
                            count += 1
                
                if (count < minimum):
                    
                    isrecursion = True 
                    old_class = attribute_i
                    new_class = '.'.join(old_class.split('.')[:-1])
                    
                    ordered_list[index] = new_class 

                    df_attributes.loc[i, 'attribute_class'] = new_class
                    df.loc[df['class'] == old_class, 'class'] = new_class
                    
                    # Debug purposes only, this make the log.txt file bigger and dirty
                    #self.utils.debug(f"[Node Change] (Size/Amount): ({len(attribute_i)}/{count}) | [{old_class}] -> [{new_class}]")
           
                else:
                    
                    ordered_list[index] = attribute_i
                    filtered_attributes.add((attribute_i, index))

        ordered_attributes = sorted(filtered_attributes, key=lambda x: x[1])
        ordered_list = [x[0] for x in ordered_attributes]
        
        filtered_attributes = sorted(filtered_attributes, key=lambda x: len(x[0]), reverse=True)
        filtered_attributes = [x[0] for x in filtered_attributes]
        
        data = self.remove_not_used_classes(dataset.dataset_objects, filtered_attributes)
        
        dataset.dataset_dict['attributes'][-1] = ('class', ordered_list)
        dataset.dataset_dict['data'] = data
        dataset.save_dataset(output_path)
        
        self.utils.debug(f"Classes with less than {minimum} instances removed", "info") 
        
    def remove_not_used_classes(self, objects: list, filtered_attributes: list) -> list:
        """Change objects classes based on filtered_attributes list."""
                
        result_vec = []
        
        for att_class in filtered_attributes:
            for i in range(len(objects)):
                if objects[i] is None:
                    continue
                
                if str(objects[i][-1]).startswith(att_class):
                    objects[i][-1] = att_class
                    result_vec.append((objects[i], i))
                    objects[i] = None

        result_vec = sorted(result_vec, key=lambda x: x[1])
        
        return [x[0] for x in result_vec]

        
    def remove_root(self, path):
        save_set = Dataset(path)
        attributes_class = save_set.dataset_dict['attributes'][-1][1]
        objects = save_set.dataset_dict['data']

        for i in range(len(attributes_class)):
            if attributes_class[i].startswith('R') or attributes_class[i].startswith('R.'):
                attributes_class[i] = '.'.join(attributes_class[i].split('.')[1:])

        for i in range(len(objects)):
            if objects[i][-1].startswith('R') or objects[i][-1].startswith('R.'):
                objects[i][-1] = '.'.join(objects[i][-1].split('.')[1:])

        save_set.dataset_dict['attributes'][-1] = ('class', attributes_class)
        save_set.dataset_dict['data'] = objects
        save_set.save_dataset(path)
