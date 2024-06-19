# ==============================================================================
# Main file to manipulate the dataset 
# 
#
# Author: Guilherme Santos
# Last edited: 2023-01-23
# ==============================================================================

import arff

from sklearn.preprocessing import KBinsDiscretizer
from src.utils import Utils

class Dataset: 
    """This class is used to extract the dataset information, such as the dataset name, 
    the dataset attributes and the dataset data itself."""

    # ==============================================================================
    # Constructor, all the variables and the algorithm are initialized here
    # ==============================================================================
    
    def __init__(self, file_path: str) -> None:

        self.utils = Utils() # Debugging object
        
        try:
            self.dataset_dict = arff.load(open(file_path, 'r')) # a dictionary with the dataset information

        except Exception as e:
            self.utils.debug("Error reading the dataset.", type="error")
            self.utils.debug(f"File path: {file_path}", type="error")
            print(e)
            exit()

        self.dataset_description = self.dataset_dict['description'] # a inlined string

        self.dataset_name = self.dataset_dict['relation'] # a inlined string

        self.dataset_attributes = self.dataset_dict['attributes'] #list of tuples [('attribute_name', 'value), ('', '')] both strings

        self.dataset_objects = self.dataset_dict['data'] #list of lists [[value, value, value], [value, value, value]] all strings

        self.attribute_class = self.dataset_attributes[-1][0] # the last attribute is the class
        
        self.only_attributes = [attr[0] for attr in self.dataset_attributes] # list of the attributes names

    def save_dataset(self, path: str):
        """Save the dataset in a new file."""
        
        try:            
            with open(path, 'w') as f:
                f.write('@relation ' + self.dataset_name + '\n')
                
                for attribute_name, variation in self.dataset_dict['attributes']:
                    variation = ','.join(variation)
                    f.write("@attribute " + attribute_name + " {" + variation + "}\n")
                    
                f.write("@DATA\n")
                    
                for data in self.dataset_dict['data']:
                    data = [str(d) for d in data]
                    data = ','.join(data)
                    f.write(data + '\n')
        
        except Exception as e:
            self.utils.debug("Error saving the dataset.", type="error")
            self.utils.debug(f"File path: {path}", type="error")
            print(e)
            exit()
            


    def get_dataset_info(self):
        return self.dataset_name, self.dataset_attributes, self.dataset_objects

    def get_numeric_categorical_info(self):
        numeric_indices = []
        categorical_indices = []
        for i, (attr_name, attr_type) in enumerate(self.dataset_attributes):
            if 'numeric' in attr_type.lower():
                numeric_indices.append(i)
            else:
                categorical_indices.append(i)
        return numeric_indices, categorical_indices
    
    def read_dataset(self):
        data = []
        a_class = []
        dist_class = []
        header_attr = []
        f_type = []

        for line in self.dataset_objects:
            v_value = []
            for i, value in enumerate(line[:-1]):
                v_value.append(float(value))
                if len(f_type) <= i:
                    f_type.append(1 if isinstance(value, (int, float)) else 2)
            classe = line[-1]
            if classe not in dist_class:
                dist_class.append(classe)
            a_class.append(classe)
            data.append(v_value)

        return data, a_class, dist_class, header_attr, f_type
  
