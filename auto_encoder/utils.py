from auto_encoder.logger import logging
from auto_encoder.exception import AutoencoderException
import os,sys
import yaml
import numpy as np
import pickle
from typing import List


def get_data_path()->List[str]:
    try:
        logging.info(f'arranging file path of all chrome history csv files in a list!')
        file_paths = []
        data_dir_path = 'C:/Users/Ranjit Singh/Desktop/Data'
        elemnts = os.listdir(data_dir_path)
        for item in elemnts:
            if item.endswith('.csv'):
                file_path = os.path.join(data_dir_path,item)
                file_paths.append(file_path)
        return file_paths    
    except Exception as e:
        raise AutoencoderException(e,sys)
    

def save_text_file(file_path:str, text:str):
    try:
        logging.info(f"saving the cleaned text file")
        dirNames  = os.path.dirname(file_path)
        os.makedirs(dirNames,exist_ok=True)
        with open(file_path,'w') as file:
            file.write(text)
        logging.info(f"successfully saved the cleaned text file !")
    except Exception as e:
        raise AutoencoderException(e,sys)
    

    
def write_yaml_file(file_path:str,data:dict):
    try:
        file_dir = os.path.dirname(file_path) 
        os.makedirs(file_dir,exist_ok=True)  
        logging.info('writing Validationroport.yaml file')
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise AutoencoderException(e,sys)
    
# to save the transform data    
def save_numpy_array_data(file_path:str,array):
    try:
        logging.info('saving transformed data !')
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir,exist_ok=True)
        np.save(open(file_path,'wb'),array)
    except Exception as e:
        raise AutoencoderException(e, sys)
    
def load_numpy_array(file_path:str)->np.array:
    try:
        logging.info(f"loading numpy array from {file_path}")
        Data = np.load(open(file_path,'rb'))
        return Data
    except Exception as e:
        raise AutoencoderException(e,sys)
    
def save_object(file_path:str,obj):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path,exist_ok=True)
        pickle.dump(obj,open(file_path,'wb'))
        logging.info(f"saved object file path :- {file_path}")
    except Exception as e:
        raise AutoencoderException(e,sys)
  

def load_obj(file_path:str):
    try:
        if not os.path.exists(file_path):
            raise Exception(f"file not found error {file_path}")
        else:
            model_obj = pickle.load(open(file_path,'rb'))
            logging.info(f"successfully load the obj :- {file_path}")
        return model_obj
    except Exception as e:
        raise AutoencoderException(e,sys)