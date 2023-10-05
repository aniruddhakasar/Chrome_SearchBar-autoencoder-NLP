from auto_encoder.logger import logging
from auto_encoder.exception import AutoencoderException
from auto_encoder.entity import config_entity,artifact_entity
from auto_encoder import utils
import os,sys
import pandas as pd
import numpy  as np
import re

class DataValidation:
    def __init__(self,dataingestionartifact:artifact_entity.DataIngestionArtifact,
                 datavalidationconfig:config_entity.DataValidationConfig):
        try:
            self.dataingestionartifact = dataingestionartifact
            self.datavalidationconfig = datavalidationconfig
            self.validation_data = dict()
        except Exception as e:
            raise AutoencoderException(e,sys)
        
    def file_type(self,report_key:str):
        try:
            file_name = os.path.basename(self.dataingestionartifact.merge_df_path)
            if file_name.endswith('.csv'):
                self.validation_data[report_key] = '.csv file'
            else:
                self.validation_data[report_key] = 'unknown'
        except Exception as e:
            raise AutoencoderException(e,sys)
        

    def duplicate_record(self,report_key:str):
        try:
            df_file_path = self.dataingestionartifact.merge_df_path
            dataframe = pd.read_csv(df_file_path)
            no_of_sum = dataframe.duplicated().sum()
            self.validation_data[report_key] = str(no_of_sum)
        except Exception as e:
            raise AutoencoderException(e,sys)
        
    
    def columns(self,report_key:str):
        try:
            df_file_path = self.dataingestionartifact.merge_df_path
            df = pd.read_csv(df_file_path)
            list_of_columns = list(df.columns)
            self.validation_data[report_key] = list_of_columns
        except Exception as e:
            raise AutoencoderException(e,sys)
    
    def shape_of_df(self,report_key:str):
        try:
            df_file_path = self.dataingestionartifact.merge_df_path
            df = pd.read_csv(df_file_path)
            a = dict()
            a['rows'] = df.shape[0]
            a['columns'] = df.shape[1]
            self.validation_data[report_key] = a
        except Exception as e:
            raise AutoencoderException(e,sys)
        
    def initiate_validation(self)->artifact_entity.DataValidationArtifact:
        try:
            logging.info(f"initiating datavalidation class")
            self.file_type(report_key='file_extension')
            self.duplicate_record(report_key='no_of_duplicate_record')
            self.columns(report_key='name_of_columns')
            self.shape_of_df(report_key='shape_of_df')

            utils.write_yaml_file(file_path=self.datavalidationconfig.validation_report_file_path,
                                  data=self.validation_data)
            data_validation_artifact = artifact_entity.DataValidationArtifact(validation_report_file_path=self.datavalidationconfig.validation_report_file_path)
            return data_validation_artifact
        except Exception as e:
            raise AutoencoderException(e,sys)
    
        


    
