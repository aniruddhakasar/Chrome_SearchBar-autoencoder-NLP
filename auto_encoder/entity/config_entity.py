import os,sys
from auto_encoder.exception import AutoencoderException
from auto_encoder.logger import  logging
from datetime import datetime
from auto_encoder import utils
from typing import List

## Loading all necessary enviroment variable
FILE_NAME = os.getenv('FILE_NAME')
CLEANED_FILE_NAME = os.getenv('CLEANED_FILE_NAME')
VALIDATION_REPORT_FILE_NAME = os.getenv('VALIDATION_REPORT_FILE_NAME')
LEMMATIZED_FILE_NAME = os.getenv('LEMMATIZED_FILE_NAME')
TRANSFORM_FILE_NAME = os.getenv('TRANSFORM_FILE_NAME')
TOKENIZER_NAME = os.getenv('TOKENIZER_NAME')
SEQUENCE_DATA_FILE = os.getenv('SEQUENCE_DATA_FILE')
SEQUENCED_NP_ARRAY = os.getenv('SEQUENCED_NP_ARRAY')
WORD_INDEX_FIL = os.getenv('WORD_INDEX_FIL')
X_TRAIN_DATA_FILE_NAME = os.getenv('X_TRAIN_DATA_FILE_NAME')
Y_TRAIN_DATA_FILE_NAME = os.getenv('Y_TRAIN_DATA_FILE_NAME')
MODEL_NAME = os.getenv('MODEL_NAME')
LEMMETIZER_NAME = os.getenv('LEMMETIZER_NAME')



class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),'artifacts',f"{datetime.now().strftime('%m%d%Y__%H%M%S')}")
        except Exception as e:
            raise AutoencoderException(e,sys)
    

class DataingestionConfig:
    def __init__(self,Training_pipeline_config:TrainingPipelineConfig): 
        try:
            logging.info('configuring all the dataIngestion path variable')
            self.history_file_path:List[str] = utils.get_data_path()
            self.data_ingestion_dir:str = os.path.join(Training_pipeline_config.artifact_dir,'DataIngestion')
            self.data_dir_path:str = os.path.join(self.data_ingestion_dir,'Dataset')
            self.data_file_path:str = os.path.join(self.data_dir_path,FILE_NAME)
            logging.info('successfully configured')
        except Exception as e:
            raise AutoencoderException(e,sys)


class DataCleaningConfig:
    def __init__(self,Training_pipeline_config:TrainingPipelineConfig):
        try:
            self.data_cleainig_dir = os.path.join(Training_pipeline_config.artifact_dir,'datacleaning')
            self.data_dir = os.path.join(self.data_cleainig_dir,'cleaned_file')
            self.cleaned_file_path = os.path.join(self.data_dir,CLEANED_FILE_NAME)
        except Exception as e:
            raise AutoencoderException(e,sys)
        
class DataValidationConfig:
    def __init__(self,Training_pipeline_config:TrainingPipelineConfig):
        self.validatin_Dir = os.path.join(Training_pipeline_config.artifact_dir,'datavalidation')
        self.validation_report_file_path = os.path.join(self.validatin_Dir,'VALIDATION_REPORT_FILE_NAME')


class DataTransformationConfig:
    def __init__(self,Training_pipeline_config:TrainingPipelineConfig):
        try:
            self.transformation_dir = os.path.join(Training_pipeline_config.artifact_dir,'DataTransformation')
            self.transformed_data_file_path = os.path.join(self.transformation_dir,TRANSFORM_FILE_NAME)
            self.lemmetized_dir = os.path.join(self.transformation_dir,'lemmetized')
            self.lemmetized_file_path = os.path.join(self.lemmetized_dir,LEMMATIZED_FILE_NAME)
            self.tokenizer_dir = os.path.join(self.transformation_dir,'Tokenizer')
            self.tokenizer_model_path = os.path.join(self.tokenizer_dir,TOKENIZER_NAME)
            self.sequence_data_dir = os.path.join(self.transformation_dir,'SequenceData')
            self.sequence_data_textfile_path = os.path.join(self.sequence_data_dir,SEQUENCE_DATA_FILE)
            self.sequence_data_npFile = os.path.join(self.sequence_data_dir,SEQUENCED_NP_ARRAY)
            self.word_index_dir = os.path.join(self.transformation_dir,'wordindex')
            self.word_index_file_path = os.path.join(self.word_index_dir,WORD_INDEX_FIL)
            self.Train_data_dir  = os.path.join(self.transformation_dir,'Trained_data')
            self.x_train_file_path = os.path.join(self.Train_data_dir,X_TRAIN_DATA_FILE_NAME)
            self.y_train_file_path = os.path.join(self.Train_data_dir,Y_TRAIN_DATA_FILE_NAME)
            self.lemmetizer_obj_path = os.path.join(self.lemmetized_dir,LEMMETIZER_NAME)
        except Exception as e:
            raise AutoencoderException(e,sys)


class ModelTrainerConfig:
        def __init__(self,Training_pipeline_config:TrainingPipelineConfig):
            try:
                self.model_trainer_dir = os.path.join(Training_pipeline_config.artifact_dir,'Model')
                self.model_file_path = os.path.join(self.model_trainer_dir,MODEL_NAME)
                self.epoch_history_dir = os.path.join(self.model_trainer_dir,"epoch_history")
                self.epoch_history_file_path = os.path.join(self.epoch_history_dir,'epoch_history.pkl')                
            except Exception as e:
                raise AutoencoderException(e,sys)


class ModelPusherConfig:
    def __init__(self,Training_pipeline_config:TrainingPipelineConfig):
        self.model_Pusher_dir  = os.path.join(Training_pipeline_config.artifact_dir,'modelPusher')
        try:
            self.saved_model = os.path.join('saved_models')
            self.pusher_model_dir = os.path.join(self.model_Pusher_dir,'saved_models')
            self.pusher_model_path = os.path.join(self.pusher_model_dir,MODEL_NAME)
            self.pusher_Toknizer_path = os.path.join(self.pusher_model_dir,TOKENIZER_NAME)
            self.pusher_lemmetizer_path = os.path.join(self.pusher_model_dir,LEMMETIZER_NAME)
            self.pusher_wordIndex_path = os.path.join(self.pusher_model_dir,WORD_INDEX_FIL)

        except Exception as e:
            raise AutoencoderException(e,sys)
