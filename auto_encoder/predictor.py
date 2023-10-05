import os,sys
from typing import Optional
from auto_encoder.exception import AutoencoderException

MODEL_NAME = os.getenv('MODEL_NAME')
TOKENIZER_NAME = os.getenv('TOKENIZER_NAME')
LEMMETIZER_NAME = os.getenv('LEMMETIZER_NAME')
WORD_INDEX_FIL = os.getenv('WORD_INDEX_FIL')

class ModelResolver:
    # all are default instance variable name
    def __init__(self, model_registry:str = "saved_models",
                lemmetizer:str = 'lemmetizer',
                tokenizer:str = 'tokenizer',
                word_index:str = 'wordIndex',
                model_dir_name = "model"  
                ):  

        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok=True)
        self.lemmetizer = lemmetizer
        self.tokenizer = tokenizer
        self.word_index = word_index
        self.model_dir_name = model_dir_name

# 1
    def get_latest_dir_path(self)->Optional[str]:
        try:
            dir_name = os.listdir(self.model_registry)
            if len(dir_name) == 0:
                return None
            
            dir_name = list(map(int, dir_name))
            latest_dir_name = max(dir_name)
            return os.path.join(self.model_registry, f"{latest_dir_name}")

        except Exception as e:
            raise AutoencoderException(e,sys)
# 2

    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is Not avaliable")

            return os.path.join(latest_dir, self.model_dir_name, MODEL_NAME)
        except Exception as e:
            raise AutoencoderException(e,sys)
# 3

    def get_latest_tokenizer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transform data is not avaliable")

            return os.path.join(latest_dir, self.tokenizer,TOKENIZER_NAME)
        except Exception as e:
            raise AutoencoderException(e,sys)

# 4
    def get_latest_lemmetizer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Traget encoder data is not avaliable")

            return os.path.join(latest_dir, self.lemmetizer, LEMMETIZER_NAME)
        except Exception as e:
            raise AutoencoderException(e,sys)
        

# 4
    def get_latest_wordindex_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Traget encoder data is not avaliable")

            return os.path.join(latest_dir, self.word_index,WORD_INDEX_FIL )
        except Exception as e:
            raise AutoencoderException(e,sys)
        

    def get_latest_save_dir_path(self)->str:

        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir ==  None:
                return os.path.join(self.model_registry, f"{0}")
            
            latest_dir_num = int(os.path.basename(self.get_latest_dir_path())) #basename --> last location of this path
            latest_dir  = os.path.join(self.model_registry, f"{latest_dir_num+1}") ## add 1 so that it will increase everytime
            return latest_dir 
        except Exception as e:
            raise AutoencoderException(e,sys)
# 6

    def get_latest_save_model_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, MODEL_NAME)
        except Exception as e:
            raise AutoencoderException(e,sys)
# 7
    def get_latest_save_tokenizer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.tokenizer, TOKENIZER_NAME) 
        except Exception as e:
            raise AutoencoderException(e,sys)

# 8
    def get_latest_save_lemmetizer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.lemmetizer, LEMMETIZER_NAME) 
        except Exception as e:
            raise AutoencoderException(e,sys)
        
# 9

    def get_latest_save_wordindex_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.word_index, WORD_INDEX_FIL) 
        except Exception as e:
            raise AutoencoderException(e,sys)