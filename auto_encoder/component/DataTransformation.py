from auto_encoder.logger import logging
from auto_encoder.exception import AutoencoderException
from auto_encoder.entity import config_entity,artifact_entity
from auto_encoder import utils
import os,sys,re
import pandas as pd
import numpy as np
import nltk
import re                                                       # to regular expression
from nltk.corpus import stopwords                               # to stopwords
from nltk.stem.porter  import PorterStemmer                     # to stemming
from nltk.stem import WordNetLemmatizer                         # to lemmatizer
from sklearn.feature_extraction.text import CountVectorizer     # to BOW
from sklearn.feature_extraction.text import TfidfVectorizer     # to  TF-IDF
from gensim.models import Word2Vec                              # to word2vec
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
import pickle
import warnings
warnings.filterwarnings('ignore')

class DataTransformation:
    def __init__(self,datacleaningartifact:artifact_entity.DataCleaningArtifact,
                 datatransformationconfig:config_entity.DataTransformationConfig):
        try:
            self.datacleaningartifact = datacleaningartifact
            self.datatransformationconfig = datatransformationconfig
        except Exception as e:
            raise AutoencoderException(e,sys)
        
    def load_text_file(self):
        try:
            file_path = self.datacleaningartifact.cleaned_file_path
            file = open(file_path,'r')
            data = file.read()
            logging.info(f"read text file from {file_path}")
            return data
        except Exception as e:
            raise AutoencoderException(e,sys)

    def lemmetized(self,text_data:str):
        try:
            lemmetizer = WordNetLemmatizer()
            words = nltk.word_tokenize(text_data)            
            lemmetized = [lemmetizer.lemmatize(word) for word in words]
            lemmetized_data = ' '.join(lemmetized)

            logging.info(f"text_data successfully lemmetized !")
            lemmetized_file_path = self.datatransformationconfig.lemmetized_file_path
            lemmetizer_obj_path = self.datatransformationconfig.lemmetizer_obj_path
            logging.info(f"saving the lemmetized file at this position :- {lemmetized_file_path}")

            utils.save_text_file(file_path=lemmetized_file_path,text=lemmetized_data)
            utils.save_object(file_path=lemmetizer_obj_path,obj=lemmetizer)

            
            return lemmetized_file_path
        
        except Exception as e:
            raise AutoencoderException(e,sys)
        

    def Tokenizer(self):
        try:
            tokenizer = Tokenizer()
            lemmetized_file_path = self.datatransformationconfig.lemmetized_file_path
            data = open(lemmetized_file_path,'r')
            Data  = data.read()
            tokenizer.fit_on_texts([Data])
            vocab_size = len(tokenizer.word_index)+1
            word_index = tokenizer.word_index
            word_index_file_path = self.datatransformationconfig.word_index_file_path
            utils.save_object(file_path=word_index_file_path,
                              obj=word_index)
            logging.info(f"word index successfully save at this position :- {word_index_file_path}")
            logging.info(f"no. of unique words :- {vocab_size}")


            tokenizer_path = self.datatransformationconfig.tokenizer_model_path
            directory = os.path.dirname(tokenizer_path)
            os.makedirs(directory,exist_ok=True)
            pickle.dump(tokenizer,open(tokenizer_path,'wb'))
            logging.info(f"tokenizer successfully fit and saved on this position :- {tokenizer_path}")

            sequence_data = tokenizer.texts_to_sequences([Data])[0]

            return vocab_size,sequence_data
        except Exception as e:
            raise AutoencoderException(e,sys)
        
    def transform_sequences(self,sequence_data):
        try:
            sequences = []
            for i in range(3,len(sequence_data)):
                words = sequence_data[i-3:i+1]
                sequences.append(words)
            sequences = np.array(sequences)

            sequence_data_textfile_path = self.datatransformationconfig.sequence_data_textfile_path
            utils.save_text_file(file_path=sequence_data_textfile_path,text=str(sequence_data[0:]))
            logging.info(f"sequence data textfile successfully saved at this position :- {sequence_data_textfile_path}")

            numpy_sequence_file_path = self.datatransformationconfig.sequence_data_npFile
            utils.save_numpy_array_data(file_path=numpy_sequence_file_path,array=sequences)
            logging.info(f'successfully saved sequence numpy array !')
            return sequences
        except Exception as e:
            raise AutoencoderException(e,sys)
        
    def set_independent_and_depen(self,sequences):
        try:
            x = []
            y = []
            for i in sequences:
                x.append(i[0:3])
                y.append(i[3])
            x = np.array(x)
            y = np.array(y)
            return x,y
        except Exception as e:
            raise AutoencoderException(e,sys)
        
    def transform_dependent_var(self,y_data,vocab_size):
        try:
            y = to_categorical(y_data,num_classes=vocab_size)
            logging.info(f"successfully transformed the y_data !")
            return y
        except Exception as e:
            raise AutoencoderException(e,sys)




    def InitiateTransformation(self)->artifact_entity.DataTransformationArtifact:
        try:
            text_data = self.load_text_file()
            lemmetized_file_path = self.lemmetized(text_data=text_data)
            vocab_size,sequence_data = self.Tokenizer()
            sequences = self.transform_sequences(sequence_data=sequence_data)
            x_data,y_data = self.set_independent_and_depen(sequences=sequences)
            y_dat = self.transform_dependent_var(y_data=y_data,vocab_size=vocab_size)

            x_train_file_path = self.datatransformationconfig.x_train_file_path
            y_train_file_path = self.datatransformationconfig.y_train_file_path
            word_index_file_path = self.datatransformationconfig.word_index_file_path

            utils.save_object(file_path=x_train_file_path,obj=x_data)
            utils.save_object(file_path=y_train_file_path,obj=y_dat)

            train_data_dir_path = self.datatransformationconfig.Train_data_dir
            logging.info(f"succesfully saved your trained data at this position :- {train_data_dir_path}")


            

            transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformed_data_file_path=self.datatransformationconfig.transformed_data_file_path,
                lemmetized_data_file_path=lemmetized_file_path,
                vocab_size=vocab_size,
                x_train_data_path=x_train_file_path,
                y_train_data_path=y_train_file_path ,
                word_index_path = word_index_file_path,
                tokenizer_model_path=self.datatransformationconfig.tokenizer_model_path,
                lemmetizer_model_path=self.datatransformationconfig.lemmetizer_obj_path
                )
            return transformation_artifact
        except Exception as e:
            raise AutoencoderException(e,sys)
        




