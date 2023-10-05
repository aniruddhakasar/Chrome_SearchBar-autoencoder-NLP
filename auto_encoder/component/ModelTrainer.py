from auto_encoder.entity import config_entity,artifact_entity
from auto_encoder.logger import logging
from auto_encoder.exception import AutoencoderException
from auto_encoder import utils
import os,sys,re,pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint


class ModelTrainer:
    def __init__(self,transformerartifact:artifact_entity.DataTransformationArtifact,
                 modeltrainerconfig:config_entity.ModelTrainerConfig):
        try:
            self.transformerartifact = transformerartifact
            self.modeltrainerconfig = modeltrainerconfig
        except Exception as e:
            raise AutoencoderException(e,sys)
        

    def get_model(self,vocab_size,n_input):
        try:
            model = Sequential()
            model.add(Embedding(vocab_size,10,input_length=3))
            model.add(LSTM(1000,return_sequences=True)) # to use next LSTM
            model.add(LSTM(1000))
            model.add(Dense(850,activation='relu'))
            model.add(Dense(vocab_size,activation='softmax'))
            model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
            logging.info(f"Model has been declared successfully !")
            return model
        except Exception as e:
            raise AutoencoderException(e,sys)
    

    def InitiateModelTrainer(self)->artifact_entity.ModelTrainerArtifact:
        try:
            x_train_data_path = self.transformerartifact.x_train_data_path
            y_train_data_path = self.transformerartifact.y_train_data_path
            model_file_path = self.modeltrainerconfig.model_file_path
            vocab_size = self.transformerartifact.vocab_size
            

            x_train = utils.load_obj(file_path=x_train_data_path)
            y_train = utils.load_obj(file_path=y_train_data_path)
            logging.info(f"successfully loaded data for the training !")


            model = self.get_model(vocab_size=vocab_size,n_input=3)
            checkpoint = ModelCheckpoint(model_file_path,monitor='loss',verbose=1,save_best_only=True)
            model_history = model.fit(x_train,y_train,epochs=1,batch_size=50,callbacks=[checkpoint])

            logging.info(f"model trained successfully !")
            history = model_history.history
            epoch_history_file_path = self.modeltrainerconfig.epoch_history_file_path
            utils.save_object(file_path=epoch_history_file_path,obj=history)


            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(
                model_file_path=model_file_path,
                epoch_history_file_path=epoch_history_file_path,
                x_train_file_path=x_train_data_path,
                y_train_file_path=y_train_data_path
            )
            return model_trainer_artifact
        except Exception as e:
            raise AutoencoderException(e,sys)

