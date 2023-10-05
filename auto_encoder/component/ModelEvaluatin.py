from auto_encoder.entity import config_entity,artifact_entity
from auto_encoder.logger import logging
from auto_encoder.exception import AutoencoderException
from auto_encoder import utils
import os,sys,re,pickle
import pandas as pd
import numpy as np
import tensorflow as tf
class ModelEvaluation:
    def __init__(self,modeltrainerartifact:artifact_entity.ModelTrainerArtifact,
                 modeltrainerconfig:config_entity.ModelTrainerConfig):
        try:
            self.modeltrainerartifact = modeltrainerartifact
            self.modeltrainerconfig = modeltrainerconfig
        except Exception as e:
            raise AutoencoderException(e,sys)
        

    def load_model(self,file_path:str):
        try:
            model = tf.keras.models.load_model(file_path)
            return model
        except Exception as e:
            raise AutoencoderException(e,sys)
        

    def InitiateModelEvaluation(self):
        try:
            model_file_path = self.modeltrainerartifact.model_file_path
            x_train_file_path = self.modeltrainerartifact.x_train_file_path
            y_train_file_path = self.modeltrainerartifact.y_train_file_path

            model = self.load_model(file_path=model_file_path)
            x_train = utils.load_obj(file_path=x_train_file_path)
            y_train = utils.load_obj(file_path=y_train_file_path)

            evaluation_result = model.evaluate(x_train,y_train)

            loss = evaluation_result[0] * 10
            accuracy = evaluation_result[1] * 100
            loss = round(loss,2)
            accuracy = round(accuracy,2)
            logging.info(f"model evaluted with this accuracy :-{accuracy} and loss is :- {loss}")

            difference= accuracy - loss

            if (difference >20) or (difference < 0):
                logging.info(f"Model is underfit !")
                print(f"model is underfit Try to improve the accuracy of your model!")

            elif difference < 4:
                logging.info(f"model is overfitted ! Try to regularize your model !")
                print(f"model is overfitted !")

            elif (difference > 5) and (difference < 15):
                logging.info(f"Hurray , model is generelized !")
                print(f'model is generailized !')
            print(f"model successfully evaluated !")

        except Exception as e:
            raise AutoencoderException(e,sys)
