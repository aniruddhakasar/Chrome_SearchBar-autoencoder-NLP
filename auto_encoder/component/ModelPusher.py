import numpy as np
import pandas as pd
import os,sys
from auto_encoder import utils
from auto_encoder.entity import config_entity,artifact_entity
from auto_encoder.exception import AutoencoderException
from auto_encoder.logger import logging
from auto_encoder.predictor import ModelResolver
import tensorflow as tf

class ModelPusher:
    def __init__(self,modelpusherconfig:config_entity.ModelPusherConfig,
                 transformationArtifact:artifact_entity.DataTransformationArtifact,
                 modeltrainerArtifact:artifact_entity.ModelTrainerArtifact):
        
        try:
            self.modelpusherconfig = modelpusherconfig
            self.transformationArtifact = transformationArtifact
            self.modeltrainerArtifact = modeltrainerArtifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise AutoencoderException(e,sys)


    def initiate_Modelpusher(self,)->artifact_entity.ModelpusherArtifact:
        try:

            tokenizer = utils.load_obj(file_path=self.transformationArtifact.tokenizer_model_path)
            lemmitizer = utils.load_obj(file_path=self.transformationArtifact.lemmetizer_model_path)
            model = tf.keras.models.load_model(self.modeltrainerArtifact.model_file_path)            
            wordIndex = utils.load_obj(file_path=self.transformationArtifact.word_index_path)
            
            # saved in artifact --> modelpusher
            # utils.save_object(file_path=self.modelpusherconfig.pusher_model_path,model_obj=model)
            utils.save_object(file_path=self.modelpusherconfig.pusher_lemmetizer_path,obj=lemmitizer)
            utils.save_object(file_path=self.modelpusherconfig.pusher_Toknizer_path,obj=tokenizer)
            utils.save_object(file_path=self.modelpusherconfig.pusher_wordIndex_path,obj=wordIndex)
            model.save(self.modelpusherconfig.pusher_model_path)
            logging.info(f"saved models in artifact -> Modelpusher")

            # get the path to save in saved_models
            tokenizer_path = self.model_resolver.get_latest_save_tokenizer_path()
            lemmitizer_path = self.model_resolver.get_latest_save_lemmetizer_path()
            model_path = self.model_resolver.get_latest_save_model_path()
            wordindex_path = self.model_resolver.get_latest_save_wordindex_path()

            # to save the models in saved_models
            utils.save_object(file_path=tokenizer_path,obj=tokenizer)
            utils.save_object(file_path=lemmitizer_path,obj=lemmitizer)
            utils.save_object(file_path=wordindex_path,obj=wordIndex)
            model.save(model_path)
            logging.info(f"saved models in saved_models")
            logging.info(f"Training Pipeline executed Successfully !")
            print(f"Training Pipeline executed Successfully !")


            modelPusherArtifact = artifact_entity.ModelpusherArtifact(pusher_model_dir=self.modelpusherconfig.pusher_model_dir,
                                                                      saved_model_dir=self.modelpusherconfig.saved_model,
                                                                      lemmetizer_path=lemmitizer_path,
                                                                      tokenizer_path=tokenizer_path,
                                                                      wordindex_path=wordindex_path,
                                                                      vocab_size=self.transformationArtifact.vocab_size,
                                                                      trained_model_path=model_path)
            return modelPusherArtifact

        except Exception as e:
            raise AutoencoderException(e,sys)