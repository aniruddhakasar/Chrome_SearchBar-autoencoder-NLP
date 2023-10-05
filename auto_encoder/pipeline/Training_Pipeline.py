from auto_encoder.logger import logging
from auto_encoder.exception import AutoencoderException
from auto_encoder.entity import config_entity,artifact_entity
from auto_encoder.component.DataIngestion import DataIngestion
from auto_encoder.component.DataCleaning import DataCleaning
from auto_encoder.component.DataValidation import DataValidation
from auto_encoder.component.DataTransformation import DataTransformation
from auto_encoder.component.ModelTrainer import ModelTrainer
from auto_encoder.component.ModelEvaluatin import ModelEvaluation
from auto_encoder.component.ModelPusher import ModelPusher
import os,sys
from auto_encoder import utils

def RunTrainingPipeline():
    try:
        TrainingPipelineObj = config_entity.TrainingPipelineConfig()
        DataingestionConfigObj = config_entity.DataingestionConfig(
            Training_pipeline_config=TrainingPipelineObj
        )
        DataIngestionObj = DataIngestion(dataingestionconfig=DataingestionConfigObj)
        DataIngestionArtifact = DataIngestionObj.initiateingestion()


        DataValidationConfig = config_entity.DataValidationConfig(Training_pipeline_config=TrainingPipelineObj)
        datavalidationObj = DataValidation(dataingestionartifact=DataIngestionArtifact,
                       datavalidationconfig=DataValidationConfig)
        dataValidationArtifact = datavalidationObj.initiate_validation()
        

        DataCleaningConfig = config_entity.DataCleaningConfig(Training_pipeline_config=TrainingPipelineObj)
        DataCleaningObj = DataCleaning(datacleaningconfig=DataCleaningConfig,
                                       dataingestionartifact=DataIngestionArtifact,
                                       datavalidationartifact=dataValidationArtifact)
        datacleaningArtifact = DataCleaningObj.initiate_datacleaning()


        TransformatonConfigObj = config_entity.DataTransformationConfig(Training_pipeline_config=TrainingPipelineObj)
        DataTransformationObj = DataTransformation(datacleaningartifact=datacleaningArtifact,
                                                   datatransformationconfig=TransformatonConfigObj)
        DataTransformationArtifact =  DataTransformationObj.InitiateTransformation()


        ModelTrainerconfigObj = config_entity.ModelTrainerConfig(Training_pipeline_config=TrainingPipelineObj)
        ModelTrainerObj = ModelTrainer(transformerartifact=DataTransformationArtifact,
                                       modeltrainerconfig=ModelTrainerconfigObj) 
        model_trainer_artifact = ModelTrainerObj.InitiateModelTrainer()


        ModelEvaluation_obj = ModelEvaluation(modeltrainerartifact=model_trainer_artifact,
                                          modeltrainerconfig=ModelTrainerconfigObj)
        ModelEvaluation_obj.InitiateModelEvaluation()


        modelpusherConfig = config_entity.ModelPusherConfig(Training_pipeline_config=TrainingPipelineObj)
        obj_of_model_pusher = ModelPusher(modelpusherconfig=modelpusherConfig,
                                                transformationArtifact=DataTransformationArtifact,
                                                modeltrainerArtifact=model_trainer_artifact)
        obj_of_modelPusherArtifact = obj_of_model_pusher.initiate_Modelpusher()
        
        return obj_of_modelPusherArtifact                  
    except Exception as e:
        raise AutoencoderException(e,sys)