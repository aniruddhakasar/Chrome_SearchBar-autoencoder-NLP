from auto_encoder.pipeline.Training_Pipeline import RunTrainingPipeline
from auto_encoder.exception import AutoencoderException
import os,sys

if __name__=="__main__":
    try:
        Pusher_Artifact = RunTrainingPipeline()
    except Exception as e:
        raise AutoencoderException(e,sys)