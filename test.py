import os,sys
from auto_encoder.exception import AutoencoderException
from auto_encoder.pipeline.suggestion import word_suggetions,auto_suggestion
from auto_encoder.pipeline.Training_Pipeline import RunTrainingPipeline
from auto_encoder.exception import AutoencoderException
from auto_encoder.logger import logging
from auto_encoder import utils
import tensorflow as tf
import os,sys



if __name__=="__main__":
    try:
        logging.info(f"suggestion started !!")
        Pusher_Artifact = RunTrainingPipeline()
        permission = input('Do you want to test the suggestion ? press Y/N  >>  ')
        if permission == 'Y':
            tokenizer_path = Pusher_Artifact.tokenizer_path
            model_path = Pusher_Artifact.trained_model_path
            lemmetizer_path = Pusher_Artifact.lemmetizer_path

            logging.info(f"loading all the models to get suggestions ..")
            tokenizer = utils.load_obj(file_path=tokenizer_path)
            lemmetizer = utils.load_obj(file_path=lemmetizer_path)
            model = tf.keras.models.load_model(model_path)
            logging.info(f"successfully loaded all the model to get the suggestion !!")

            logging.info(f"calling the suggestion functions!!")
            auto_suggestion(word_suggetions=word_suggetions,model=model,tokenizer=tokenizer,lemmetizer=lemmetizer)

        else:
            print(f'Thanks for confirmation !')

    except Exception as e:
        raise AutoencoderException(e,sys)





