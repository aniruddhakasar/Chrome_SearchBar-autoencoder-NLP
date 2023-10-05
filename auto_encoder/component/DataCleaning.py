from auto_encoder.logger import logging
from auto_encoder.exception import AutoencoderException
from auto_encoder.entity import config_entity,artifact_entity
from auto_encoder import utils
import os,sys
import pandas as pd
import numpy  as np
import re

class DataCleaning:
    def __init__(self,datacleaningconfig:config_entity.DataCleaningConfig,
                 dataingestionartifact:artifact_entity.DataIngestionArtifact,
                 datavalidationartifact:artifact_entity.DataValidationArtifact):
        try:
            self.datacleaningconfig = datacleaningconfig
            self.dataingestionartifact = dataingestionartifact
            self.datavalidationartifact = datavalidationartifact
        except Exception as e:
            raise AutoencoderException(e,sys)
        
    def drop_duplicate(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            logging.info(f"dropping the duplicate history from history")
            unique_titles = df.drop_duplicates()
            unique_titles.reset_index(inplace=True)
            unique_titles = unique_titles.drop('index',axis=1)
            unique_titles['title'] = unique_titles['title'].astype('str')
            logging.info(f"successfully droped duplicacy, and now size :- {unique_titles.shape}")
            return unique_titles
        except Exception as e:
            raise AutoencoderException(e,sys)
        

    def remove_less_legth_history_and_linc(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            new_titles = []
            logging.info(f"remove less lenght and linc history !")
            for i in range(len(df)): 
                if  (2 < len(df['title'][i].split())) and (0 == df['title'][i].count('/')):
                    new_titles.append(df['title'][i])
            df1 = pd.DataFrame({'title':new_titles})
            logging.info(f"successfully removed less length and linc, and now size :- {df1.shape}")
            return df1
        except Exception as e:
            raise AutoencoderException(e,sys)


    def text_cleaning(self,df:pd.DataFrame)->str:
        try:
            logging.info(f"cleaning unnecessay words and lowerise the sentences")
            for i in range(len(df)):
                lowered = df['title'][i].lower()
                cleaned_content = re.sub("[^a-zA-Z]"," ",lowered)  
                df['title'][i] = cleaned_content
            logging.info(f"successfully cleaned the text!")


            logging.info(f"concatenating entire titles as a single str!")
            text1 = ""
            for i in range(len(df)):
                text1 += f"{df['title'][i]}"
            logging.info(f"successfully concated entire text!")


            logging.info(f"removing unneccessary extra spaces.")
            text2 = []
            for i in [''.join(i) for i in text1.split(' ')]:
                if i == '':
                    pass
                else:
                    text2.append(i)

            logging.info(f"preparing final text data!")
            final_text = ""
            for i in text2:
                final_text += f" {i}"
            logging.info(f'successfully prepared the final text with length :- {len(final_text)}')

            return final_text
        except Exception as e:
            raise AutoencoderException(e,sys)
        

    def initiate_datacleaning(self)->artifact_entity.DataCleaningArtifact:
        try:
            logging.info(f"initiating the datacleaning instance methods!")
            df_file_path = self.dataingestionartifact.merge_df_path
            dataframe = pd.read_csv(df_file_path)

            dataframe = self.drop_duplicate(df=dataframe)
            dataframe = self.remove_less_legth_history_and_linc(df=dataframe)
            final_text = self.text_cleaning(df=dataframe)

            cleaned_file_path = self.datacleaningconfig.cleaned_file_path
            utils.save_text_file(file_path=cleaned_file_path,text=final_text)

            datacleaningArtifact = artifact_entity.DataCleaningArtifact(cleaned_file_path=cleaned_file_path)
            return datacleaningArtifact
        except Exception as e:
            raise AutoencoderException(e,sys)
