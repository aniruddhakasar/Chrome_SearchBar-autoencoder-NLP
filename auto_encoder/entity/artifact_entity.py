from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    merge_df_path:str

@dataclass
class DataValidationArtifact:
    validation_report_file_path:str

@dataclass
class DataCleaningArtifact:
    cleaned_file_path:str



@dataclass
class DataTransformationArtifact:
    transformed_data_file_path:str
    lemmetized_data_file_path:str
    vocab_size:int
    x_train_data_path:str
    y_train_data_path:str
    word_index_path:str
    tokenizer_model_path:str
    lemmetizer_model_path:str


@dataclass
class ModelTrainerArtifact:
    model_file_path:str
    epoch_history_file_path:str
    x_train_file_path:str
    y_train_file_path:str

@dataclass
class ModelEvaluationArtifact:
    pass

@dataclass
class ModelpusherArtifact:
    pusher_model_dir:str
    saved_model_dir:str
    lemmetizer_path:str
    tokenizer_path:str
    wordindex_path:str
    vocab_size:str
    trained_model_path:str