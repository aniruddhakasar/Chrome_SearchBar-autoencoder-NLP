import numpy as  np
import os,sys
from auto_encoder.exception import AutoencoderException
from auto_encoder.logger import logging

################ lemmetizer

def word_suggetions(model,tokenizer,lemmetizer ,text):
    try:
        lemmetized_text = [lemmetizer.lemmatize(word) for word in text]
        sequence = tokenizer.texts_to_sequences([lemmetized_text])
        sequence = np.array(sequence)
        y_pred = np.argmax(model.predict(sequence))

        predicted_word = ""

        for key,value in tokenizer.word_index.items():
            if value == y_pred:
                predicted_word = key
                break
        return predicted_word
    except Exception as e:
        raise AutoencoderException(e,sys)
    
def auto_suggestion(word_suggetions,model,tokenizer,lemmetizer):
    while True:
        text = input('Search Here .. ')

        if text == 'stop':
            print(f"Thanks for the testing !")
            break
        elif len(text.split(' '))<3:
            print(f"please inseart minumum 3 words, your text length is low !!")
            text = input('search again here ...')
        else:
            words = text.split(" ")
            words = words[-3:]
            print(f"i am suggesting to you this word on the bases of these words :- {words}")
            predicted_word = word_suggetions(model=model,tokenizer=tokenizer,lemmetizer=lemmetizer,text=words)
            search_paragraph = text+" "+predicted_word
        print(predicted_word)
        print(f"your search paragraph :- {search_paragraph}")