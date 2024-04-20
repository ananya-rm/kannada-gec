import pandas as pd
import re

model = pd.read_pickle('model.pkl')

from bert.bert_model import BERT_model
import logging
from util import logger

def detect(text):

    kannada_pattern = re.compile(r'[\u0C80-\u0CFF]+')

    # Check if the text contains Kannada characters
    b = bool(re.search(kannada_pattern, text))

    if(b != True):
        return "Please enter text in Kannada"

    words = text.split()

    if(len(words) < 2):
        return "Enter sentence with subject and verb"

    with open("C:/Users/anany/Desktop/stem.txt","w",encoding='utf-8') as file:
        file.write(text)

    def test_classifier_on_new_data(new_df, classifier):
        # Load the tokenizer and BERT model
        bert = BERT_model()
        bert.load_BERT(small=True)

        # Tokenize the sentences in the new dataframe
        tokenized_df = new_df[0].apply(lambda sent: bert.tokenize_sentence(sent))
        MAX_LEN = 128
        # Convert tokenized sentences to BERT embeddings
        bert_hidden_states = bert.convert_tokenized_sent_to_bert_emb(tokenized_df, MAX_LEN)
        bert_feature_array = bert_hidden_states[:, 0, :].numpy()

        # Predict labels using the trained classifier
        predicted_labels = classifier.predict(bert_feature_array)

        return predicted_labels


    df1=pd.read_csv("C:/Users/anany/Desktop/stem.txt",header=None)
    labels = test_classifier_on_new_data(df1, model)

    if labels[0] == 0:
        return "Sentence satisfies subject-verb agreement"
    return "Sentence does not satisfy subject-verb agreement"
