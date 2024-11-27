import pandas as pd
import re
import stemmer

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
    
    final_input = ""

    # Loop through words
    for idx, word in enumerate(words):
        with open("C:/Users/anany/Desktop/fyp/src/sub.txt", 'r', encoding='utf-8') as file:
            is_subject = False
            # Check if the word is a subject
            for line in file:
                columns = line.strip().split('\t')
                if columns[0] == word:
                    is_subject = True
                    gend = columns[1]
                    count = columns[2]
                    final_input+=word
                    final_input+=" "
                    break  # Found the subject, no need to continue checking
        # If the word is a subject, or it's the last word in the list
        if is_subject or idx == len(words) - 1:
            final_input += word
            if idx < len(words) - 1:  # If it's not the last word, add a space
                final_input += ""


    with open("C:/Users/anany/Desktop/stem3.txt","w",encoding='utf-8') as file:
        file.write(final_input)

    
    # Split the sentence into words
    words = final_input.split()
    
    
    
    # Get the last word and strip the last two letters
    last_word = words[-1]
    stripped_word = last_word[-5:]
    
    # Read the verb_suffix.txt and parse it
    with open("C:/Users/anany/Desktop/fyp/src/verb_suffix.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    verb_dict = {}
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 3:
            verb_dict[parts[0]] = (parts[1], parts[2])
    
    # Check if the stripped word is in the verb dictionary
    if stripped_word in verb_dict:
        verb_suffix_2nd_col, verb_suffix_3rd_col = verb_dict[stripped_word]

        if gend == verb_suffix_2nd_col and count == verb_suffix_3rd_col:
            return "Sentence satisfies subject-verb agreement"
        
        return "Sentence does not satisfy subject-verb agreement"

        

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


    df1=pd.read_csv("C:/Users/anany/Desktop/stem3.txt",header=None)
    labels = test_classifier_on_new_data(df1, model)

    if labels[0] == 0:
        return "Sentence satisfies subject-verb agreement"
    return "Sentence does not satisfy subject-verb agreement"