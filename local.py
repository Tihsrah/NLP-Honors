import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open('datad.txt') as story:
  story_data = story.read()

lower_data = story_data.lower()           # Converting the string to lower case to get uniformity

split_data = lower_data.splitlines()      # Splitting the data to get every line seperately but this will give the list of uncleaned data

# print(split_data) 
split_data.pop(0)

final = ''                                # initiating a argument with blank string to hold the values of final cleaned data

for line in split_data:
  final += '\n' + line

final_data = final.split('\n')       # splitting again to get list of cleaned and splitted data ready to be processed
# print(final_data)
final_data = [x for x in final_data if x != '']


# Instantiating the Tokenizer
max_vocab = 1000000
tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(final_data)

model = load_model("chad_generator_1000.h5")
input_seq = []

for line in final_data:
  token_list = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(token_list)):
    n_gram_seq = token_list[:i+1]
    input_seq.append(n_gram_seq)
max_seq_length = max(len(x) for x in input_seq)

def predict_words(seed, no_words):
  for i in range(no_words):
    token_list = tokenizer.texts_to_sequences([seed])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_length-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=1)

    new_word = ''

    for word, index in tokenizer.word_index.items():
      if predicted == index:
        new_word = word
        break
    seed += " " + new_word
  return seed

# Set up the Streamlit app
st.title('Chad-GPT is not a GPT')
st.write('Enter some text and the model will predict something')

# Create a text input box for the user to input text
text_input = st.text_input('Enter some text:')

# Create a button to submit the input and make a prediction
if st.button('Predict'):
    # Call your ML model to make a prediction

    next_words = 20

    prediction=predict_words(text_input, next_words)

    # Display the prediction to the user
    st.write(f'The model predicts: {prediction}')