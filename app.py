import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler


VALID_PGA_ROUNDS = {1, 2}
VALID_EUR_ROUNDS = {3, 4}
EUR_MODEL_LOC = 'svm_ror_EUR_R[3, 4]_2b_all_data.sav'
PGA_MODEL_LOC = 'svm_ror_PGA_R[1, 2]_3b_all_data.sav'

st.title('ML Golf Betting')

tournament = st.radio("Select the tournament", ["PGA 3-Ball", "EUR 2-Ball"])
round = st.radio("Select the round", ["1", "2", "3", "4"])

load_data_state = st.text('Loading Hisotrical Data...')

@st.cache
def load_data(tourney):
    if tourney == "PGA 3-Ball":
        return pd.read_csv('Streamlit_3ball_classification.csv')
    else:
        return pd.read_csv('Streamlit_2ball_classification.csv')

data = load_data(tournament)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)
load_data_state = load_data_state.text('Done loading historical data!')
load_model_state = st.text('Loading Model...')

@st.cache
def load_model(tourney, rnd):
    if tourney == "PGA 3-Ball" and rnd in VALID_PGA_ROUNDS:
        return pickle.load(open(PGA_MODEL_LOC, 'rb'))
    elif tourney == "EUR 2-Ball" and rnd in VALID_EUR_ROUNDS:
        return pickle.load(open(EUR_MODEL_LOC, 'rb'))

model = load_model(tournament, round)    
load_model_state = load_model_state.text('Done loading model!')

st.text('Now, enter the data in the following format:')
st.text('Tiger Woods, 0.8, -3,  1.2, +1,  1.0, -1')

player_1_data = st.text_input('Please input the data of Player 1 from the past 3 tournaments as shown above')
player_2_data = st.text_input('Please input the data of Player 2 from the past 3 tournaments as shown above')
player_3_data = st.text_input('Please input the data of Player 3 from the past 3 tournaments as shown above')

def standardize_data(data):
    # convert text input into format needed for model
    pass

@st.cache
def normalize_data(data, transformer):
    # df = standardize(data)
    # new_data = transformer.transform(data)
    pass

# normalized_input_data = normalize_data([player_1_data, player_2_data, player_3_data], scaler)
# st.text('Here's the normalized dataframe of your inputs:')
# st.dataframe(normalized_input_data)
# pred_load = st.text('Predicting outcome...')

@st.cache
def post_process_output(df, model):
    predictions = model.predict(df)
    # do more to predictions before returning 
    return predictions

# output = post_process_output(normalized_input_data)
pred_load.text('Finished! See below for results')

st.write(output)
