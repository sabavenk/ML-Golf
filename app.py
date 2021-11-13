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
X_features_EUR = ['p0_pl_back', 'p0_R-3_scr', 'p0_R-2_scr', 'p0_R-1_scr', 'p1_pl_back', 
              'p1_R-3_scr', 'p1_R-2_scr', 'p1_R-1_scr']
X_features_PGA = X_features_EUR + ['p2_pl_back', 'p2_R-3_scr', 'p2_R-2_scr', 'p2_R-1_scr']
Y_features = ['p0_win_ind', 'p1_win_ind', 'p2_win_ind']

mu_PGA = [2.2820451034908866, -1.163422922459067, -1.1479765214704973, -1.2236638863144886, 2.9192585727525486, -0.6431881371640408, 
          -0.5193080012357121, -0.6558541859746679, 4.223450725980847, -0.002162496138399753, 0.06487488415199258, 0.18906394810009267]
sigma_PGA = [0.295843443460005, 3.1986321834377978, 3.146614032920214, 3.148083810737567, 0.3660466943506574, 3.221840162240196, 
             3.1493580875939386, 3.1987322540515333, 1.5463797053766386, 3.3081370020015886, 3.2568160636600405, 3.379337753568236]
mu_EUR = [1.843833736884584, -1.0653753026634383, -1.1832122679580306, -1.2300242130750605, 
          2.6536319612590797, -0.3357546408393866, -0.5819209039548022, -0.38579499596448746]
sigma_EUR = [0.1936053784751192, 3.0491937304074836, 3.0177589532997438, 3.1301397570277776, 
             0.4731022206648472, 3.1569549991015955, 3.096436941689078, 3.0658089992624813]

st.title('ML Golf Betting')

tournament = st.radio("Select the tournament", ["PGA 3-Ball", "EUR 2-Ball"])
round = st.radio("Select the round", ["1", "2", "3", "4"])

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
st.text("p0_pl_back, p0_R-3_scr, p0_R-2_scr, 'p0_R-1_scr")

player_1_data = st.text_input('Please input the data of Player 1 from the past 3 tournaments as shown above')
player_2_data = st.text_input('Please input the data of Player 2 from the past 3 tournaments as shown above')
player_3_data = st.text_input('Please input the data of Player 3 from the past 3 tournaments as shown above')

# convert text input into format needed for model
@st.cache
def standardize_data(data): 
    output = []
    for row in data:
        output += str.split(',')
    return list(map(int, output))


def normalize_data(data, tournment):
    df = np.array(standardize(data))
    if tournament == "PGA 3-Ball":
      mu, std = np.array(mu_PGA), np.array(sigma_PGA)
    else:
      mu, std = np.array(mu_EUR), np.array(sigma_EUR)
    new_df = (data - mu)/std
    return new_df

normalized_input_data = normalize_data([player_1_data, player_2_data, player_3_data], scaler)
st.text('Here is the normalized dataframe of your inputs:')
st.dataframe(normalized_input_data)
pred_load = st.text('Predicting outcome...')

@st.cache
def post_process_output(df, model):
    predictions = model.predict(df)
    # do more to predictions before returning 
    return predictions

output = post_process_output(normalized_input_data)
pred_load.text('Finished! See below for results')

st.write(output)
