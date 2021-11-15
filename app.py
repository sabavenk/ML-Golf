import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import logging 

html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">ML Golf Prediction App</h1> 
    </div> 
    """
st.markdown(html_temp, unsafe_allow_html = True) 

VALID_PGA_ROUNDS = {1, 2}
VALID_EUR_ROUNDS = {3, 4}
EUR_MODEL_LOC = 'Streamlit_2ball_all_data.sav'
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

tournament = st.radio("A) Select the scenario", ["PGA 3-Ball", "EUR 2-Ball"])
round = st.radio("B) Select the tournament round", ["1", "2", "3", "4"])

scrs_url = 'https://www.livesport.com/en/golf/rankings/owgr/'
odds_url = 'https://www.betfair.com/exchange/plus/en/golf-betting-3'

st.write('C) Select currently available head-to-head matchup with live odds @ [Betfair](%s)' % odds_url)
st.text('D) Now enter the following data points...')

all_data, pl_names = [],[]
player_1_data , player_2_data , player_3_data = [], [], []
st.text('1. Inuput data for Player with lowest odds:')
st.write('... historical player scores available @ [Livesport](%s)' % scrs_url)
f_name1 = st.text_input('Name [for Player with lowest odds]:')
f_01 = st.number_input('Odds [for Player with lowest odds]:')
f_02 = st.number_input('Round score from 3 tourns ago [for Player with lowest odds]')
f_03 = st.number_input('Round score from 2 tourns ago [for Player with lowest odds]')
f_04 = st.number_input('Round score from 1 tourn ago [for Player with lowest odds]')
player_1_data = [f_01, f_02, f_03, f_04]
all_data += player_1_data

if tournament == "PGA 3-Ball":
    st.text('2. Inuput data for Player with 2nd lowest odds:')
    st.write('... historical player scores available @ [Livesport](%s)' % scrs_url)
    f_name2 = st.text_input('Name [for Player with 2nd lowest odds]:')
    f_11 = st.number_input('Odds [for Player with 2nd lowest odds]:')
    f_12 = st.number_input('Round score from 3 tourns ago [for Player with 2nd lowest odds]')
    f_13 = st.number_input('Round score from 2 tourns ago [for Player with 2nd lowest odds]')
    f_14 = st.number_input('Round score from 1 tourn ago [for Player with 2nd lowest odds]')
    player_2_data = [f_11, f_12, f_13, f_14]
    all_data += player_2_data
    st.text('3. Inuput data for Player with highest odds:')
    st.write('... historical player scores available @ [Livesport](%s)' % scrs_url)
    f_name3 = st.text_input('Name [for Player with highest odds]:')
    f_21 = st.number_input('Odds [for Player with highest odds]:')
    f_22 = st.number_input('Round score from 3 tourns ago [for Player with highest odds]')
    f_23 = st.number_input('Round score from 2 tourns ago [for Player with highest odds]')
    f_24 = st.number_input('Round score from 1 tourn ago [for Player with highest odds]')
    player_3_data = [f_21, f_22, f_23, f_24]
    all_data += player_3_data
    pl_names = [f_name1, f_name2, f_name3, 
                f_name1+' & '+f_name2, f_name1+' & '+f_name3, f_name2+' & '+f_name3,
                f_name1+' & '+f_name2+' & '+f_name3]
else:
    st.text('2. Inuput data for Player with highest odds:')
    st.write('... historical player scores available @ [Livesport](%s)' % scrs_url)
    f_name2 = st.text_input('Name [for Player with highest odds]:')
    f_11 = st.number_input('Odds [for Player with highest odds]:')
    f_12 = st.number_input('Round score from 3 tourns ago [for Player with highest odds]')
    f_13 = st.number_input('Round score from 2 tourns ago [for Player with highest odds]')
    f_14 = st.number_input('Round score from 1 tourn ago [for Player with highest odds]')
    player_2_data = [f_11, f_12, f_13, f_14]
    all_data += player_2_data
    pl_names = [f_name1, f_name2, f_name1+' & '+f_name2]
    
# convert text input into format needed for model

st.dataframe(pd.DataFrame(all_data))

def normalize_data(df, tournment):
    if tournament == "PGA 3-Ball":
      mu, std = mu_PGA, sigma_PGA
    else:
      mu, std = mu_EUR, sigma_EUR
    new_df = (np.array(df) - np.array(mu))/np.array(std)
    return new_df

  
if st.button("E) Predict"):
    normalized_input_data = normalize_data(all_data, tournament)
    st.text('Here is the normalized dataframe of your inputs:')
    st.dataframe(normalized_input_data)
    
    load_model_state = st.text('Loading Model...')
    model = pickle.load(open(PGA_MODEL_LOC, 'rb')) if tournament == "PGA 3-Ball" else pickle.load(open(EUR_MODEL_LOC, 'rb'))   
    load_model_state = load_model_state.text('Done loading model!')
    output = model.predict(normalized_input_data.reshape(1, -1))
    st.success('Finished! See below for predicted winner...')
    st.write(output)
    st.write(pl_names[int(output)])
      
