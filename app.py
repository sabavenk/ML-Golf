import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn 

st.title('ML Golf Betting')

tournament = st.radio("Select the tournament", ["PGA 3-Ball", "EUR 2-Ball"])
round = st.radio("Select the round", ["1", "2", "3", "4"])

load_model = st.text('Loading Model...')
load_model = load_model.text('Done!')

'''
@st.cache
def load_data(tourney):
    pass

@st.cache
def load_model(tourney, rnd):
    if tourney == "PGA 3-Ball" and rnd == "EUR 2-Ball":
        
        

scaler = StandardScaler()
X_train_norm = scaler.fit_transform(X_train)   
X_cv_norm = scaler.transform(X_cv)
                     
@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)
'''
