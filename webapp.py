import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from googletrans import Translator
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

# Initialize Translator
translator = Translator()

# Language Selection
languages = {"English": "en", "Hindi": "hi", "Telugu": "te"}
selected_lang = st.sidebar.selectbox("Choose Language", list(languages.keys()))

def translate_text(text):
    try:
        translated = translator.translate(text, dest=languages[selected_lang])
        return translated.text
    except Exception as e:
        return text  # Fallback to original text if translation fails

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')

# Features and labels
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Split the data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=42)
RF = RandomForestClassifier(n_estimators=20, random_state=5)
RF.fit(Xtrain, Ytrain)

# Save and Load Model
RF_pkl_filename = 'RF.pkl'
with open(RF_pkl_filename, 'wb') as file:
    pickle.dump(RF, file)
RF_Model_pkl = pickle.load(open('RF.pkl', 'rb'))

# Prediction Function
def predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    prediction = RF_Model_pkl.predict([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    return prediction[0]

# Streamlit UI
st.markdown(f"<h1 style='text-align: center;'>{translate_text('SMART CROP RECOMMENDATIONS')}</h1>", unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center;'>{translate_text('Predict the best crop to grow based on soil and weather conditions')}</h3>", unsafe_allow_html=True)
st.image("crop.png")

st.sidebar.title("🌱 BioSage")
st.sidebar.header(translate_text("Enter Crop Details"))

# Input fields
nitrogen = st.sidebar.slider(translate_text("Nitrogen"), 0.0, 140.0, 0.0, 0.1)
phosphorus = st.sidebar.slider(translate_text("Phosphorus"), 0.0, 145.0, 0.0, 0.1)
potassium = st.sidebar.slider(translate_text("Potassium"), 0.0, 205.0, 0.0, 0.1)
temperature = st.sidebar.slider(translate_text("Temperature (°C)"), 0.0, 51.0, 0.0, 0.1)
humidity = st.sidebar.slider(translate_text("Humidity (%)"), 0.0, 100.0, 0.0, 0.1)
ph = st.sidebar.slider(translate_text("pH Level"), 0.0, 14.0, 0.0, 0.1)
rainfall = st.sidebar.slider(translate_text("Rainfall (mm)"), 0.0, 500.0, 0.0, 0.1)

# Prediction Button
if st.sidebar.button(translate_text("Predict")):
    if all(val > 0 for val in [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]):
        prediction = predict_crop(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        st.markdown(f"<h2 style='text-align: center; color: #9DC08B;'>{translate_text('The recommended crop is:')} {translate_text(prediction)}</h2>", unsafe_allow_html=True)
    else:
        st.error(translate_text("Please fill in all input fields with valid values before predicting."))
