import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import RobustScaler, StandardScaler
from PIL import Image
import joblib



model = load_model('models/lstm.h5')
sidebar_image = Image.open('images/Customer-Churn.png')
main_image = Image.open('images/Understanding Customer Churn.png')

st.sidebar.info('Notre application de prédiction du churn')
st.sidebar.image(sidebar_image)

st.image(main_image)
st.title("Prédiction de Churn Client")

def preprocess_data(df):
    st.write("Données après prétraitement et avant redimensionnement :")
    st.write(df)
    df["NewAGT"] = df["age"] - df["tenure"]  # on cree une nouvelle colonne qui est la difference entre l'age et le tenure pour voir si il y a une correlation entre les deux
    df["AgeScore"] = df['age'] # on divise la colonne age en 8 parties pour avoir une meilleure visualisation
    df["BalanceScore"] = df['balance']  # on divise la colonne balance en 10 parties car elle contient beaucoup de valeurs differentes pour avoir une meilleure visualisation
    df["EstSalaryScore"] = df['estimatedsalary']
    df["NewEstimatedSalary"] = df["estimatedsalary"] / 12
    template = pd.DataFrame.from_records(
        [{col : 0 for col in ['geography_France', 'geography_Germany', 'geography_Spain', 'gender_Male']}]
    )
    if df['gender'].iloc[0] == 'Male':
        template['gender_Male'] = 1
    if df['geography'].iloc[0] == 'France':
        template['geography_France'] = 1
    if df['geography'].iloc[0] == 'Germany':
        template['geography_Germany'] = 1
    if df['geography'].iloc[0] == 'Spain':
        template['geography_Spain'] = 1

    df = pd.concat([df, template], axis=1)

    df = pd.get_dummies(df, columns=["geography", "gender"], drop_first=True)

    return df

# Créer des sliders pour les entrées de données
creditscore = st.slider('Credit Score', 300, 850, 650)
age = st.slider('Age', 18, 100, 30)
tenure = st.slider('Tenure', 0, 20, 5)
geography = st.selectbox('geography', ['France', 'Germany', 'Spain'])
gender = st.selectbox('Gender', ['Male', 'Female'])
balance = st.slider('Balance', 0, 300000, 150000)
numofproducts = st.slider('Number of Products', 1, 4, 2)
isactivemember = st.selectbox('Active mumber', ['0', '1'])
hascrcard = st.selectbox('hascrcard', ['0', '1'])
estimatedsalary = st.slider('Estimated Salary', 10000, 200000, 50000)


if st.button('Prédire le Churn'):
    input_data = pd.DataFrame([[creditscore, geography, gender, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary]],
                              columns=['creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary'])
    input_data = preprocess_data(input_data)
    input_data = input_data.values.astype(np.float32)

    input_data = np.reshape(input_data, (input_data.shape[0], 1, input_data.shape[1]))

    # Faire la prédiction
    prediction = model.predict(input_data)
    churn = np.argmax(prediction, axis=1)

    # Afficher le résultat
    if churn[0] == 0:
        st.write("Le client ne va probablement pas quitter.")
    else:
        st.write("Le client risque de quitter.")