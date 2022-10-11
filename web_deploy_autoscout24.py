#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 1 07:03:58 2022

@author: groupelegant
"""

#Import libraries
from matplotlib.backend_bases import LocationEvent
import streamlit as st
import pandas as pd
import numpy as np


from PIL import Image



#load the model from disk
import joblib
filename = 'RandomForestRegressor.sav'
model = joblib.load(filename)

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Fenyx Autoscout24 Model App')

      #Setting Application description
    st.markdown("""
     :dart: Data for Autoscout24 \n
     :dart:  Data for Autoscout24. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('App.png')
    image1 = Image.open('importance.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Autoscout24 use case')
    st.sidebar.image(image)
    st.sidebar.info('This app uses Random Forest Model')
    st.sidebar.image(image1)
    if add_selectbox == "Online":
        st.info("Input data below")
        st.subheader("Initials of Vehicle data")
        #Based on our optimal features selection
        location = st.slider('Location', 0,3000,0)
        First_Registration_Year = st.slider('First Registration Year' ,0,2022,1980)
        Power_kW = st.slider('Power kW', 0,220,0)
        Empty_Weight_kg = st.slider('Empty Weight (kg)', 0,5000,300)
        mileage = st.slider('Mileage of Vehicle (km) ', 0,300000,1000)
        gears = st.slider('Gears', 0,9,0)
        make_model = st.slider('Make Model',0,800,0)
        cylinders = st.slider('Cylinders', 0,12,0)
        Engine_Size_cc = st.slider('Engine Size (cc)', 0,3000,600)
        Gearbox = st.slider('Gearbox', 0,2,0)
        fuel_type = st.slider('Fuel Type', 0,9,0)
        fuel_country = st.slider('Fuel Country', 0,7,0)
        fuel_city = st.slider('Fuel City', 0,16,0)
        fuel_comb = st.slider('Fuel Comb', 0,10,0)
        co2_emissions = st.slider('CO2 Emissions', 0,300,0)
        body_type = st.slider('Body Type', 0,5,0)
        seats = st.slider('Count of Seats', 0,9,0)
        doors = st.slider('Count of Doors', 0,6,0)
        colour = st.slider('Colour', 0,13,0)
        upholstery = st.slider('Upholstery', 0,5,0)
        drivetrain = st.slider('Drivetrain', 0,3,0)

     
        

        st.subheader("Details of Vehicle Data")                       
        title_0 = st.selectbox('LED Headlights:', ('No','Yes'))
        title_1 = st.selectbox('Digital cockpit:', ('No','Yes'))
        title_2 = st.selectbox('Heated streering weel:', ('No','Yes'))
        title_3 = st.selectbox('Panorama roof:', ('No','Yes'))
        title_4 = st.selectbox('Electrically adjustable seats:', ('No','Yes'))
        title_5 = st.selectbox('Emergency System:', ('No','Yes')) 
        title_6 = st.selectbox('Electric tailgate:', ('No','Yes'))
        title_7 = st.selectbox('High bean assist:', ('No','Yes'))  
        title_8 = st.selectbox('Full Service History:', ('No','Yes'))



        
        data = {
                'Location' : location,
                'First Registration Year':First_Registration_Year,
                'Power kW' : Power_kW,
                'Empty Weight (kg)': Empty_Weight_kg,
                'Mileage of Vehicle (km)': mileage,
                'Gears': gears,
                'Make Model': make_model,
                'Cylinders': cylinders,
                'Engine Size (cc)': Engine_Size_cc,
                'Gearbox': Gearbox,
                'Fuel Type': fuel_type,
                'Fuel Country' : fuel_country,
                'Fuel City' : fuel_city,
                'Fuel Comb' : fuel_comb,
                'CO2 Emissions': co2_emissions,
                'Body Type' : body_type,
                'Count of Seats': seats,
                'Count of Doors': doors,
                'Colour': colour,
                'Upholstery': upholstery,
                'Drivetrain' : drivetrain,
                
                'title_0':title_0, 
                'title_1':title_1, 
                'title_2':title_2, 
                'title_3':title_3, 
                'title_4':title_4,
                'title_5':title_5,
                'title_6':title_6, 
                'title_7':title_7, 
                'title_8':title_8, 
                                
                }

        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Your vehicles value is:')

        

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file,encoding= 'utf-8')
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
               #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the passenger survive.', 0:'No, the passenger died'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()
