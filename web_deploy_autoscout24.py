#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 1 07:03:58 2022

@author: groupelegant
"""

#Import libraries
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
        #Based on our optimal features selection
        st.subheader("Initials of Vehicle data")
        mileage = st.slider('Mileage of Vehicle ', 0,300000,1000)
        make_model = st.slider('make_model',0,800,0)
        First_Registration_Year = st.slider('First_Registration_Year' ,0,2022,1980)
        Gearbox = st.slider('Gearbox', 0,2,0)
        Engine_Size_cc = st.slider('Engine_Size_cc', 0,3000,600)
        fuel_type = st.slider('fuel_type', 0,9,0)
        seats = st.slider('seats', 0,9,0)
        doors = st.slider('doors', 0,6,0)
        gears = st.slider('gears', 0,9,0)
        cylinders = st.slider('cylinders', 0,12,0)
        Empty_Weight_kg = st.slider('Empty_Weight_kg', 0,5000,300)
        co2_emissions = st.slider('co2_emissions', 0,300,0)
        colour = st.slider('colour', 0,13,0)
        upholstery = st.slider('upholstery', 0,5,0)
        

        st.subheader("Details of Vehicle Data")                       
        title_0 = st.selectbox('Full Service History:', ('No','Yes'))
        title_1 = st.selectbox('Non Smoker Vehicle:', ('No','Yes'))
        title_2 = st.selectbox('Air Conditioning:', ('Yes','No'))
        title_3 = st.selectbox('Air Suspension:', ('No','Yes'))               # fsize = st.number_input('Passenger Family Size', min_value=0, max_value=9, value=1)
        title_4 = st.selectbox('Cruise Control:', ('Yes','No'))
        title_5 = st.selectbox('Navigation System:', ('No','Yes'))                        
        title_6  = st.selectbox('ABS:', ('Yes','No')) 
        title_7  = st.selectbox('Sunroof:', ('No','Yes')) 
        title_8  = st.selectbox('Rain_sensor', ('No','Yes'))
        title_9 = st.selectbox('Alarm System:', ('No','Yes')) 
        title_10 = st.selectbox('Emergency System:', ('No','Yes')) 
        title_11 = st.selectbox('Immobilizer:', ('No','Yes')) 
        title_12 = st.selectbox('Bluetooth:', ('No','Yes'))                     
        title_13 = st.selectbox('Radio:', ('No','Yes'))
        
        



        
        data = {
                'Mileage': mileage,
                'make_model': make_model,
                'First_Registration_Year':First_Registration_Year,
                'Gearbox': Gearbox,
                'Engine_Size_cc': Engine_Size_cc,
                'fuel_type': fuel_type,
                'seats': seats,
                'doors': doors,
                'gears': gears,
                'cylinders': cylinders,
                'Empty_Weight_kg': Empty_Weight_kg,
                'co2_emissions': co2_emissions,
                'colour': colour,
                'upholstery': upholstery,
                'title_0':title_0, 
                'title_1':title_1, 
                'title_2':title_2, 
                'title_3':title_3, 
                'title_4':title_4,
                'title_5':title_5,
                'title_6':title_6, 
                'title_7':title_7, 
                'title_8':title_8, 
                'title_9':title_9, 
                'title_10':title_10, 
                'title_11':title_11,
                'title_12':title_12, 
                'title_13':title_13, 

                
            
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
