#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 1 07:03:58 2022

@author: groupelegant
"""

#Import libraries
# from matplotlib.backend_bases import LocationEvent
import streamlit as st
import pandas as pd
import numpy as np


from PIL import Image



#load the model from disk
import joblib
filename = 'RandomForestRegressor.sav'
model = joblib.load(filename)

#Import python scripts
from preprocessing_Elegant import preprocess, fuel_countryd, locationd, make_modeld, Gearboxd, Upholstery_Colord, upholsteryd, drivetraind, typed,body_typed, sellerd, fuel_typed, colourd, n_dictd

def main():
    #Setting Application title
    st.title('Group Elegant Autoscout24 Model App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict AutoScout use case.
    The application is functional for online prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
#     colours_name = ['Please choose colour','Grey', 'White', 'Black', 'Red', 'Silver', 'Brown', 'Blue',
#        'Violet', 'Beige', 'Yellow', 'Green', 'Orange', 'Bronze', 'Gold']
#     drivetrain_names = ['Please choose type of drivetrain','Front', '4WD', 'Rear']
#     location_names = ['Please choose city','BREDA','ALMERE','UTRECHT','GRONINGEN','DEN BOSCH','AMSTERDAM','ROTTERDAM','HAARLEM','OSS',
#  'APELDOORN','ZWOLLE','TILBURG','EINDHOVEN','AMERSFOORT','LEEUWARDEN','ARNHEM','VENLO','ENSCHEDE','DEN HAAG','MAASTRICHT','DELFT','LEIDEN',]
#     fuel_types = ['Please choose fuel type','Gasoline', 'Diesel', 'Electric/Diesel', 'LPG', 'Electric','Electric/Gasoline', 'CNG', 'Ethanol', 'Others', 'Hydrogen']
#     upholstery_names = ['Please choose upholstery type','Cloth', 'Partleather', 'Fullleather', 'alcantara', 'Other','Velour']
    
    
    
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
        st.subheader("The Most Important Features of Model:")
        #Based on our optimal features selection
        location = st.selectbox('Location of the Auto' , (locationd))
        First_Registration_Year = st.slider('First Registration Year' ,1980,2022,2020)     #st.number_input('First Registration Year of the Car', min_value=1940, max_value=2022, value=2000)
        Power_kW = st.slider('Power kW', 0,220,100)
        Empty_Weight_kg = st.slider('Empty Weight (kg)', 300,5000,2000)
        mileage = st.slider('Mileage of Vehicle (km) ', 1000,300000,100000)
        gears = st.selectbox('Gear of the Car', ('1','2','3', '4','5','6','7', '8','9'))
        make_model = st.selectbox('Make and Model of the Auto' , (make_modeld))
        cylinders = st.selectbox('Cylinder of the Car', ('1','2','3', '4','5','6','7', '8','9','10'))
        Engine_Size_cc = st.slider('Engine Size (cc)', 600,3000,1400)
        Gearbox = st.selectbox('Gearbox', (Gearboxd))
        fuel_type = st.selectbox('Fuel Type', (fuel_typed))
        fuel_country = st.selectbox('Fuel Country', (fuel_countryd))
        fuel_city = st.slider('Fuel City', 0,16,10)
        fuel_comb = st.slider('Fuel Comb', 0,10,5)
        co2_emissions = st.slider('CO2 Emission per km', 0,300,90)
        body_type = st.selectbox('Body Type', (body_typed))
        seats = st.selectbox('Seat of the Car', ('1','2','3', '4','5','6','7', '8','9'))
        doors = st.selectbox('Door of the Car', ('1','2','3', '4','5','6','7', '8','9'))
        colour = st.selectbox('Colour', (colourd))
        upholstery = st.selectbox('Upholstery', (upholsteryd))
        drivetrain = st.selectbox('Drivetrain', (drivetraind))

        
        

        # st.subheader("Details of Vehicle Data")                       
        # title_0 = st.selectbox('LED Headlights:', ('No','Yes'))
        # title_1 = st.selectbox('Digital cockpit:', ('No','Yes'))
        # title_2 = st.selectbox('Panorama roof:', ('No','Yes'))
        # title_3 = st.selectbox('Full Service History:', ('No','Yes'))



        
        data = {
                'Location' : location,
                'First Registration Year':First_Registration_Year,
                'Power kW' : Power_kW,
                'Empty Weight (kg)': Empty_Weight_kg,
                'Mileage of Vehicle (km)': mileage,
                'Gears': n_dictd[gears],
                'Make Model': make_model,
                'Cylinders': n_dictd[cylinders],
                'Engine Size (cc)': Engine_Size_cc,
                'Gearbox': Gearbox,
                'Fuel Type': fuel_type,
                'Fuel Country' : fuel_country,
                'Fuel City' : fuel_city,
                'Fuel Comb' : fuel_comb,
                'CO2 Emissions': co2_emissions,
                'Body Type' : body_type,
                'Count of Seats':  n_dictd[seats],
                'Count of Doors': n_dictd[doors],
                'Colour': colour,
                'Upholstery': upholstery,
                'Drivetrain' : drivetrain,
                
                # 'title_0':title_0, 
                # 'title_1':title_1, 
                # 'title_2':title_2, 
                # 'title_3':title_3, 

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
        # uploaded_file = st.file_uploader("Choose a file")
        # if uploaded_file is not None:
        #     data = pd.read_csv(uploaded_file,encoding= 'utf-8')
        #     #Get overview of data
        #     st.write(data.head())
        #     st.markdown("<h3></h3>", unsafe_allow_html=True)
        #     #Preprocess inputs
        #     preprocess_df = preprocess(data, "Batch")
        #     if st.button('Predict'):
        #        #Get batch prediction
        #         prediction = model.predict(preprocess_df)
        #         prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
        #         prediction_df = prediction_df.replace({1:'Yes, the passenger survive.', 0:'No, the passenger died'})

        #         st.markdown("<h3></h3>", unsafe_allow_html=True)
        #         st.subheader('Prediction')
        #         st.write(prediction_df)
            
if __name__ == '__main__':
        main()
