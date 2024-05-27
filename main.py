import streamlit as sl
import pickle
import numpy as np
import pandas as pd
sl.title('Heart Disease Prediction')

with sl.form('User_record'):
    gender= sl.radio(
            'Gender',
            {'Male','Female'},
        )
    age=sl.text_input('Age')
    
    education=sl.radio('Education',
                       ['Uneducated','High School', 'Bachelor\'s Degree', 'Graduate Degree']
                       )
    currentSmoker=sl.radio('Current a Smoker',
                           ['Yes','No']
                           )
    cigsPerDay=sl.text_input('Number of cigarette per day')
    BPMeds=sl.radio('Currently on blood pressure medication',
                           ['Yes','No']
                           )
    prevalentStroke=sl.radio('Had a stroke previously ',
                           ['Yes','No']
                           )
    PrevalentHyp=sl.radio('Presence of hypertension (high blood pressure)',
                           ['Yes','No']
                           )
    diabetes=sl.radio('Presence of diabets',
                      ['Yes','No']
                      )
    totChol=sl.text_input('Total Cholesterol')
    sysBP=sl.text_input('Systolic Blood Pressure')
    DiaBP=sl.text_input('Diastolic Blood Pressure')
    BMI=sl.text_input('Body Mass Index')
    heartRate=sl.text_input('heartRate')
    glucose=sl.text_input('glucose')

    submit=sl.form_submit_button('Predict')
record=[0]*15
education_mapping = {
    'Uneducated': 1,
    'High School': 2,
    'Bachelor\'s Degree': 3,
    'Graduate Degree': 4
}
if submit:
    record[0]=1 if gender=='Male' else 0
    record[1]=float(age)
    record[2]=float(education_mapping[education])
    record[3]=1 if currentSmoker == 'Yes' else 0
    record[4]=float(cigsPerDay)
    record[5]=1 if BPMeds=='Yes' else 0
    record[6]=1 if prevalentStroke=='Yes' else 0
    record[7]=1 if PrevalentHyp=='Yes' else 0
    record[8]=1 if diabetes=='Yes' else 0
    record[9]=float(totChol)
    record[10]=float(sysBP)
    record[11]=float(DiaBP)
    record[12]=float(BMI)
    record[13]=float(heartRate)
    record[14]=float(glucose)
    
    record=np.array(record).reshape(1, -1)
    record=pd.DataFrame(record)
   
    with open('RandomForest','rb') as rf:
        randomforest=pickle.load(rf)
    with open('LogisticRegression','rb') as lr:
        logisticRegression=pickle.load(lr)
    with open('knn','rb') as knn:
        knn=pickle.load(knn)

     
    sl.write(f'Random Forest model : { "Chance of Heart disease" if randomforest.predict(record)==1 else "Less or No Chance of Heart Disease"} ')
    sl.write(f'Logistic Regression model : { "Chance of Heart disease" if logisticRegression.predict(record)==1 else "Less or No Chance of Heart Disease"} ')
    sl.write(f'K Nearest Neighbors model : { "Chance of Heart disease" if knn.predict(record)==1 else "Less or No Chance of Heart Disease"} ')