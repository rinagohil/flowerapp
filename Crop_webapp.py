import streamlit as st
st.title("Crop Classification WebApp")
activities=['RF','DT']
option=st.sidebar.selectbox('Which model you use?',activities)
f1=st.slider('N', 0.0, 150.0)
f2=st.slider('P', 0.0, 150.0)
f3=st.slider('K', 0.0, 250.0)
f4=st.slider('temperature', 0.0, 100.0)  
f5=st.slider('humidity', 0.0, 100.0)
f6=st.slider('ph', 0.0, 10.0)
f7=st.slider('rainfall', 0.0, 300.0)  
import pandas as pd
#list Data
datau = {'N':[f1],'P':[f2],'K':[f3],
         'temperature':[f4],'humidity':[f5],
         'ph':[f6],'rainfall':[f7]}
df1 = pd.DataFrame(datau)
import pickle
if st.button('Predict'):        
    if option=='RF':
        RF_model=pickle.load(open('Crop_RF.pkl', 'rb'))
        st.success(RF_model.predict(df1))
    else:
        DT_model=pickle.load(open('Crop_DT.pkl', 'rb'))
        st.success(DT_model.predict(df1))

