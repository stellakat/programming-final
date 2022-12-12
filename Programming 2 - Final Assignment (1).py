import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import streamlit as st
from PIL import Image

s = pd.read_csv("social_media_usage.csv")
ss = pd.DataFrame({
    "income":np.where(s["income"]> 9,np.nan,s["income"]),
    "education":np.where(s["educ2"]> 8,np.nan,s["educ2"]),
    "parent":np.where(s["par"] == 1,1,0),
    "married": np.where(s["marital"] ==1,1,0),
    "female": np.where(s["gender"] ==2,1,0),
    "age":np.where(s["age"] >98, np.nan,s["age"])})


ss = ss.dropna()
ss

y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,      
                                                    test_size=0.2,    
                                                    random_state=313) 


lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)


NewData = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82]    
})


NewData["prediction_linkedin"] = lr.predict(NewData)
NewData

st.title("Are you a LinkedIn User?")

image = Image.open("C:/Users/sumit/Desktop/FINAL/DahlgrenSpringFlowers.jpg")
st.image(image, caption='Old North in the spring')

income = st.slider("Income (low=1 to high=9)", 1, 9)
education = st.slider("Level of Education (less than high school=1 to advanced degrees=8)", 1, 8)
parent = st.selectbox("Do you have children?", ('Yes', 'No'))
married = st.selectbox("Are you married?", ('Yes', 'No'))
female = st.selectbox("What gender do you identify as?", ('Male', 'Female'))
age = st.slider("How old are you?", 0, 130)
sm_li = st.selectbox("Do you use LinkedIn", ('Yes', 'No'))


