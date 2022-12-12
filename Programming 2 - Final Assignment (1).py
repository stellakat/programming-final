#!/usr/bin/env python
# coding: utf-8

# # Programming 2 - Final Assignment
# ## Sumita Tellakat
# ### December 8, 2022

# #### Import Packages 

# In[7]:


import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotnine as p9
import plotly.express as px
import altair as alt
import pandas as pd
from plotnine import *
from plotnine.data import mpg


# ---

# #### Import Data

# In[8]:


s = pd.read_csv("C:/Users/sumit/Downloads/social_media_usage.csv")


# In[9]:


print(s.shape)


# In[10]:


print(s.head)


# ---

# #### Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[13]:


toy_data = {'first_column': ['Luna', 'Hazel', 'Pepper'],
        'second_column': [5, 3, 1]
        }

x = pd.DataFrame(toy_data)
print(x)


# In[14]:


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x
clean_sm(x)


# In[15]:


clean_sm(x)


# ---

# #### Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[16]:


ss = pd.DataFrame({
    "sm_li":np.where(s["web1h"] == 1, 1, 0),
    "income":np.where(s["income"]> 9,np.nan,s["income"]),
    "education":np.where(s["educ2"]> 8,np.nan,s["educ2"]),
    "parent":np.where(s["par"] == 1,1,0),
    "married": np.where(s["marital"] ==1,1,0),
    "female": np.where(s["gender"] ==2,1,0),
    "age":np.where(s["age"] >98, np.nan,s["age"])})


# In[17]:


ss = ss.dropna()
ss


# Looking at the preliminary values presented above, it is evident that middle-aged individuals with higher education levels and higher incomes are more likely to use LinkedIn than their older counterparts. Additionally, it is evident that individuals with lower education levels are less likely to be on LinkedIn regardless of age.
# 
# ---

# #### Create a target vector (y) and feature set (X)

# In[18]:


y = ss["sm_li"]
X = ss[["income", "education", "parent", "married", "female", "age"]]


# ---

# #### Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,      
                                                    test_size=0.2,    
                                                    random_state=313) 


# Each of the X values contain income, education, parent, marriage status, gender, and age variables. These are then further split into a training variable and a test variable - 80% in training and 20% in test. THe same is done to the Y vales which contain a binary variable indicating whether or not someone is a LinkedIn user. This is important because it allows us to appropriately train the model and confirm that it is effective in answering our questions - we can then use this effectively trained model to predict outcomes using the test data. This allows us to prove the validity and usefulness of our model.
# 
# ---

# #### Initiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[20]:


lr = LogisticRegression(class_weight = "balanced")
lr.fit(X_train, y_train)


# ---

# #### Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[21]:


y_pred = lr.predict(X_test)


# In[27]:


confusion_matrix(y_test, y_pred)


# In[23]:





# #### Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[28]:


LinkedIn_Df = pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["True negative", "False positive"],
            index=["False negative","True positive"])


# In[29]:


print(LinkedIn_Df)


# ---

# #### Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# In[31]:


#### Precision - by hand
52/(52+61)


# Precision calculates correct positive predictions relative to total positive predictions. In this particular case it is evident that the model predicts negative outcomes with 81% accuracy, while it predicts positive outcomes with 48% accuracy. It is best to use precision as a guiding metric when we want the prediction of 1 to be as correct as possible.

# In[32]:


#### Recall - by hand
52/(52+32)


# Use recall when we want our model to spot as many real positive results as possible. In this case - since LinkedIn usage is not life or death - out of the total positive results, 71% are predicted to be positive and recall is a strong metric to use.

# In[33]:


#### f1-score - by hand
2 * (.46 * .619) / (.46 + .619)


# The F1 score gives the same weightage to recall and precision and gives us a harmonic mean. In this case, it shows that the model is correct about negative outcomes 69% of the time and correct about positive outcomes 57% of the time. The f1-score is the most commonly used, especially in cases when there is uneven class distribution.

# In[30]:


print(classification_report(y_test, y_pred))


# ---

# #### Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[48]:


NewData = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82]    
})


# In[49]:


NewData


# In[50]:


NewData["prediction_linkedin"] = lr.predict(NewData)
NewData


# The data above shows that the model predicts that the 42 year old individual is a linkedin user and the 82 year old is not.

# ---

# In[ ]:




