#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - GRIP July 2021
# By: SAKSHI MOGHA
# Task 1: Prediction using Supervised Machine Learning

# In[34]:


#Importing all the libraries that are required in this notebook

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# Reading the Data

# In[28]:


#Reading the data

url = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
df=pd.read_csv(url)
print("Data Imported Successfully")
df.head()


# In[29]:


df.shape


# In[30]:


df.info()


# In[31]:


df.describe()


# Plotting the Distribution of Scores

# In[32]:


df.plot(x='Hours',y='Scores', style='o', grid=True, legend=True)
plt.title("Hours of Studies vs Percentage Scored")
plt.xlabel("Hours")
plt.ylabel("Percentage")
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score
# Preparing the Data
# Divide the data into "attributes" and "labels"

# In[33]:


x = df.iloc[:,:-1].values
y = df.iloc[:,1].values


# Once we have attributes and labels, next we have to split the data in our training and test sets.

# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


xtrain, xtest, ytrain, ytest = train_test_split(x ,y ,test_size = 0.2 ,random_state = 0)


# Once the data is split into training and test sets, then next we have to train our algorithm.

# In[40]:


from sklearn.linear_model import LinearRegression


# In[41]:


reg = LinearRegression()
reg.fit(xtrain, ytrain)
print("Training Successfully completed")


# In[42]:


print("Intercept :", reg.intercept_)
print("Co-efficent :", reg.coef_)


# In[43]:


#Plotting the regression line
line = reg.coef_*x+reg.intercept_

#Plotting for the test data
plt.scatter(x, y)
plt.plot(x, line, color='Red')
plt.show()


# Making Predictions

# In[44]:


#Test of data in hours
print(xtest)  

#Predicting the scores
ypred = reg.predict(xtest)


# In[45]:


#Comparing Actual vs Predicted 
df2=pd.DataFrame({"Actual": ytest, "Predicted": ypred})
df2.head()


# In[46]:


reg.score(x, y)


# Evaluating the Model

# In[47]:


from sklearn import metrics


# In[48]:


print("Mean Absolute Error :",metrics.mean_absolute_error(ytest, ypred))
print("Mean Squared Error :", metrics.mean_squared_error(ytest, ypred))
print("Root mean squared Error :", np.sqrt(metrics.mean_squared_error(ytest, ypred)))


# In[49]:


hours = 9.25
value1= np.array([hours])
value1= value1.reshape(-1,1)
ownprediction = reg.predict(value1)
print("The Predicted Score for 9.25 Hours is :",format(ownprediction[0]))


# # Conclusion : The Predicted Percentage / Score for studying for 9.25 hours is 93.69 %
