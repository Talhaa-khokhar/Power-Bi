#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('titanic.csv')
data.shape


# In[2]:


data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data = data.drop(['Name', 'Ticket','Cabin'], axis=1)


# In[5]:


data.head()


# In[6]:


data['Embarked'].value_counts()


# In[7]:


data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])


# In[8]:


data['Embarked'].value_counts()


# In[9]:


# Replace null values with zero in the specific column
data['Age'] = data['Age'].fillna(0)


# In[10]:


data.isnull().sum()


# In[11]:


data.shape


# In[12]:


# Encode categorical data
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})


# In[13]:


# Scale numerical data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Split the data into training and testing sets
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[15]:


# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)


# In[16]:


probability = model.predict_proba(X_test)


# In[17]:


predictions_probability = probability[:, 1]


# In[18]:


import matplotlib.pyplot as plt

plt.bar(predictions, predictions_probability, color='blue')
plt.xlabel('Predictions (0: No Survival, 1: Survival)')
plt.ylabel('Prediction Probabilities')
plt.title('Prediction Probabilities for Titanic Passengers')
plt.show()


# In[20]:


# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy,2)*100)


# In[36]:


data.insert(4,"Predictictions",model.predict(X))
data.head()


# In[34]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Titanic dataset from a CSV file
data = pd.read_csv('C:/Users/Talha/Downloads/titanic.csv')

# Prepare the data for machine learning
# Handling missing values
data.dropna(inplace=True)

# Encode categorical data
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Scale numerical data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Split the data into training and testing sets
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)  # Changed X_train and y_train to X and y

# Make predictions on the entire dataset
predictions = model.predict(X)  # Changed X_test to X

# Insert predictions into the DataFrame
data.insert(4, "Predictions", predictions)

# Print the first few rows of the DataFrame
print(data.head())


# In[40]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

 # Load the Titanic dataset from a CSV file
data = pd.read_csv('C:/Users/Talha/Downloads/titanic.csv')

# Prepare the data for machine learning
# Handle missing values
data.dropna(inplace=True)

# Encode categorical data
data['Sex'] = data['Sex'].map({'male': 1, 'female': 0})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Scale numerical data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Split the data into training and testing sets
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)
data.insert(4,"Predictictions",model.predict(X))
# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)


# In[ ]:




