#!/usr/bin/env python
# coding: utf-8

# ## GRIP:The Sparks Foundation.
# ## Data Science and Business Analytics intership
# ## Task 6: Prediction using Decision Tree Algorithum
# ## Author: Mayuri Ajay Kalange

# In[1]:


# Importing the required Libraries
import numpy as np
import pandas as pd 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns
import matplotlib.pyplot as plt


# ## Step 1: Import the dataset

# In[3]:


iris_data=pd.read_csv("C:/Users/DELL/Desktop/IRIS.csv")
iris_data


# ## Step 2: Exploratory Data Analysis

# In[4]:


iris_data.info()


# In[5]:


iris_data.tail()


# In[6]:


features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']


# In[7]:


x= iris_data.loc[:, features].values
print(x)


# In[8]:


y=iris_data.Species
print(y)


# ## Step 3: Data Visualization comparing various features

# In[9]:


sns.pairplot(iris_data)


# ## Step 4: Decision Tree Tree Model Training

# In[10]:


# Model Training
x_train, x_test, y_train, y_test= train_test_split(x, y, random_state=0)
clf = DecisionTreeClassifier(max_depth = 2, random_state = 0)
clf.fit(x_train, y_train)
clf.predict(x_test[0:1])


# ## Step 5: Calculating the Model accuracy

# In[11]:


score = clf.score(x_test, y_test)
print(score)


# In[13]:


from sklearn import metrics


# In[14]:


print(metrics.classification_report(y_test, clf.predict(x_test)))


# In[15]:


cm= metrics.confusion_matrix(y_test, clf.predict(x_test))


# In[16]:


plt.figure(figsize=(7,7))


# In[17]:


sns.heatmap(cm, annot=True,
           fmt='.0f',
           linewidths=.5,
           square= True,
           cmap = 'Blues');
plt.ylabel('Actual label',fontsize = 17);
plt.xlabel('Predicted label', fontsize = 17);
plt.title('Accuracy Score: {}'.format(score),size =17);
plt.tick_params(labelsize= 15)


# In[18]:


accuracy = []


# In[19]:


Max_depth_range= list(range(1, 6))
print(Max_depth_range)


# In[20]:


for depth in Max_depth_range:
    
    clf = DecisionTreeClassifier(max_depth = depth,
                                 random_state = 0)
    clf.fit(x_train, y_train)
    
    score = clf.score(x_test, y_test)
    accuracy.append(score)


# ## Step 6: Visualizing the Trained Model

# In[21]:


fn= ['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm']
cn= ['setosa','versicolor','virginica']


# In[22]:


fig, axes = plt.subplots(nrows =1, ncols= 1, figsize= (7,4), dpi = 150)
tree.plot_tree(clf,
               feature_names = fn,
               class_names= cn,
               filled = True);


# ## Thank you

# In[ ]:




