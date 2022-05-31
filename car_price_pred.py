#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('car data.csv')


# In[3]:


df.head()


# In[5]:


df.shape


# In[9]:


print(df['Seller_Type'].unique())
print(df['Fuel_Type'].unique())
print(df['Transmission'].unique())
print(df['Owner'].unique())


# In[ ]:


# MIssing Values


# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


df.columns


# In[15]:


final_dataset = df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]


# In[16]:


final_dataset.head()


# In[17]:


final_dataset['Current_Year']=2020


# In[18]:


final_dataset.head()


# In[19]:


final_dataset['no_years']=final_dataset['Current_Year']-final_dataset['Year']


# In[20]:


final_dataset.head()


# In[22]:


final_dataset.drop(['Year'],axis=1,inplace=True)


# In[24]:


final_dataset.drop(['Current_Year'],axis=1,inplace=True)


# In[25]:


final_dataset.head()


# In[27]:


final_dataset = pd.get_dummies(final_dataset,drop_first=True)


# In[28]:


final_dataset


# In[29]:


final_dataset.corr()


# In[35]:


final_dataset.corr().index


# In[33]:


import seaborn as sns
sns.pairplot(final_dataset)


# In[34]:


import seaborn as sns
sns.pairplot(final_dataset.corr())


# In[36]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[37]:


corrmat = final_dataset.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g = sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')


# In[39]:


final_dataset.head(1)


# In[40]:


X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]


# In[47]:


X


# In[43]:


# Feature importance


# In[44]:


from sklearn.ensemble import ExtraTreesRegressor


# In[45]:


model = ExtraTreesRegressor()
model.fit(X,y)


# In[46]:


print(model.feature_importances_)


# In[52]:


feat_imp = pd.Series(model.feature_importances_,index=X.columns)
feat_imp.nlargest(5).plot(kind='bar')
plt.show()


# In[53]:


from sklearn.model_selection import train_test_split


# In[54]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.32)


# In[56]:


X.shape


# In[55]:


X_train.shape


# In[57]:


X


# In[58]:


X_train


# In[59]:


from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()


# In[61]:


import numpy as np


# In[62]:


## HYper parameters

# No of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100,stop = 1200, num = 12)]

print(n_estimators)


# In[63]:


# no of features to consider at every split
max_features = ['auto', 'sqrt']


# In[64]:


# max no of levels in tree
max_depth = [int(x) for x in np.linspace(5,30,num=6)]


# In[65]:


# min no of samples required to split a node
min_samples_split = [2,5,10,15,100]


# In[66]:


# Min no of samples required at each leaf node
min_samples_leaf = [1,2,5,10]


# In[67]:


from sklearn.model_selection import RandomizedSearchCV


# In[71]:


# Create the Random grid
random_grid  = {'n_estimators':n_estimators,
                'max_depth': max_depth,
                'max_features':max_features,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf}

print(random_grid)


# In[72]:


rf = RandomForestRegressor()


# In[73]:


rd_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=42,n_jobs=1)


# In[74]:


rd_random.fit(X_train,y_train)


# In[76]:


predictions=rd_random.predict(X_test)


# In[77]:


predictions


# In[79]:


sns.distplot(y_test-predictions)


# In[83]:


plt.scatter(y_test,predictions)


# In[84]:


import pickle
# open a file, where you want to store data
file = open('random_forest_regression_model.pkl','wb')

#dump information to that file
pickle.dump(rd_random,file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




