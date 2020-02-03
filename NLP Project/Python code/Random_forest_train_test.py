#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy  as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import pickle 


# In[19]:


def build_search_model(search_model):
    """this function takes the search model then train and test the model 
    and return predicted labels,confusion_matrix,classification_report, accuracy_score"""
    
    print("search_model---> Done")
    search_model.fit(X_train,y_train)
    print("fitting the model--> Done")
    best_search = search_model.best_estimator_
    print("choosing the best values--->Done")
    y_pred_search=best_search.predict(X_test)
    print("predicting labels---> Done")
    
    V1=y_pred_search
    V2=confusion_matrix(y_test,y_pred_search)
    V3=classification_report(y_test,y_pred_search)
    V4=accuracy_score(y_test, y_pred_search)*100
    V5= best_search
    print("Done")
    #evaluating the grid search best model 
    #return y_pred_search, confusion_matrix(y_test,y_pred_search),classification_report(y_test,y_pred_search), accuracy_score(y_test, y_pred_search)*100
    return  V1, V2, V3, V4, V5


# In[2]:


clnd_data=pd.read_csv("correct_labels_.csv")


# In[3]:


clnd_data.head()


# # word embedding 

# In[4]:


tfidf_vctrr= TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
tfidf_features = tfidf_vctrr.fit_transform(clnd_data['Reviews']).toarray()


# In[5]:


X=tfidf_features
Y=clnd_data["Label"]


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(tfidf_features, Y, test_size=0.2, random_state=0)


# In[9]:


#y_train


# # Building the model

# In[10]:


#training the model on 80% of the datset 
rand_forest = RandomForestClassifier(n_estimators=200, random_state=0)
rand_forest.fit(X_train, y_train)


# In[13]:


#testing the model on 20% of the dataset 
#predicting the labels for the test set 
y_pred = rand_forest.predict(X_test)


# In[14]:


#evaluating the model 
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy: ",accuracy_score(y_test, y_pred)*100)


# # the model performs better without tuning so we will save this random forest model 

# # Saving the model 

# In[91]:


#saving the model to pickle file 
with open("model_random_forest.pickle","wb") as f:
    pickle.dump(rand_forest,f)


# In[92]:


#open the model 
with open("model_random_forest.pickle","rb") as f:
    md=pickle.load(f)


# In[93]:


#use the imported model to predict the labels for the test set 
md.predict(X_test)


# # searching for the best parameters 

# In[16]:


#setting parameters for random search with 3 folds cross validation 
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[17]:


#create instance of Random forest classifier 
rf = RandomForestClassifier()
#random search with 3 folds cross validation & 100 iterations 
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)


# In[39]:


pred_rand_search_labels,conf_mtrx_rand_search,class_report_rand_search,acc_score_rand_search,model_rand_search = build_search_model(rf_random)


# In[49]:


print(conf_mtrx_rand_search,class_report_rand_search,"random search Accuracy: ",acc_score_rand_search)


# In[18]:


#grid search parameters 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [10,100, 200, 300, 1000]
}

# Create an instance of random forest classifier model
rf_G_search = RandomForestClassifier(random_state = 42)

# Create an instance of the grid search model 
grid_search = GridSearchCV(estimator = rf_G_search, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, return_train_score=True)


# In[21]:


pred_grid_search_labels,conf_mtrx_grid_search,class_report_grid_search,acc_score_grid_search,model_grid_search = build_search_model(grid_search)


# In[24]:


print(conf_mtrx_grid_search,class_report_grid_search,"Grid search Accuracy: ",acc_score_grid_search )


# In[22]:


#saving the model to pickle file 
with open("model_grid_search.pickle","wb") as f:
    pickle.dump(model_grid_search,f)


# In[23]:


model_grid_search


# In[37]:


#open the model 
with open("model_grid_search.pickle","rb") as f:
    md2=pickle.load(f)


# In[38]:


md2.predict(X_test)


# In[39]:


y_test


# In[ ]:




