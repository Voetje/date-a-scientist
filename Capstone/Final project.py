
# coding: utf-8

# ## Libraries:

# In[5]:


import pandas as pd


# In[6]:


import numpy as np


# In[7]:


from sklearn import preprocessing


# In[8]:


from sklearn.neighbors import KNeighborsClassifier


# In[9]:


from matplotlib import pyplot as plt


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


from sklearn.neighbors import KNeighborsRegressor


# In[63]:


from sklearn.metrics import accuracy_score


# In[69]:


from sklearn.metrics import precision_score


# In[14]:


from sklearn.cluster import KMeans


# In[15]:


import timeit


# In[108]:


from sklearn.svm import SVC


# In[109]:


from sklearn.metrics import classification_report, confusion_matrix


# ## Dataset:

# In[16]:


dataset = pd.read_csv("https://img.springe.st/profiles.csv")


# In[17]:


df = pd.DataFrame(dataset)


# In[18]:


df = df.dropna()


# ## Adding columns:

# ### Education:

# In[19]:


education_mapping = {"dropped out of space camp": 0, "working on space camp": 1, "space camp": 1, "graduated from space camp": 2,
                     "dropped out of high school": 0, "working on high school": 3, "high school": 3, "graduated from high school": 4,
                    "dropped out of two-year college": 4, "working on two-year college": 5, "two-year college": 5, "graduated from two-year college": 6,
                    "dropped out of college/university": 6, "working on college/university": 7, "college/university": 7, "graduated from college/university": 8,
                    "dropped out of masters program": 8, "working on masters program": 9, "masters program": 9, "graduated from masters program": 10,
"dropped out of law school": 8, "working on law school": 9, "law school": 9, "graduated from law school": 10,
"dropped out of med school": 8, "working on med school": 9, "med school": 9, "graduated from med school": 10,
"dropped out of ph.d program": 10, "working on ph.d program": 11, "ph.d program": 11, "ph.d program": 12}


# In[20]:


df["education_code"] = df.education.map(education_mapping)


# ### Religion:

# In[21]:


religion_mapping = {"catholicism and very serious about it": 5, "catholicism and somewhat serious about it":4, "catholicism": 3, "catholicism but not too serious about it":2, "catholicism and laughing about it": 1,
                    "christianity and very serious about it": 5, "christianity and somewhat serious about it":4, "christianity": 3, "christianity but not too serious about it":2, "christianity and laughing about it": 1, 
                    "atheism and very serious about it": 0, "atheism and somewhat serious about it":0.1, "atheism": 0.2, "atheism but not too serious about it":0.3, "atheism and laughing about it": 0.4, "agnosticism and very serious about it": 1, 
                    "agnosticism and somewhat serious about it":0.8, "agnosticism": 0.6, "agnosticism but not too serious about it":0.4, "agnosticism and laughing about it": 0.2, 
                    "other and very serious about it": 5, "other and somewhat serious about it":4, "other": 3, "other but not too serious about it":2, "other and laughing about it": 1, 
                    "judaism and very serious about it": 5, "judaism and somewhat serious about it":4, "judaism": 3, "judaism but not too serious about it":2, "judaism and laughing about it": 1, 
                    "buddhism and very serious about it": 5, "buddhism and somewhat serious about it":4, "buddhism": 3, "buddhism but not too serious about it":2, "buddhism and laughing about it": 1,
                    "hinduism and very serious about it": 5, "hinduism and somewhat serious about it":4, "hinduism": 3, "hinduism but not too serious about it":2, "hinduism and laughing about it": 1,
                    "islam and very serious about it": 5, "islam and somewhat serious about it":4, "islam": 3, "islam but not too serious about it":2, "islam and laughing about it": 1  }


# In[22]:


df["religion_code"] = df.religion.map(religion_mapping)


# ### Drinks:

# In[23]:


drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}


# In[24]:


df["drinks_code"] = df.drinks.map(drink_mapping)


# ### Drugs:

# In[25]:


drugs_mapping = {"never": 0, "sometimes": 1, "ofter": 2}


# In[26]:


df["drugs_code"] = df.drugs.map(drugs_mapping)


# ### Smokes:

# In[27]:


smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "trying to quit": 3, "yes": 4,}


# In[28]:


df["smokes_code"] = df.smokes.map(smokes_mapping)


# ## Removing income not fileld in:

# In[29]:


df = df[(df[['income']] != -1).all(axis=1)]


# ## Min-Max scaling:

# In[35]:


feature_data = df[['smokes_code', 'drinks_code', 'drugs_code', 'religion_code', 'education_code']]


# In[36]:


df = df.dropna()


# In[37]:


x = feature_data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)


feature_data = pd.DataFrame(x_scaled, columns=feature_data.columns)


# # Dataset 1:

# In[38]:


x = df[['religion_code','education_code']]
y = df[['income']]


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)


# # Dataset 2:

# In[40]:


x2 = df[['religion_code','education_code', 'smokes_code', 'drugs_code', 'drinks_code', 'age']]
y2 = df[['income']]


# In[41]:


x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, train_size = 0.8, test_size = 0.2, random_state=6)


# # Multiple Linear Regression model:

# ### Creating the model

# In[162]:


model = LinearRegression()


# ### Training the model with dataset 1

# In[164]:


model.fit(x2_train, y2_train)


# ### Measuring accuracy with dataset 1:

# In[165]:


print(model.score(x2_train, y2_train))
print(model.score(x2_test, y2_test))


# ## measure recall and precision:

# In[127]:


## can't be done, only score is possible
##y_pred_model1 = model.predict(x_test) 
##print(confusion_matrix(y_test, y_pred_model1))  
##print(classification_report(y_test, y_pred_model1))


# ### Measuring timing with dataset 1 for MLR model:

# In[128]:


total = 0

for i in range(100):

  start = timeit.default_timer()
  y_predict = model.predict(x_test)
  stop = timeit.default_timer()
  total += stop-start

average = total/100

print("Average Runtime: ", end='')
print(average)


# ### Training the model with dataset 2

# In[129]:


model.fit(x2_train, y2_train)


# ### Measuring accuracy with dataset 2:

# In[130]:


print(model.score(x2_train, y2_train))
print(model.score(x2_test, y2_test))


# ## Measuring precision recall dataset 2:

# In[131]:


## can't be done, only score is possible
#y_pred_model1 = classifier.predict(x2_test) 
#print(confusion_matrix(y2_test, y_pred_model1))  
#print(classification_report(y2_test, y_pred_model1))


# ### Measuring timing with dataset 2

# In[132]:


total = 0

for i in range(100):

  start = timeit.default_timer()
  y_predict = model.predict(x2_test)
  stop = timeit.default_timer()
  total += stop-start

average = total/100

print("Average Runtime: ", end='')
print(average)


# # K-NearestNeighbor Regression model:

# ### Creating the model

# In[ ]:


-


# ### Training the model with dataset 1

# In[ ]:


-


# ### Measuring accuracy with dataset 2:

# In[166]:


accuracies = []
for k in range(1, 101):
  model2 = KNeighborsRegressor(n_neighbors = k)
  model2.fit(x2_train, y2_train)
  accuracies.append(model2.score(x2_test, y2_test))

k_list = range(1, 101)
print(accuracies)
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Income predictor accuracy")
plt.show


# In[167]:


print(model2.score(x2_test, y2_test))


# ### Measuring timing with dataset 1

# In[137]:


total = 0

for i in range(100):

  start = timeit.default_timer()
  accuracies = []
  for k in range(1, 101):
    model2 = KNeighborsRegressor(n_neighbors = k)
    model2.fit(x_train, y_train)
    accuracies.append(model2.score(x_test, y_test))
  stop = timeit.default_timer()
  total += stop-start

average = total/100

print("Average Runtime: ", end='')
print(average)


# ## Measuring precision & recall for KNN Classifier:

# In[159]:


y_pred = classifier.predict(x2_test) 
print(confusion_matrix(y2_test, y_pred))  
print(classification_report(y2_test, y_pred))


# ### Training the model with dataset 2

# In[ ]:


-


# ### Measuring accuracy with dataset 2:

# In[139]:


accuracies = []
for k in range(1, 101):
  model2 = KNeighborsRegressor(n_neighbors = k)
  model2.fit(x2_train, y2_train)
  accuracies.append(model2.score(x2_test, y2_test))

k_list = range(1, 101)
print(accuracies)
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Income predictor accuracy")
plt.show


# ### Measuring timing with dataset 2

# In[140]:


total = 0

for i in range(100):

  start = timeit.default_timer()
  accuracies = []
  for k in range(1, 101):
    model2 = KNeighborsRegressor(n_neighbors = k)
    model2.fit(x2_train, y2_train)
    accuracies.append(model2.score(x2_test, y2_test))
  stop = timeit.default_timer()
  total += stop-start

average = total/100

print("Average Runtime: ", end='')
print(average)


# ## measure recall and precision:

# In[122]:


y2_pred = model2.predict(x2_test) 
print(confusion_matrix(y2_test, y2_pred))  
print(classification_report(y2_test, y2_pred))


# # Classifier model 1:

# ### Creating the model

# In[154]:


classifier = KNeighborsClassifier(n_neighbors = 13)


# ### Training the model with dataset 1

# In[155]:


classifier.fit(x2_train, y2_train)


# ### Measuring precision with dataset 1

# In[156]:


print(classifier.score(x2_test, y2_test))


# ### Measuring accuracy with dataset 2:

# In[161]:


accuracies_classifier = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(x2_train, y2_train)
  accuracies_classifier.append(classifier.score(x2_test, y2_test))

k_list = range(1, 101)
print(accuracies_classifier)
plt.plot(k_list, accuracies_classifier)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.title("Income predictor accuracy")
plt.show


# ### Measuring recall with dataset 1

# In[146]:


y_pred_class1 = classifier.predict(x_test) 
print(confusion_matrix(y_test, y_pred_class1))  
print(classification_report(y_test, y_pred))


# ### Measuring timing with dataset 2

# In[157]:


total = 0

for i in range(100):

  start = timeit.default_timer()
  accuracies = []
  for k in range(1, 101):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(x2_train, y2_train)
    accuracies_classifier.append(classifier.score(x2_test, y2_test))
  stop = timeit.default_timer()
  total += stop-start

average = total/100

print("Average Runtime: ", end='')
print(average)


# ### Training the model with dataset 2

# ### Measuring precision with dataset 2

# ### Measuring accuracy with dataset 2:

# ### Measuring recall with dataset 2

# ### Measuring timing with dataset 2

# # Classifier model 2:

# ### Creating the model

# In[102]:


classifier2 = SVC(kernel = 'poly', degree = '2')


# ### Training the model with dataset 1

# In[105]:


classifier2.fit(x_train, y_train)


# ### Measuring precision with dataset 1

# In[106]:


classifier2.score(x_test, y_test)


# ### Measuring accuracy with dataset 1:

# ### Measuring recall with dataset 1

# ### Measuring timing with dataset 1

# ### Training the model with dataset 2

# ### Measuring precision with dataset 2

# ### Measuring accuracy with dataset 2:

# ### Measuring recall with dataset 2

# ### Measuring timing with dataset 2
