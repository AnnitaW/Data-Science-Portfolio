#!/usr/bin/env python
# coding: utf-8

# <a href="https://www.bigdatauniversity.com"><img src="https://ibm.box.com/shared/static/cw2c7r3o20w9zn8gkecaeyjhgw3xdgbj.png" width="400" align="center"></a>
# 
# <h1 align="center"><font size="5">Classification with Python</font></h1>

# In this notebook we try to practice all the classification algorithms that we learned in this course.
# 
# We load a dataset using Pandas library, and apply the following algorithms, and find the best one for this specific dataset by accuracy evaluation methods.
# 
# Lets first load required libraries:

# In[1]:


import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# ### About dataset

# This dataset is about past loans. The __Loan_train.csv__ data set includes details of 346 customers whose loan are already paid off or defaulted. It includes following fields:
# 
# | Field          | Description                                                                           |
# |----------------|---------------------------------------------------------------------------------------|
# | Loan_status    | Whether a loan is paid off on in collection                                           |
# | Principal      | Basic principal loan amount at the                                                    |
# | Terms          | Origination terms which can be weekly (7 days), biweekly, and monthly payoff schedule |
# | Effective_date | When the loan got originated and took effects                                         |
# | Due_date       | Since it’s one-time payoff schedule, each loan has one single due date                |
# | Age            | Age of applicant                                                                      |
# | Education      | Education of applicant                                                                |
# | Gender         | The gender of applicant                                                               |

# Lets download the dataset

# In[2]:


get_ipython().system('wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv')


# ### Load Data From CSV File  

# In[3]:


df = pd.read_csv('loan_train.csv')
df.head()


# In[4]:


df.shape


# ### Convert to date time object 

# In[5]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# # Data visualization and pre-processing
# 
# 

# Let’s see how many of each class is in our data set 

# In[6]:


df['loan_status'].value_counts()


# 260 people have paid off the loan on time while 86 have gone into collection 
# 

# Lets plot some columns to underestand data better:

# In[7]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[8]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[9]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# # Pre-processing:  Feature selection/extraction

# ### Lets look at the day of the week people get the loan 

# In[10]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# We see that people who get the loan at the end of the week dont pay it off, so lets use Feature binarization to set a threshold values less then day 4 

# In[11]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# ## Convert Categorical features to numerical values

# Lets look at gender:

# In[12]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# 86 % of female pay there loans while only 73 % of males pay there loan
# 

# Lets convert male to 0 and female to 1:
# 

# In[13]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# ## One Hot Encoding  
# #### How about education?

# In[14]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# #### Feature befor One Hot Encoding

# In[15]:


df[['Principal','terms','age','Gender','education']].head()


# #### Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame 

# In[16]:


Feature = df[['Principal','terms','age','Gender','weekend','dayofweek']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


# ### Feature selection

# Lets defind feature sets, X:

# In[17]:


X = Feature
X[0:5]


# What are our lables?

# In[18]:


#y = df['loan_status'].values
y = df['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1]).values
y[0:5]


# ## Normalize Data 

# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# In[19]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[20]:


## training data
X_train, y_train=X, y


# In[21]:


X_train.shape, y_train.shape


# In[22]:


#Dictionary to keep score for the calculation
scoreTable=dict()


# In[23]:


###import scoring methods
from sklearn.neighbors import KNeighborsClassifier
#from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# # Classification 

# Now, it is your turn, use the training set to build an accurate model. Then use the test set to report the accuracy of the model
# You should use the following algorithm:
# - K Nearest Neighbor(KNN)
# - Decision Tree
# - Support Vector Machine
# - Logistic Regression
# 
# 
# 
# __ Notice:__ 
# - You can go above and change the pre-processing, feature selection, feature-extraction, and so on, to make a better model.
# - You should use either scikit-learn, Scipy or Numpy libraries for developing the classification algorithms.
# - You should include the code of the algorithm in the following cells.

# # K Nearest Neighbor(KNN)
# Notice: You should find the best k to build the model with the best accuracy.  
# **warning:** You should not use the __loan_test.csv__ for finding the best k, however, you can split your train_loan.csv into train and test to find the best __k__.

# In[24]:


bestscore=0.0
scorelist=[]

for k in range(2,10):
    kneigh=KNeighborsClassifier(n_neighbors=k,algorithm='auto')

#validation of the best K
    scores=cross_val_score(kneigh,X,y,cv=10)
    score=scores.mean()
    scorelist.append(score)

    if score>bestscore:
        bestscore=score
        best_kneigh=kneigh
        bestk=k

print('The best K is:', bestk, "Validation accuracy:", bestscore)
kneigh=best_kneigh


# In[25]:


kneigh.fit(X_train,y_train)
y_pred=kneigh.predict(X_train)
print(X_train[0:5],y_train[0:5],y_pred[0:5])


# In[26]:


#writing the scores into a list
scoreTable['knn_jaccard']=jaccard_similarity_score(y_train, y_pred)
scoreTable['knn_f1-score']=f1_score(y_train, y_pred, average='weighted')


# In[27]:


scoreTable


# # Decision Tree

# In[28]:


from sklearn.tree import DecisionTreeClassifier

loanTree=DecisionTreeClassifier()
loanTree=loanTree.fit(X_train, y_train) #this shows the default of the parameter
y_pred=loanTree.predict(X_train)


# In[29]:


scoreTable['DT_jaccard']=jaccard_similarity_score(y_train, y_pred)
scoreTable['DT_f1-score']=f1_score(y_train, y_pred, average='weighted')


# In[30]:


scoreTable


# # Support Vector Machine

# In[31]:


##evaluation  to check the accuracy of the model
from sklearn import svm


# In[32]:


svmdata=svm.LinearSVC(random_state=7)
svmdata.fit(X_train, y_train)
y_pred=svmdata.predict(X_train)


# In[33]:


scoreTable['SVM_jaccard']=jaccard_similarity_score(y_train, y_pred)
scoreTable['SVM_f1-score']=f1_score(y_train, y_pred, average='weighted')


# In[34]:


scoreTable


# # Logistic Regression

# In[35]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(random_state=0, solver='lbfgs',
                         multi_class='multinomial')
LR.fit(X_train, y_train)
y_pred=LR.predict(X_train)
y_proba=LR.predict_proba(X_train)


# In[36]:


scoreTable['LR-jaccard']=jaccard_similarity_score(y_train, y_pred)
scoreTable['LR_f1-score']=f1_score(y_train, y_pred, average='weighted') 
scoreTable['LR-logLoss']=log_loss(y_train, y_proba)
 


# In[37]:


scoreTable


# # Model Evaluation using Test set

# In[38]:


from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# First, download and load the test set:

# In[39]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# ### Load Test set for evaluation 

# In[40]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()


# In[41]:


test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek           ###adding an extra column 
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)

test_df1=test_df.copy()
test_df1['Gender'].replace(to_replace=['male','female'], value=[0,1], inplace=True)
#print(test_df1.head())

Feature1 = test_df1[['Principal','terms','age','Gender','weekend','dayofweek']]
Feature1 = pd.concat([Feature1,pd.get_dummies(test_df1['education'])], axis=1)
#print(len(test_df1[test_df1.education=='Master or Above']))
Feature1.drop(['Master or Above'], axis = 1,inplace=True)
#print(Feature1.head())
#print(Feature.head())

X1 = Feature1

#test_y = test_df1['loan_status'].values
#print(test_y)
y = test_df1['loan_status'].replace(to_replace=['PAIDOFF','COLLECTION'], value=[0,1]).values
test_y=y.astype(float)
test_X= preprocessing.StandardScaler().fit_transform(X1)


# In[42]:


scoreTable_test=dict()  ##dictionary to store values


# In[43]:


##Prediction for KNN neighbor

y_LR_pred=LR.predict(test_X)
y_LR_proba=LR.predict_proba(test_X)
scoreTable_test['LR-jaccard']=jaccard_similarity_score(test_y, y_LR_pred)
scoreTable_test['LR_f1-score']=f1_score(test_y, y_LR_pred, average='weighted') 
scoreTable_test['LR-logLoss']=log_loss(test_y, y_LR_proba)

y_svm_pred=svmdata.predict(test_X)
scoreTable_test['SVM_jaccard']=jaccard_similarity_score(test_y, y_svm_pred)
scoreTable_test['SVM_f1-score']=f1_score(test_y, y_svm_pred, average='weighted')

y_loanTree_pred=loanTree.predict(test_X)
scoreTable_test['DT_jaccard']=jaccard_similarity_score(test_y, y_loanTree_pred)
scoreTable_test['DT_f1-score']=f1_score(test_y, y_loanTree_pred, average='weighted')

y_knn_pred=kneigh.predict(test_X)
scoreTable_test['knn_jaccard']=jaccard_similarity_score(test_y, y_knn_pred)
scoreTable_test['knn_f1-score']=f1_score(test_y, y_knn_pred, average='weighted')


# In[44]:


scoreTable_test


# In[ ]:





# # Report
# You should be able to report the accuracy of the built model using different evaluation metrics:

# | Algorithm          | Jaccard | F1-score | LogLoss |
# |--------------------|---------|----------|---------|
# | KNN                | 0.7407      |0.7144       | NA      |
# | Decision Tree      | 0.7592       | 0.761        | NA      |
# | SVM                | 0.7592       | 0.695        | NA      |
# | LogisticRegression | 0.7777       | 0.7089        | 0.4739       |

# <h2>Want to learn more?</h2>
# 
# IBM SPSS Modeler is a comprehensive analytics platform that has many machine learning algorithms. It has been designed to bring predictive intelligence to decisions made by individuals, by groups, by systems – by your enterprise as a whole. A free trial is available through this course, available here: <a href="http://cocl.us/ML0101EN-SPSSModeler">SPSS Modeler</a>
# 
# Also, you can use Watson Studio to run these notebooks faster with bigger datasets. Watson Studio is IBM's leading cloud solution for data scientists, built by data scientists. With Jupyter notebooks, RStudio, Apache Spark and popular libraries pre-packaged in the cloud, Watson Studio enables data scientists to collaborate on their projects without having to install anything. Join the fast-growing community of Watson Studio users today with a free account at <a href="https://cocl.us/ML0101EN_DSX">Watson Studio</a>
# 
# <h3>Thanks for completing this lesson!</h3>
# 
# <h4>Author:  <a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a></h4>
# <p><a href="https://ca.linkedin.com/in/saeedaghabozorgi">Saeed Aghabozorgi</a>, PhD is a Data Scientist in IBM with a track record of developing enterprise level applications that substantially increases clients’ ability to turn data into actionable knowledge. He is a researcher in data mining field and expert in developing advanced analytic methods like machine learning and statistical modelling on large datasets.</p>
# 
# <hr>
# 
# <p>Copyright &copy; 2018 <a href="https://cocl.us/DX0108EN_CC">Cognitive Class</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/">MIT License</a>.</p>
