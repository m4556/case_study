#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from statistics import mean
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE, RandomOverSampler
import warnings 
warnings.simplefilter(action='ignore')
from xgboost import XGBClassifier

sns.set(style = "whitegrid", palette = "Set2")


# # Load data

# In[2]:


retail_data = pd.read_csv('Retail data.csv', sep=';')
potential_customers = pd.read_csv('Potential customers.csv', sep=';')

info = retail_data.info()


# # Data cleaning
# 
# - Rename column names to make them more intuitive
# - Check for Date format incorrect values and change Date object to pandas Datatime
# - Remove incorect date values or replace them
# - Typecasting: convert object to categorical/numerical
# - Check for missing values, duplicates
# - Add new potential columns
# - Transform some date columns
# - Drop the first column 

# Rename column values to make them more intuitive for visualization

# In[3]:


retail_data = retail_data.rename(columns=str.lower)
potential_customers = potential_customers.rename(columns=str.lower)
retail_data.rename({'martial_status':'marital_status',
           'cust_income':'customer_income',
           'mortgage_yn':'has_mortgage'}, axis=1, inplace=True)

retail_data['gender'] = retail_data.gender.map({'M':'Male', 'F':'Female'})
retail_data['education'] = retail_data.education.map({'HGH':'High School', 'MAS':'Masters', 'BCR':'Bachelor', 'OCR':'Ordinary', 'SEC':'Secondary'  })
retail_data['marital_status'] = retail_data.marital_status.map({'M':'Married', 'S':'Single', 'D':'Divorced','*noval*':'Single'})


# In[4]:


retail_data['current_address_date'] = pd.to_datetime(retail_data.current_address_date, errors = 'coerce')
retail_data['current_job_date'] = pd.to_datetime(retail_data.current_job_date, errors = 'coerce')
retail_data['current_with_bank_date'] = pd.to_datetime(retail_data.current_with_bank_date, errors = 'coerce')
retail_data['marital_status'] = retail_data.marital_status.astype('category')
retail_data['education'] = retail_data.education.astype('category')
retail_data['employment'] = retail_data.employment.astype('category')
retail_data['gender'] = retail_data.gender.astype('category')
retail_data['customer_income'] = retail_data.customer_income.str.replace(',', '.').astype(float)
retail_data['current_balance_eur'] = retail_data.current_balance_eur.str.replace(',', '.').astype(float)

retail_data.age_at_origination.fillna(0, inplace=True)
retail_data['current_job_date'].isna().value_counts() # replcae missing values
retail_data['age_at_origination'].isna().value_counts() 
retail_data['current_address_date'] = retail_data.current_address_date.fillna(pd.Timestamp.min.ceil('D'))
retail_data['current_job_date'] = retail_data.current_job_date.fillna(pd.Timestamp.min.ceil('D'))
retail_data.drop('cocunut',axis=1, inplace=True)


# In[5]:


retail_data.head(20)


# # Exploratory Data Analysis

# Descriptive statistics for attributes separated by morgage status:

# In[6]:


attributes = ['age','years_with_bank', 'marital_status', 'employment','education'
               ,'employment', 'gender', 'customer_income', 'current_balance_eur']
             
customers_with_mortgage = retail_data[retail_data.has_mortgage == 'Y'][attributes]
customers_without_mortgage = retail_data[retail_data.has_mortgage == 'N'][attributes]

pd.concat([customers_with_mortgage.describe(),customers_without_mortgage.describe()], axis=1, keys=['Mortgage','No mortgage'])


# Observation:  
# - Mean for customer income and customer balance are higher for those who have mortgage. 
# - The mean of years_with_bank is higher for those with the status of mortgage.  
# - Mean age is lower for those with positive mortgage status.  
#   

# In[7]:


fig = plt.figure()
mortgage = sns.countplot(x='has_mortgage',data=retail_data)     
Counter(retail_data.has_mortgage.values)


# - Dataset has skrewed distribution.

# ### Observing the variables effect on the target variable
#   
# Average age of people when they got mortgage:

# In[8]:


customers_with_mortgage = retail_data[retail_data.has_mortgage == 'Y']

age = sns.histplot(x='age_at_origination',data=customers_with_mortgage)


# - Age of people when they accepted mortgage is in range 38-40 years.

# Employment 

# In[9]:


employment_plot = sns.countplot(x='employment',data=customers_with_mortgage)


# - The majority of mortgage users are employed(PVE).

# Gender

# In[10]:


gender_plot = sns.countplot(x='gender',data=customers_with_mortgage)


# - Gender may not have any significant impact when it comes to mortgage status.

# Marital status

# In[11]:


gender_plot = sns.countplot(x='marital_status',data=customers_with_mortgage)


# - Married people are more likely to have mortgage.

# Education

# In[12]:


education_plot = sns.countplot(x='education',data=customers_with_mortgage)


# - The majority of mortgage users have Bachelor degreee.

# ### Plotting the target vs distribution of numerical variables: customer income, current_balance_eur

# In[13]:


fig, ax = plt.subplots(2, 2, figsize=(12, 10))

target_uniq = retail_data.has_mortgage.unique()
ax[0, 0].set_title("Distribution for target=" + str(target_uniq[0]))
#ax[0, 0].set_xlim(-5,20)
sns.histplot(data=retail_data[retail_data.has_mortgage == target_uniq[0]],x='customer_income',kde=True,ax=ax[0, 0],color="teal",stat="density",)

target_uniq = retail_data.has_mortgage.unique()
ax[0, 1].set_title("Distribution for target=" + str(target_uniq[1]))
#ax[0, 1].set_xlim(-5,20)
sns.histplot(data=retail_data[retail_data.has_mortgage == target_uniq[1]],x='customer_income',kde=True,ax=ax[0, 1],color="teal",stat="density",)

ax[1, 0].set_title("Boxplot ")
sns.boxplot(data=retail_data, x='has_mortgage', y='customer_income', ax=ax[1, 0])
plot = sns.boxplot(data=retail_data, x='has_mortgage', y='customer_income', ax=ax[1, 1],showfliers=False)


# Confusion matrix:

# In[14]:


fig = plt.figure(figsize=(10, 5))
corr = retail_data.corr()
hm = sns.heatmap(round(corr,2), annot=True,fmt='.2f',linewidths=.05)


# - Other things can further be explored, such as correlations between some variables, exploring/transforming variables related to time format, etc

# # Model building

# The general orientation  maintain rather than redefining the meaning of positive and negative. Binary classification model with positives/negatives is used.
# Dataset is unbalanced, with low number of positives (~1%) and high number of negatives. Some techniques to balance dataset are applied later
# 

#   Scaling numerical features

# In[15]:


numeric_feature_names = ['age', 'years_with_bank','customer_income', 'current_balance_eur']

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(retail_data[numeric_feature_names])
retail_data[numeric_feature_names] = ss.transform(retail_data[numeric_feature_names])


# One hot encoding for categorical features

# In[16]:


retail_data['has_mortgage'] = retail_data['has_mortgage'].map({'Y': 1, 'N': 0})

df_model = pd.get_dummies(retail_data, columns=['marital_status','education','gender','employment'], drop_first= True)
    
X = df_model.drop(['age_at_origination','has_mortgage','current_address_date','current_job_date','current_with_bank_date'],axis=1)
y = retail_data['has_mortgage']


# Some classification models are considered, such as:
# - Logistic Regression
# - Decision trees
# - Random forests
# - SVM
# - Neural networks

# Logistic regression

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

lg = LogisticRegression(solver="newton-cg", random_state=1)
model = lg.fit(X_train, y_train)
y_pred = lg.predict(X_test)
       
print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))
print("Classification Report: \n {}\n".format(classification_report(y_test, y_pred)))
print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred))) 


# - Since dataset is unbalanced, some other tecnhiques such as oversampling are added. Further tuning needeed.  
# - The estimate is changed from accuracy to other measures. 

# Model evaluation:    
# Predicting a customer will 'accept' a mortgage but in reality the customer would not.  
# Predicting a customer will not accept but in reality the customer would have accepted a mortgaga -loss of opportunity. Reducing FN, recall.

# Logistic regression with oversampling

# In[18]:


over = RandomOverSampler(sampling_strategy=1)
X_smote, y_smote = over.fit_resample(X_train, y_train)
model = LogisticRegression().fit(X_smote, y_smote)
y_pred = model.predict(X_test)

print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))
print("Classification Report: \n {}\n".format(classification_report(y_test, y_pred)))


# Random forest with weights

# In[19]:


rf = RandomForestClassifier(n_estimators=10, class_weight='balanced')
model = rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
       
print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))
print("Classification Report: \n {}\n".format(classification_report(y_test, y_pred)))
print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred))) 

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
plot = feat_importances.nlargest(4).plot(kind='barh',grid='False')


# XGBoost

# In[20]:


xgb = XGBClassifier(scale_pos_weight=99,use_label_encoder=False, eval_metric='logloss')
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
model = xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
       
print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))
print("Classification Report: \n {}\n".format(classification_report(y_test, y_pred)))
print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred))) 

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(4).plot(kind='barh')

filename = 'model.sav'
pickle.dump(model, open(filename, 'wb'))

y_pred_prob = xgb.predict_proba(X_test) # output as probabilities.
# The model should be used on potential_customers set. Plot auc and other. Additional analysis. 
# Separate function for accuracy, classification, confusion matrix.
# Visual of confusion matrix. Further investigate models.

