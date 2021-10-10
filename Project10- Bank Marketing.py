#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


bank= pd.read_csv(r"C:\Users\shruti\Desktop\Decodr\Project\Decodr Project\Bank marketing project\bank.csv", delimiter=";")


# In[3]:


bank.head()


# In[4]:


bank.tail()


# In[5]:


# Renaming "y" column with "deposit"

bank.rename(columns={"y":"deposit"}, inplace=True)


# In[6]:


bank.head()


# # Data Exploration

# In[7]:


# To get total number of rows 

print("Bank Marketing Dataset contains {rows} rows.".format(rows=len(bank)))


# In[8]:


# To get percentage of missing values in each columns

missing_values= bank.isnull().mean()*100
missing_values.sum()


# ### Categorical Columns Exploration

# In[9]:


cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month','poutcome']

fig, axs = plt.subplots(3, 3, sharex=False, sharey=False, figsize=(10, 8))

counter = 0
for cat_column in cat_columns:
    value_counts = bank[cat_column].value_counts()
    
    trace_x = counter // 3
    trace_y = counter % 3
    x_pos = np.arange(0, len(value_counts))
    
    axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label = value_counts.index)
    
    axs[trace_x, trace_y].set_title(cat_column)
    
    for tick in axs[trace_x, trace_y].get_xticklabels():
        tick.set_rotation(90)
    
    counter += 1

plt.show()


# ### Numerical Columns Exploration

# In[10]:


num_columns = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

fig, axs = plt.subplots(2, 3, sharex=False, sharey=False, figsize=(10, 8))

counter = 0
for num_column in num_columns:
    
    trace_x = counter // 3
    trace_y = counter % 3
    
    axs[trace_x, trace_y].bar(x_pos, value_counts.values, tick_label = value_counts.index)
    axs[trace_x, trace_y].set_title(num_column)
    
    counter += 1

plt.show()


# In[11]:


bank[["pdays", "campaign", "previous"]].describe()


# In[12]:


len(bank[bank["pdays"]> 400])/ len(bank)*100


# In[13]:


len(bank[bank["campaign"]> 34])/ len(bank)*100


# In[14]:


len(bank[bank["previous"]> 34])/ len(bank)*100


# ## Analysis of Categorical columns

# In[15]:


value_counts= bank["deposit"].value_counts()
value_counts.plot.bar(title= "Deposit value counts")


# In[16]:


# Plotting Deposit Vs Jobs

j_bank= pd.DataFrame()

j_bank["yes"]= bank[bank["deposit"] == "yes"]["job"].value_counts()
j_bank["no"]= bank[bank["deposit"] == "no"]["job"].value_counts()

j_bank.plot.bar(title= "Job & Deposit")


# In[17]:


# Plotting Deposit Vs Marital Status

j_bank= pd.DataFrame()

j_bank["yes"]= bank[bank["deposit"] == "yes"]["marital"].value_counts()
j_bank["no"]= bank[bank["deposit"] == "no"]["marital"].value_counts()

j_bank.plot.bar(title= "Marital Status & Deposit")


# In[18]:


# Plotting Deposite Vs Education

j_bank= pd.DataFrame()

j_bank["yes"]= bank[bank["deposit"] == "yes"]["education"].value_counts()
j_bank["no"]= bank[bank["deposit"] == "no"]["education"].value_counts()

j_bank.plot.bar(title= "Education & Deposit")


# In[19]:


# Plotting Deposit Vs Contact

j_bank= pd.DataFrame()

j_bank["yes"]= bank[bank["deposit"] == "yes"]["contact"].value_counts()
j_bank["no"]= bank[bank["deposit"] == "no"]["contact"].value_counts()

j_bank.plot.bar(title= "Contact & Deposit")


# ## Analysis of Numeric columns

# In[20]:


# Balance & Deposit

b_bank= pd.DataFrame()

b_bank['balance_yes'] = (bank[bank['deposit'] == 'yes'][['deposit','balance']].describe())['balance']
b_bank['balance_no'] = (bank[bank['deposit'] == 'no'][['deposit','balance']].describe())['balance']

b_bank


# In[21]:


b_bank.drop(["count", "25%", "50%", "75%"]).plot.bar(title= "Balance & Deposit Statistics")


# In[22]:


# Age & Deposit

b_bank= pd.DataFrame()

b_bank['age_yes'] = (bank[bank['deposit'] == 'yes'][['deposit','age']].describe())['age']
b_bank['age_no'] = (bank[bank['deposit'] == 'no'][['deposit','age']].describe())['age']

b_bank


# In[23]:


b_bank.drop(["count", "25%", "50%", "75%"]).plot.bar(title= "Age & Deposit Statistics")


# In[24]:


# Campaign & Deposit

b_bank= pd.DataFrame()

b_bank['campaign_yes'] = (bank[bank['deposit'] == 'yes'][['deposit','campaign']].describe())['campaign']
b_bank['campaign_no'] = (bank[bank['deposit'] == 'no'][['deposit','campaign']].describe())['campaign']

b_bank


# In[25]:


b_bank.drop(["count", "25%", "50%", "75%"]).plot.bar(title= "Campaign & Deposit Statistics")


# In[26]:


# Previous Campaign & Deposit

b_bank= pd.DataFrame()

b_bank['previous_yes'] = (bank[bank['deposit'] == 'yes'][['deposit','previous']].describe())['previous']
b_bank['previous_no'] = (bank[bank['deposit'] == 'no'][['deposit','previous']].describe())['previous']

b_bank


# In[27]:


b_bank.drop(["count", "25%", "50%", "75%"]).plot.bar(title= "Previous Campaign & Deposit Statistics")


# # Data Cleaning

# In[28]:


def get_dummy_from_bool(row, column_name):
    """Returns 0 if value in column_name is no, returns 1 if value in column_name is yes"""
    return 1 if row[column_name] == "yes" else 0

def get_correct_values(row, column_name, threshold, bank):
    """Returns mean value if value in column_name is above threshold"""
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean= bank[bank[column_name] <= threshold][column_name].mean()
        return mean
    
def clean_data(bank):
    '''
    INPUT
    df - pandas dataframe containing bank marketing campaign dataset
    
    OUTPUT
    df - cleaned dataset:
    1. columns with 'yes' and 'no' values are converted into boolean variables;
    2. categorical columns are converted into dummy variables;
    3. drop irrelevant columns.
    4. impute incorrect values
    '''
    
    cleaned_bank = bank.copy()
    
    # Converting columns containing 'yes' and 'no' values to boolean variables and drop original columns
    
    bool_columns = ['default', 'housing', 'loan', 'deposit']
    for bool_col in bool_columns:
        cleaned_bank[bool_col + '_bool'] = bank.apply(lambda row: get_dummy_from_bool(row, bool_col),axis=1)
    
    cleaned_bank = cleaned_bank.drop(columns = bool_columns)
    
    # Converting categorical columns to dummies
    
    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    
    for col in  cat_columns:
        cleaned_bank = pd.concat([cleaned_bank.drop(col, axis=1),
                                pd.get_dummies(cleaned_bank[col], prefix=col, prefix_sep='_',
                                               drop_first=True, dummy_na=False)], axis=1)
    
    # Dropping irrelevant columns
        
    cleaned_bank = cleaned_bank.drop(columns = ['pdays'])
    
    # Imputing incorrect values and drop original columns
    
    cleaned_bank['campaign_cleaned'] = bank.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_bank),axis=1)
    cleaned_bank['previous_cleaned'] = bank.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_bank),axis=1)
    
    cleaned_bank = cleaned_bank.drop(columns = ['campaign', 'previous'])
    
    return cleaned_bank


# In[29]:


cleaned_bank= clean_data(bank)
cleaned_bank.head()


# # Predicting Campaign Model

# ### Classification Model

# In[30]:


X= cleaned_bank.drop(columns= "deposit_bool")
y= cleaned_bank[["deposit_bool"]]


# In[31]:


TEST_SIZE = 0.3
RAND_STATE= 42


# In[41]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= TEST_SIZE, random_state= RAND_STATE)


# In[42]:


pip install xgboost


# In[43]:


import xgboost
import warnings

xgb = xgboost.XGBClassifier(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)


# In[44]:


xgb.fit(X_train, y_train.squeeze().values)


# In[45]:


y_train_preds= xgb.predict(X_train)
y_test_preds= xgb.predict(X_test)


# In[47]:


from sklearn.metrics import accuracy_score

print("XGB accuracy score for train data : %.3f and for test data : %.3f" % (accuracy_score(y_train, y_train_preds),
                                                                            accuracy_score(y_test, y_test_preds)))


# # Get Feature Importance from Trained Model

# In[50]:


headers= ["name", "score"]
values= sorted(zip(X_train.columns, xgb.feature_importances_), key= lambda x: x[1]*-1)
xgb_feature_importances_=pd.DataFrame(values,columns=headers)
xgb_feature_importances_


# In[52]:


x_pos= np.arange(0, len(xgb_feature_importances_))
plt.figure(figsize=(10,8))
plt.bar(x_pos, xgb_feature_importances_["score"])
plt.xticks(x_pos, xgb_feature_importances_["name"])
plt.xticks(rotation=90)
plt.title("Feature Importance (XGB)")
plt.show()


# In[ ]:




