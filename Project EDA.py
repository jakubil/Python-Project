#!/usr/bin/env python
# coding: utf-8

# In[1]:


#EDA IN PYTHON


# In[87]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df1 = pd.read_csv(r"C:\Users\jakubil\Downloads\previous_application.csv")
df1


# In[4]:


df2 = pd.read_csv(r"C:\Users\jakubil\Downloads\application_data.csv")
df2


# In[5]:


result = pd.concat([df1, df2], axis=1, join='inner')
result


# In[6]:


#result.groupby('CODE_GENDER').size().plot(kind='pie',autopct='%1.1f%%');
#filtered_df = df2[df2['CODE_GENDER'] != 'XNA']

#plt.title('Gender Distribution')

#plt.show()


# In[7]:


pd.set_option('display.float_format', lambda x:'%2f' % x)


# In[8]:


df1.info()


# In[9]:


df1.describe()


# In[10]:


# Counting the number of null values
df1.isnull().sum()


# In[11]:


df2.isnull().sum()


# In[12]:


# Count of unique Values
df1.nunique()


# In[13]:


df1.sort_values(by="NAME_CONTRACT_TYPE")


# In[14]:


df1.sort_values(by="NAME_SELLER_INDUSTRY", ascending = False).head()


# In[15]:


df2.sort_values(by="NAME_CONTRACT_TYPE", ascending = False).head()


# In[16]:


df1.duplicated()


# In[17]:


df2.duplicated()


# In[18]:


# Removing null values
df1.dropna(subset=["AMT_ANNUITY"], inplace = True)


# In[19]:


# Removing null values
df2.dropna(subset=["AMT_REQ_CREDIT_BUREAU_DAY"], inplace = True)


# In[20]:


# Removing null values
df2.dropna(subset=["AMT_REQ_CREDIT_BUREAU_WEEK"], inplace = True)


# In[21]:


# Removing null values
df2.dropna(subset=["AMT_REQ_CREDIT_BUREAU_MON"], inplace = True)


# In[22]:


# Removing null values
df2.dropna(subset=["AMT_REQ_CREDIT_BUREAU_QRT"], inplace = True)


# In[23]:


# Removing null values
df2.dropna(subset=["AMT_REQ_CREDIT_BUREAU_YEAR"], inplace = True)


# In[24]:


#Correlation
#Excluding Non numric Column


# In[25]:


numeric_df = df1.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
correlation_matrix



# In[26]:


#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
#correlation_matrix1 = numeric_df.corr().round(3)
#correlation_matrix1


# Now, correlation_matrix will have values rounded to two decimal places
#print(correlation_matrix1)


# In[27]:


import seaborn as sns
import matplotlib.pyplot as plt

# Increase the figure size
plt.figure(figsize=(12, 10))

# Set a larger font scale
sns.set(font_scale=0.6)

# Plot the heatmap with annotations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.3f')

# Show the plot
plt.show()


# In[28]:


#Excluding Non numric Column


# In[29]:


numerics_df = df2.select_dtypes(include=['float64', 'int64'])
correlation2_matrix = numerics_df.corr()
correlation2_matrix


# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt

# Increase the figure size
plt.figure(figsize=(8, 6))

# Set a larger font scale
sns.set(font_scale=0.6)

# Plot the heatmap with annotations
sns.heatmap(correlation2_matrix, annot=True, cmap='coolwarm', fmt='.3f')

# Show the plot
plt.show()


# In[31]:


df1


# In[ ]:





# In[33]:


plt.figure(figsize=(8, 6))
sns.set(font_scale=0.6)
df2.plot()


# In[34]:


plt.figure(figsize=(8, 6))
sns.set(font_scale=0.6)
df1.plot()


# In[35]:


df1


# In[36]:


plt.scatter(df1.AMT_ANNUITY, df1.AMT_GOODS_PRICE)



# In[37]:


plt.scatter(df2.AMT_INCOME_TOTAL, df2.AMT_INCOME_TOTAL, c=df1['AMT_INCOME_TOTAL'], cmap='viridis')
plt.xlabel('AMT_INCOME_TOTAL')
plt.ylabel('AMT_INCOME_TOTAL')
plt.title('Scatter Plot of AMT_INCOME_TOTAL vs. AMT_INCOME_TOTAL')
plt.colorbar()  # Add a colorbar to show the mapping of colors to values
plt.show()


# In[38]:


plt.scatter(df1.AMT_ANNUITY, df1.AMT_GOODS_PRICE, c=df1['AMT_ANNUITY'], cmap='viridis')
plt.xlabel('AMT_ANNUITY')
plt.ylabel('AMT_GOODS_PRICE')
plt.title('Scatter Plot of AMT_ANNUITY vs. AMT_GOODS_PRICE')
plt.colorbar()  # Add a colorbar to show the mapping of colors to values
plt.show()


# In[39]:


df2


# In[40]:


plt.scatter(df2.AMT_CREDIT, df2.AMT_INCOME_TOTAL, c=df2['AMT_CREDIT'], cmap='viridis')
plt.xlabel('AMT_CREDIT')
plt.ylabel('AMT_INCOME_TOTAL')
plt.title('Scatter Plot of AMT_CREDIT vs. AMT_INCOME_TOTAL')
plt.colorbar()  # Add a colorbar to show the mapping of colors to values
plt.show()


# In[ ]:





# In[41]:


plt.scatter(df2.AMT_ANNUITY, df2.AMT_CREDIT, c=df2['AMT_ANNUITY'], cmap='viridis')
plt.xlabel('AMT_ANNUITY')
plt.ylabel('AMT_CREDIT')
plt.title('Scatter Plot of AMT_ANNUITY vs. AMT_CREDIT')
plt.colorbar()  # Add a colorbar to show the mapping of colors to values
plt.show()


# In[42]:


import matplotlib.pyplot as plt
merged_df = pd.merge(df1, df2, on='SK_ID_CURR', how='inner')
plt.scatter(merged_df.AMT_ANNUITY_x, merged_df.AMT_ANNUITY_y, c=merged_df['AMT_ANNUITY_y'], cmap='viridis')
plt.xlabel('AMT_ANNUITY_x')
plt.ylabel('AMT_ANNUITY_y')
plt.title('Scatter Plot of AMT_ANNUITY_x vs. AMT_ANNUITY_y with Color')
plt.colorbar()
plt.show()


# In[43]:


import matplotlib.pyplot as plt
colors = ['blue', 'green', 'red']
filtered_df = df1[df1['NAME_CONTRACT_TYPE'] != 'XNA']
df1.NAME_CONTRACT_TYPE.value_counts(normalize=True).plot.bar(color=colors)
plt.title('Bar Plot of NAME_CONTRACT_TYPE FOR PREVIOUS APPLICATION')
plt.xlabel('Contract Type')
plt.ylabel('Normalized Counts')
plt.show()


# In[44]:


import matplotlib.pyplot as plt
colors = ['orange', 'green', 'black']
filtered_df = df2[df2['NAME_CONTRACT_TYPE'] != 'XNA']
df1.NAME_CONTRACT_TYPE.value_counts(normalize=True).plot.bar(color=colors)
plt.title('Bar Plot of NAME_CONTRACT_TYPE FOR APPLICATION DATASET')
plt.xlabel('Contract Type')
plt.ylabel('Normalized Counts')
plt.show()


# In[45]:


import matplotlib.pyplot as plt

# Assuming you want different colors for each bar
colors = ['green', 'brown', 'red']
filtered_df = df1[df1['NAME_SELLER_INDUSTRY'] != 'XNA']
df1.NAME_SELLER_INDUSTRY.value_counts(normalize=True).plot.bar(color=colors)
plt.title('Bar Plot of NAME_SELLER_INDUSTRY FOR PREVIOUS APPLICATION')
plt.xlabel('Name Seller Industry')
plt.ylabel('Normalized Counts')
plt.show()


# In[46]:


import matplotlib.pyplot as plt
colors = ['grey', 'pink', 'orange']
filtered_df = df1[df1['WEEKDAY_APPR_PROCESS_START'] != 'XNA']
df1.WEEKDAY_APPR_PROCESS_START.value_counts(normalize=True).plot.bar(color=colors)
plt.title('Bar Plot of WEEKDAY_APPR_PROCESS_START FOR PREVIOUS APPLICATION')
plt.xlabel('Week Day')
plt.ylabel('Normalized Counts')
plt.show()


# In[47]:


import matplotlib.pyplot as plt
colors = ['grey', 'orange', 'green', 'red', 'black']
filtered_df = df1[df1['NAME_YIELD_GROUP'] != 'XNA']
df1.NAME_YIELD_GROUP.value_counts(normalize=True).plot.bar(color=colors)
plt.title('Bar Plot of NAME_YIELD_GROUP FOR PREVIOUS APPLICATION')
plt.xlabel('Name Yield Group')
plt.ylabel('Normalized Counts')
plt.show()


# In[48]:


import matplotlib.pyplot as plt
colors = ['pink', 'blue']
filtered_df = df2[df2['CODE_GENDER'] != 'XNA']
df2.CODE_GENDER.value_counts(normalize=True).plot.bar(color=colors)
plt.title('Bar Plot of CODE_GENDER FOR PREVIOUS APPLICATION')
plt.xlabel('Gender')
plt.ylabel('Normalized Counts')
plt.show()


# In[ ]:


df2


# In[49]:


import matplotlib.pyplot as plt

# Filter out 'XNA' values
filtered_df = df2[df2['CODE_GENDER'] != 'XNA']

# Count the occurrences of each gender
gender_counts = filtered_df['CODE_GENDER'].value_counts()

# Create a pie chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['pink', 'orange'])
plt.title('Pie Chart of CODE_GENDER FOR PREVIOUS APPLICATION')
plt.show()


# In[50]:


df2.NAME_EDUCATION_TYPE


# In[51]:


import matplotlib.pyplot as plt
colors = ['grey', 'blue', 'orange', 'pink']
filtered_df = df2[df2['NAME_EDUCATION_TYPE'] != 'XNA']
df2.NAME_EDUCATION_TYPE.value_counts(normalize=True).plot.bar(color=colors)
plt.title('Bar Plot of NAME_EDUCATION_TYPE FOR PREVIOUS APPLICATION')
plt.xlabel('NAME_EDUCATION_TYPE')
plt.ylabel('Normalized Counts')
plt.show()


# In[52]:


import matplotlib.pyplot as plt

# Filter out 'XNA' values
filtered_df = df2[df2['NAME_EDUCATION_TYPE'] != 'XNA']

# Count the occurrences of each gender
NAME_EDUCATION_TYPE = filtered_df['NAME_EDUCATION_TYPE'].value_counts()

# Create a pie chart
plt.pie(NAME_EDUCATION_TYPE, labels=NAME_EDUCATION_TYPE.index, autopct='%1.1f%%', colors=['pink', 'orange','blue', 'green','red'])
plt.title('Pie Chart of NAME_EDUCATION_TYPE FOR PREVIOUS APPLICATION')
plt.show()


# In[53]:


import matplotlib.pyplot as plt

# Filter out 'XNA' values
filtered_df = df2[df2['DAYS_BIRTH'] != 'XNA']

# Convert DAYS_BIRTH to age in years
filtered_df['AGE'] = -filtered_df['DAYS_BIRTH'] // 365

# Define age categories
bins = [0, 17, 63, float('inf')]
labels = ['0-17 Children', '18-63 Youth', '64+ Aged']

# Categorize ages
filtered_df['AGE_CATEGORY'] = pd.cut(filtered_df['AGE'], bins=bins, labels=labels, right=False)

# Count the occurrences of each age category
age_category_counts = filtered_df['AGE_CATEGORY'].value_counts()

# Create a bar plot
age_category_counts.plot.bar(color=['grey', 'blue', 'orange'])
plt.title('Bar Plot of AGE CATEGORIES FOR PREVIOUS APPLICATION')
plt.xlabel('Age Categories')
plt.ylabel('Normalized Counts')
plt.show()


# In[ ]:


df2.DAYS_BIRTH


# In[54]:


import matplotlib.pyplot as plt

# Filter out 'XNA' values
filtered_df = df2[df2['DAYS_BIRTH'] != 'XNA']

# Convert DAYS_BIRTH to age in years
filtered_df['AGE'] = -filtered_df['DAYS_BIRTH'] // 365

# Define age categories
bins = [0, 17, 63, float('inf')]
labels = ['0-17 Children', '18-63 Youth', '64+ Aged']

# Categorize ages
filtered_df['AGE_CATEGORY'] = pd.cut(filtered_df['AGE'], bins=bins, labels=labels, right=False)

# Count the occurrences of each age category
age_category_counts = filtered_df['AGE_CATEGORY'].value_counts()

# Create a pie chart
plt.pie(age_category_counts, labels=age_category_counts.index, autopct='%1.1f%%', startangle=90, colors=['grey', 'blue', 'orange'])
plt.title('Pie Chart of AGE CATEGORIES FOR PREVIOUS APPLICATION')
plt.show()


# In[55]:


import matplotlib.pyplot as plt
colors = ['grey', 'orange', 'green', 'red', 'black']
filtered_df = df1[df1['OCCUPATION_TYPE] != 'XNA']
df1.OCCUPATION_TYPE.value_counts(normalize=True).plot.bar(color=colors)
plt.title('Bar Plot of OCCUPATION_TYPE FOR PREVIOUS APPLICATION')
plt.xlabel('Name Occupation Type')
plt.ylabel('Normalized Counts')
plt.show()


# In[56]:


import matplotlib.pyplot as plt

# Correct the column name in the filter condition
filtered_df = df1[df1['OCCUPATION_TYPE'] != 'XNA']

# Define colors for the bar plot
colors = ['grey', 'orange', 'green', 'red', 'black']

# Plot the bar chart
filtered_df['OCCUPATION_TYPE'].value_counts(normalize=True).plot.bar(color=colors)

# Add labels and title
plt.title('Bar Plot of OCCUPATION_TYPE FOR PREVIOUS APPLICATION')
plt.xlabel('Name Occupation Type')
plt.ylabel('Normalized Counts')

# Show the plot
plt.show()


# In[57]:


sns.boxplot(x=df2['AMT_REQ_CREDIT_BUREAU_YEAR'])


# In[58]:


df2['TARGET'].value_counts()


# In[59]:


# Dividing the dataset into two datasetof target=1(client with payment difficulties) and target=0(all other)
target0_df=df2.loc[df2['TARGET']==0]
target1_df=df2.loc[df2['TARGET']==1]

#Calculate imbalance percentage since majority is target zero and minority is target 1
round(len(target0_df)/len(target1_df),2)


# In[60]:


import matplotlib.pyplot as plt

# Assuming 'result' is your DataFrame
result.groupby('TARGET').size().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['skyblue', 'lightcoral', 'green', 'orange', 'lightyellow']
)

plt.title('Distribution of Payment Difficulties by Client')
plt.show()


# In[61]:


import matplotlib.pyplot as plt

# Assuming 'result' is your DataFrame
result.groupby('NAME_INCOME_TYPE').size().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['skyblue', 'lightcoral', 'green', 'orange', 'lightyellow']
)

plt.title('Income Type')
plt.show()


# In[62]:


import matplotlib.pyplot as plt

# Define income categories
income_bins = [0, 50000, 100000, float('inf')]
income_labels = ['Poor', 'Average', 'Rich']

# Assuming 'df2' is your DataFrame
df2['Income_Category'] = pd.cut(df2['AMT_INCOME_TOTAL'], bins=income_bins, labels=income_labels, right=False)

# Filter out rows where 'AMT_INCOME_TOTAL' is 'XNA'
filtered_df = df2[df2['AMT_INCOME_TOTAL'] != 'XNA']

# Plot the bar chart
colors = ['grey', 'blue', 'orange', 'pink']
filtered_df['Income_Category'].value_counts(normalize=True).sort_index().plot.bar(color=colors)

plt.title('Bar Plot of AMT_INCOME_TOTAL for Previous Application')
plt.xlabel('Income Category')
plt.ylabel('Normalized Counts')
plt.show()


# In[63]:


import matplotlib.pyplot as plt

# Assuming 'result' is your DataFrame
result.groupby('OCCUPATION_TYPE').size().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['skyblue', 'lightcoral', 'green', 'orange', 'lightyellow']
)

plt.title('Occupation Type')
plt.show()


# In[64]:


unique_family_statuses = df2['NAME_FAMILY_STATUS'].unique()
print(unique_family_statuses)


# In[65]:


import matplotlib.pyplot as plt

# Define family status categories
family_status_categories = ['Single', 'Married', 'Civil marriage', 'Separated', 'Widow']

# Assuming 'df2' is your DataFrame
df2['Family_Status_Category'] = pd.Categorical(df2['NAME_FAMILY_STATUS'], categories=family_status_categories, ordered=True)

# Filter out rows where 'NAME_FAMILY_STATUS' is 'XNA'
filtered_df = df2[df2['NAME_FAMILY_STATUS'] != 'XNA']

# Plot the bar chart
colors = ['red', 'brown', 'orange', 'pink', 'green']
filtered_df['Family_Status_Category'].value_counts(normalize=True).sort_index().plot.bar(color=colors)

plt.title('Bar Plot of Marital Status for Previous Application')
plt.xlabel('Marital Status Category')
plt.ylabel('Normalized Counts')
plt.show()


# In[ ]:





# In[66]:


import matplotlib.pyplot as plt
plt.scatter(df2['AMT_INCOME_TOTAL'], df2['DAYS_BIRTH'], c=df2['AMT_ANNUITY'])
plt.xlabel('AMT_INCOME_TOTAL')
plt.ylabel('Days Birth')
plt.colorbar(label='AMT_ANNUITY')
plt.show()


# In[67]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df2' is your DataFrame
plt.figure(figsize=(10, 6))
sns.scatterplot(x='AMT_CREDIT', y='NAME_INCOME_TYPE', data=df2, hue='AMT_CREDIT', palette='viridis')

plt.xlabel('AMT_CREDIT')
plt.ylabel('NAME_INCOME_TYPE')
plt.title('Scatter Plot for AMT_CREDIT and NAME_INCOME_TYPE')
plt.legend(title='AMT_CREDIT', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[ ]:





# In[68]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df2' is your DataFrame
plt.figure(figsize=(12, 8))
sns.histplot(data=df2, x='AMT_CREDIT', hue='OCCUPATION_TYPE', multiple='stack', bins=30, palette='viridis')

plt.xlabel('AMT_CREDIT')
plt.ylabel('Count')
plt.title('Histogram for AMT_CREDIT by OCCUPATION_TYPE')
plt.legend(title='OCCUPATION_TYPE', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[69]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df2' is your DataFrame
plt.figure(figsize=(12, 8))
sns.scatterplot(x='OCCUPATION_TYPE', y='AMT_CREDIT', data=df2, alpha=0.5)

plt.xlabel('OCCUPATION_TYPE')
plt.ylabel('AMT_CREDIT')
plt.title('Scatter Plot for OCCUPATION_TYPE and AMT_CREDIT')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()



# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df2' is your DataFrame
plt.figure(figsize=(12, 8))
sns.scatterplot(x='OCCUPATION_TYPE', y='AMT_CREDIT', data=df2, hue='OCCUPATION_TYPE', palette='viridis', alpha=0.7)

plt.xlabel('OCCUPATION_TYPE')
plt.ylabel('AMT_CREDIT')
plt.title('Scatter Plot for OCCUPATION_TYPE and AMT_CREDIT')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.legend(title='OCCUPATION_TYPE', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[71]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'df2' is your DataFrame
plt.figure(figsize=(12, 8))
sns.scatterplot(x='NAME_HOUSING_TYPE', y='AMT_CREDIT', data=df2, hue='NAME_HOUSING_TYPE', palette='viridis', alpha=0.7)

plt.xlabel('NAME_HOUSING_TYPE')
plt.ylabel('AMT_CREDIT')
plt.title('Scatter Plot for NAME_HOUSING_TYPE and AMT_CREDIT')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.legend(title='NAME_HOUSING_TYPE', loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


# In[72]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='NAME_HOUSING_TYPE', y='AMT_CREDIT', data=df2, palette='viridis')
plt.xlabel('NAME_HOUSING_TYPE')
plt.ylabel('AMT_CREDIT')
plt.title('Boxplot for NAME_HOUSING_TYPE and AMT_CREDIT')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[73]:


plt.figure(figsize=(12, 8))
sns.boxplot(x='CNT_CHILDREN', y='AMT_CREDIT', data=df2, palette='viridis')
plt.xlabel('CNT_CHILDREN')
plt.ylabel('AMT_CREDIT')
plt.title('Boxplot for COUNT OF CHILDREN and AMT_CREDIT')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()


# In[74]:


merged_df1=df1.merge(df2, how="inner", on="SK_ID_CURR")


# In[ ]:





# In[ ]:





# In[ ]:




