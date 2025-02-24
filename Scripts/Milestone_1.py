#!/usr/bin/env python
# coding: utf-8

# LOADING THE DATA SETS

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


sleep_df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
heart_df = pd.read_csv("heart.csv")
gym_df = pd.read_csv("gym_members_exercise_tracking.csv")


# In[2]:


sleep_df.head()


# In[3]:


heart_df.head()


# In[4]:


gym_df.head()


# Finding Null Values

# In[5]:


sleep_df.isnull().sum()


# In[6]:


heart_df.isnull().sum()


# In[7]:


gym_df.isnull().sum()


# In[8]:


#splitting "blood Pressure" into Systolic and Diastolic
sleep_df[['Systolic_BP', 'Diastolic_BP']] = sleep_df['Blood Pressure'].str.split('/', expand=True)
sleep_df['Systolic_BP'] = pd.to_numeric(sleep_df['Systolic_BP'])
sleep_df['Diastolic_BP'] = pd.to_numeric(sleep_df['Diastolic_BP'])
sleep_df.drop(columns=['Blood Pressure'], inplace=True)


# In[9]:


# Renaming 'sex' to 'gender' in heart_df for consistency
heart_df.rename(columns={'sex': 'gender'}, inplace=True)


# In[10]:


#Checking for negative values
sleep_cols = ['Age','Sleep Duration','Stress Level','Heart Rate','Daily Steps']
heart_cols = ['age','gender','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']
gym_cols = ['Age','Weight (kg)','Height (m)','Max_BPM','Avg_BPM','Resting_BPM','Session_Duration (hours)','Calories_Burned','Fat_Percentage','Water_Intake (liters)','Workout_Frequency (days/week)','BMI']

sleep_neg= sleep_df[sleep_cols] < 0
sleep_count=sleep_neg.sum()
heart_neg= heart_df[heart_cols] < 0
heart_count=heart_neg.sum()
gym_neg= gym_df[gym_cols] < 0
gym_count=gym_neg.sum()

print("Negative values of Sleep:\n",sleep_count)
print("Negative values of Sleep:\n",heart_count)
print("Negative values of Sleep:\n",gym_count)


# OUTLIER DETECTION

# In[11]:


def detect_outliers(df, columns):
    outlier_indices = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        outlier_indices[col] = outliers.tolist()
    return outlier_indices

sleep_outliers = detect_outliers(sleep_df, ['Sleep Duration', 'Heart Rate','Stress Level'])
heart_outliers = detect_outliers(heart_df, ['chol', 'trestbps', 'thalach'])
gym_outliers = detect_outliers(gym_df, ['Calories_Burned', 'BMI'])

print("\nOutliers in Sleep Dataset:")
for col, outlier_data in sleep_outliers.items():
    print(f"{col}: {len(outlier_data)} outliers")

print("\nOutliers in Heart Dataset:")
for col, outlier_data in heart_outliers.items():
    print(f"{col}: {len(outlier_data)} outliers")

print("\nOutliers in Gym Dataset:")
for col, outlier_data in gym_outliers.items():
    print(f"{col}: {len(outlier_data)} outliers")


# Fixing Outliers by Capping method

# In[12]:


def cap_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
      
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    return df

sleep_df = cap_outliers_iqr(sleep_df, ['Heart Rate'])
heart_df = cap_outliers_iqr(heart_df, ['chol', 'trestbps', 'thalach'])
gym_df = cap_outliers_iqr(gym_df, ['Calories_Burned', 'BMI'])


# In[13]:


sleep_df.head()


# In[14]:


#Standardizing column names
sleep_df.rename(columns={'Person ID': 'person_id', 'Gender': 'gender', 'Age': 'age'}, inplace=True)
gym_df.rename(columns={'Age': 'age', 'Gender': 'gender'}, inplace=True)


# In[15]:


#changing gender to 1,0
sleep_df['gender'] = sleep_df['gender'].map({'Male': 1, 'Female': 0})
gym_df['gender'] = gym_df['gender'].map({'Male': 1, 'Female': 0})


# In[16]:


# Converting BMI Category and sleep disorder into numerical encoding
bmi_mapping = {'Underweight': 1, 'Normal': 2, 'Overweight': 3, 'Obese': 4}
sleep_df['BMI'] = sleep_df['BMI Category'].map(bmi_mapping)
sleep_df.drop(columns=['BMI Category'], inplace=True)
sleep_df['sleep_disorder'] = sleep_df['Sleep Disorder'].map(lambda x: 1 if x != 'None' else 0)
sleep_df.drop(columns=['Sleep Disorder'], inplace=True)


# In[17]:


#merging based on Age and Gender
merged_df = heart_df.merge(sleep_df, on=['age', 'gender'], how='inner')
merged_df = merged_df.merge(gym_df, on=['age', 'gender'], how='inner')


# In[18]:


merged_df.shape


# In[19]:


duplicate_rows = merged_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_rows}")



missing_values = merged_df.isnull().sum().sum()

print(f"Number of missing values: {missing_values}")


# In[20]:


# Remove duplicate rows
merged_df = merged_df.drop_duplicates()

# Handle missing values by filling with the median for numerical columns
merged_df = merged_df.fillna(merged_df.median(numeric_only=True))


# In[21]:


merged_df.shape


# Exploratory Data Analysis (EDA)

# In[22]:


merged_df.describe()


# In[23]:


plt.figure(figsize=(8, 5))
sns.countplot(data=merged_df, x='target', palette='coolwarm')
plt.title("Count of Heart Disease Cases")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Heart Disease", "Heart Disease"])
plt.show()


# In[24]:



correlation_columns = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                       'Stress Level', 'Heart Rate', 'Systolic_BP', 'Diastolic_BP', 
                       'Resting_BPM', 'Calories_Burned', 'BMI_x', 'target']

correlation_df = merged_df[correlation_columns].dropna()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[25]:


# balanced data according to the target by down sampling as 1s(Heart Disease)
df_majority = merged_df[merged_df['target'] == 1]  
df_minority = merged_df[merged_df['target'] == 0]  

df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

balanced_df = pd.concat([df_majority_downsampled, df_minority])


# In[26]:


balanced_df.shape


# In[27]:


missing_values = balanced_df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[28]:


summary_statistics = balanced_df.describe()
print("\nSummary Statistics:\n", summary_statistics)


# In[29]:


plt.figure(figsize=(8, 5))
sns.countplot(data=balanced_df, x='target', palette='coolwarm')
plt.title("Balanced Heart Disease Class Distribution (Undersampling)")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Heart Disease", "Heart Disease"])
plt.show()


# In[30]:


plt.figure(figsize=(10, 5))
sns.histplot(balanced_df['Sleep Duration'], bins=30, kde=True, color='blue')
plt.title("Distribution of Sleep Duration (Balanced Dataset)")
plt.xlabel("Sleep Duration (Hours)")
plt.ylabel("Count")
plt.show()


# In[31]:


# Histograms for key variables
balanced_df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Calories_Burned']].hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=14)
plt.show()


# In[32]:


# Scatter plot: Sleep Duration vs Calories Burned with Stress Level hue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=balanced_df, x='Sleep Duration', y='Calories_Burned', hue='Stress Level', palette='coolwarm')
plt.title('Relationship between Sleep Duration, Calories Burned, and Stress Level')
plt.xlabel('Sleep Duration')
plt.ylabel('Calories Burned')
plt.show()


# In[33]:


# Scatter plot: Sleep Duration vs Heart Rate with Stress Level hue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=balanced_df, x='Sleep Duration', y='Heart Rate', hue='Stress Level', palette='coolwarm')
plt.title('Relationship between Sleep Duration, Heart Rate, and Stress Level')
plt.xlabel('Sleep Duration')
plt.ylabel('Heart Rate')
plt.show()


# In[34]:


# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(balanced_df[['Sleep Duration', 'Stress Level', 'Heart Rate', 'Daily Steps', 'Calories_Burned']].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap for Key Features')
plt.show()


# In[35]:


#finding correlation between Sleep Duration,Stress level, and Heart Disease Risk
correlation_columns = ['Sleep Duration', 'Stress Level', 'target']

correlation_matrix = balanced_df[correlation_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Sleep Duration, Stress Level, and Heart Disease Risk (Balanced Dataset)")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




