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


# ## Exploratory Data Analysis (EDA)

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


# balanced data according to the target by down sampling as 1s(Heart Disease)
df_majority = merged_df[merged_df['target'] == 1]  
df_minority = merged_df[merged_df['target'] == 0]  

df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

balanced_df = pd.concat([df_majority_downsampled, df_minority])
df = balanced_df


# In[25]:


balanced_df.shape


# In[26]:


df.head()


# In[27]:



correlation_columns = ['Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                       'Stress Level', 'Heart Rate', 'Systolic_BP', 'Diastolic_BP', 
                       'Resting_BPM', 'Calories_Burned', 'BMI_x', 'target']

correlation_df = merged_df[correlation_columns].dropna()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[28]:


missing_values = balanced_df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[29]:


summary_statistics = balanced_df.describe()
print("\nSummary Statistics:\n", summary_statistics)


# In[30]:


plt.figure(figsize=(8, 5))
sns.countplot(data=balanced_df, x='target', palette='coolwarm')
plt.title("Balanced Heart Disease Class Distribution (Undersampling)")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.xticks([0, 1], ["No Heart Disease", "Heart Disease"])
plt.show()


# In[ ]:





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


# # MILESTONE_2

# In[36]:


print(df.columns)


# In[37]:


df.head()


# ## Feature Engineering - Feature Creation

# In[38]:


# Added two cross-dataset features

# 1. Heart Resilience Score
df['heart_resilience_score'] = (
    (df['thalach'] / df['Heart Rate']) * (df['Daily Steps'] / 1000)
)

# 2. Lifestyle Recovery Index
df['lifestyle_recovery_index'] = (
    (df['Sleep Duration'] * df['Calories_Burned']) / (df['Stress Level'] + 1)
)

# Created project-specific risk label
df['heart_risk'] = (
    (df['Stress Level'] > 7) |
    (df['Sleep Duration'] < 6) |
    (df['Physical Activity Level'] < 3) |
    (df['Calories_Burned'] < 200) |
    (df['Water_Intake (liters)'] < 2.0) |
    (df['Heart Rate'] > 100) |
    (df['Systolic_BP'] > 140) |
    (df['chol'] > 240)
).astype(int)


# In[39]:


df[['heart_resilience_score', 'lifestyle_recovery_index', 'heart_risk']].head()


# ## Feature Engineering - Categorical Variable Encoding

# In[40]:


# Encoded 'experience_level' (ordinal)
# Assuming: 1 = Beginner, 2 = Intermediate, 3 = Advanced
#gender already encoded in Milestone 1
experience_mapping = {1: 'Beginner', 2: 'Intermediate', 3: 'Advanced'}
df['experience_level_label'] = df['Experience_Level'].map(experience_mapping)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['experience_level_encoded'] = le.fit_transform(df['experience_level_label'])

# One-hot encode 'workout_type' (nominal)
df = pd.get_dummies(df, columns=['Workout_Type'], drop_first=True)


print(df[['Experience_Level', 'experience_level_label', 'experience_level_encoded']].head())
print("\nWorkout type columns:", [col for col in df.columns if col.startswith("Workout_Type_")])


# In[41]:


feature_candidates = [
    'age', 'Sleep Duration', 'Quality of Sleep', 'Stress Level',
    'Physical Activity Level', 'Calories_Burned', 'Water_Intake (liters)',
    'Workout_Frequency (days/week)', 'Heart Rate', 'Systolic_BP',
    'Diastolic_BP', 'chol', 'trestbps', 'thalach', 'BMI_y',
    'heart_resilience_score', 'lifestyle_recovery_index',
    'gender', 'experience_level_encoded'
]

X = df[feature_candidates]
y = df['heart_risk']


# In[42]:


# Correlation matrix of numeric features
plt.figure(figsize=(14, 10))
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# ## Feature Selection - Feature Importance Evaluation

# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define feature list (all features including engineered ones)
full_features = [
    'chol', 'trestbps', 'Water_Intake (liters)', 'age', 'thalach',
    'heart_resilience_score', 'Systolic_BP', 'Diastolic_BP',
    'Sleep Duration', 'gender', 'Stress Level', 'Physical Activity Level',
    'Quality of Sleep', 'Calories_Burned', 'lifestyle_recovery_index',
    'Heart Rate', 'BMI_y', 'Workout_Frequency (days/week)', 'experience_level_encoded'
]


X_full = df[full_features]  
y = df['heart_risk']        


X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)


feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X_full.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(10, 7))
sns.set(style="whitegrid")
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='pastel')
plt.title('Feature Importance (Random Forest - Descending)', fontsize=14)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# In[44]:


sns.set(style="white")

correlation_matrix = df[full_features + ['heart_risk']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,         
    fmt=".2f",           
    cmap="coolwarm",     
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.75}
)

plt.title("Correlation Heatmap of Features and Target (heart_risk)", fontsize=14)
plt.tight_layout()
plt.show()


# In[45]:


#selecting all important features for training
project_features = importance_df['Feature'].tolist()

x_project = df[project_features]


# ## Data Modeling - Data Splitting

# In[46]:


from sklearn.model_selection import train_test_split
X_project = df[project_features]
y_project = df['heart_risk']

# splitting the dataset into 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X_project, y_project, test_size=0.2, stratify=y_project, random_state=42
)


# ### Training on three different models(with all features)

# In[47]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Scaling features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)


#Training 3 different models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "SVM": SVC(probability=True, class_weight='balanced')
}


# ### Model Evaluation and Comparison(all features)

# In[48]:


results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    })

# model performance when using all features 
results_df = pd.DataFrame(results)
print("Model Evaluation Results(all features):")
print(results_df)


# ### Testing the model(all features)

# In[49]:


import pandas as pd

# Define full feature input profile
test_input_full = {
    "chol": 210,
    "trestbps": 125,
    "Water_Intake (liters)": 2.5,
    "thalach": 165,
    "age": 45,
    "Sleep Duration": 7,
    "heart_resilience_score": (165 / 72) * (7000 / 1000),
    "Diastolic_BP": 82,
    "Systolic_BP": 130,
    "Stress Level": 5,
    "gender": 1,
    "Physical Activity Level": 4,
    "Heart Rate": 72,
    "lifestyle_recovery_index": (7 * 450) / (5 + 1),
    "Calories_Burned": 450,
    "BMI_y": 24,
    "Quality of Sleep": 4,
    "Workout_Frequency (days/week)": 3,
    "experience_level_encoded": 1
}

# Feature order used during training
project_features = [
    'chol', 'trestbps', 'Water_Intake (liters)', 'thalach', 'age',
    'Sleep Duration', 'heart_resilience_score', 'Diastolic_BP',
    'Systolic_BP', 'Stress Level', 'gender', 'Physical Activity Level',
    'Heart Rate', 'lifestyle_recovery_index', 'Calories_Burned', 'BMI_y',
    'Quality of Sleep', 'Workout_Frequency (days/week)', 'experience_level_encoded'
]

# Test function for all-features model
def predict_full_model(input_dict, model, scaler, feature_order):
    X_input = pd.DataFrame([[input_dict[feat] for feat in feature_order]], columns=feature_order)
    X_scaled = pd.DataFrame(
        scaler.transform(X_input),
        columns=X_input.columns,
        index=X_input.index
    )
    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0][1]
    print("Predicted Class:", pred_class)
    print("Probability of Heart Risk:", round(pred_proba * 100, 2), "%")

# Run prediction (using Random Forest from models dictionary)
predict_full_model(test_input_full, models["Random Forest"], scaler, project_features)


# ### Training three models with lifestyle specific features

# In[50]:


lifestyle_features = [
    'Sleep Duration', 'Quality of Sleep', 'Stress Level',
    'Physical Activity Level', 'Calories_Burned', 'Water_Intake (liters)',
    'Workout_Frequency (days/week)', 'experience_level_encoded',
    'Heart Rate', 'lifestyle_recovery_index', 'heart_resilience_score',
    'BMI_y', 'gender', 'age'
]

X_lifestyle = df[lifestyle_features]
y_lifestyle = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X_lifestyle, y_lifestyle, test_size=0.2, random_state=42, stratify=y_lifestyle
)

scaler1 = StandardScaler()
X_train_scaled = pd.DataFrame(scaler1.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler1.transform(X_test), columns=X_test.columns, index=X_test.index)

models1 = {
    "Logistic Regression1": LogisticRegression(max_iter=1000),
    "Random Forest1": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM1": SVC(kernel='rbf', probability=True, random_state=42)
}


# ### Model Evaluation and Comparison(lifestyle features)

# In[51]:



results = []

for name, model in models1.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    })
### model performance when using lifestyle specific features 
results_lifestyle_df = pd.DataFrame(results)
print("Model Evaluation of Lifestyle features:")
print(results_lifestyle_df)


# ### Testing the Model(lifestyle features)

# In[52]:


import pandas as pd

#Define input profile (lifestyle-focused only)
test_input_lifestyle = {
    "Sleep Duration": 7,
    "Quality of Sleep": 4,
    "Stress Level": 5,
    "Physical Activity Level": 4,
    "Calories_Burned": 450,
    "Water_Intake (liters)": 2.5,
    "Workout_Frequency (days/week)": 3,
    "experience_level_encoded": 1,
    "Heart Rate": 72,
    "lifestyle_recovery_index": (7 * 450) / (5 + 1),
    "heart_resilience_score": (165 / 72) * (7000 / 1000),
    "BMI_y": 24,
    "gender": 1,
    "age": 45
}

# Lifestyle features used for training
lifestyle_features = [
    'Sleep Duration', 'Quality of Sleep', 'Stress Level',
    'Physical Activity Level', 'Calories_Burned', 'Water_Intake (liters)',
    'Workout_Frequency (days/week)', 'experience_level_encoded',
    'Heart Rate', 'lifestyle_recovery_index', 'heart_resilience_score',
    'BMI_y', 'gender', 'age'
]

# Prediction function for lifestyle model
def predict_lifestyle_model(input_dict, model, scaler, feature_order):
    X_input = pd.DataFrame([[input_dict[feat] for feat in feature_order]], columns=feature_order)
    X_scaled = pd.DataFrame(
        scaler1.transform(X_input),
        columns=X_input.columns,
        index=X_input.index
    )
    pred_class = model.predict(X_scaled)[0]
    pred_proba = model.predict_proba(X_scaled)[0][1]
    print("Predicted Class:", pred_class)
    print("Probability of Heart Risk:", round(pred_proba * 100, 2), "%")

# Run prediction (using Random Forest model)
predict_lifestyle_model(test_input_lifestyle, models1["Random Forest1"], scaler1, lifestyle_features)


# In[53]:


full_model_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "SVM"],
    "Accuracy": [0.942879, 0.999218, 0.985133],
    "Precision": [0.952452, 0.998536, 0.991111],
    "Recall": [0.939883, 1.000000, 0.980938],
    "F1 Score": [0.946125, 0.999267, 0.985999]
})

lifestyle_model_results = pd.DataFrame({
    "Model": ["Logistic Regression1", "Random Forest1", "SVM1"],
    "Accuracy": [0.803599, 0.979656, 0.852113],
    "Precision": [0.808917, 0.985737, 0.859425],
    "Recall": [0.794992, 0.973396, 0.841941],
    "F1 Score": [0.801894, 0.979528, 0.850593]
})


def plot_comparison(df, title):
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
    for i, metric in enumerate(metrics):
        axes[i].bar(df["Model"], df[metric], color='skyblue')
        axes[i].set_title(metric)
        axes[i].set_ylim(0.7, 1.05)
        axes[i].set_ylabel("Score")
        axes[i].tick_params(axis='x', rotation=20)
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


plot_comparison(full_model_results, "Full Feature Model - Performance Comparison")
plot_comparison(lifestyle_model_results, "Lifestyle-Only Model - Performance Comparison")


# In[54]:


import os
import joblib

# Created folders if not present
os.makedirs("app/model", exist_ok=True)

# Saved your final full-featured Random Forest model
joblib.dump(models["Random Forest"], "app/model/rf_model.pkl")

# Saved the associated scaler
joblib.dump(scaler, "app/model/scaler.pkl")



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




