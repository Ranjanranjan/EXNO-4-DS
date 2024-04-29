# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
NAME : Ranjan K
REG NO : 212222230116
```
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/68546202-9944-4dd0-8ff9-7a125740c562)
```python

data.isnull().sum()
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/8a763b1c-4934-4b06-b1b8-93499675047e)

```python

missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/6044da14-b262-431c-9252-3f7afb3ec45b)

```python

data2=data.dropna(axis=0)
data2
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/39872854-982f-4951-973e-f2cf7041385e)

```python
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/00981010-3ffe-4733-a43a-afdaaf48073d)

```python
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/02c5c17b-2e13-4cd4-b9aa-2dd5f642de1f)

```python


data2
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/8158d5bd-096a-47ea-b2d1-dd19c0fa6c46)

```python
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/9286d10b-777d-42eb-8a9e-f7db18ee1638)

```python

columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/38cf36a0-15da-46b5-85b6-7b20d764cd86)

```python


features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/abfc79c6-c48a-47d5-b665-401146cf09a5)

```python
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/3904a97e-bb22-4511-849d-1e3a50d9958e)

```python

x=new_data[features].values
print(x)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/e76fb327-0c1c-487a-ac4a-c94927c71b7a)

```python

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/3650de34-c070-4543-8f22-46dc86adcc6e)

```python

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/bc8116c5-c407-45f1-85b1-38951911ca36)

```python

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/d875fbf1-d977-4d86-afa3-2881b7f0df76)

```python

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/e5ea2843-bb49-43df-9165-63eeaaa867ef)

```python

data.shape
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/bb30435d-a54b-450f-b79b-e3f6f41e7fc4)

```python

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/50b52323-8b6c-4fbc-b52b-9543d758ce0b)

```python

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/c7a2f4c4-89a2-4128-8a50-66ff54e5ccf4)

```python

tips.time.unique()
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/cfb7f6b0-d971-4e5d-a674-1838503de15c)

```python

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/4a3545e3-ebbf-46e8-956d-6af0a2c22764)

```python

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/Ranjanranjan/EXNO-4-DS/assets/130027697/96c7a6fb-d1b8-4c3d-a9e3-842faaa601f8)

# RESULT:
      Thus, Feature selection and Feature scaling has been used on thegiven dataset.
