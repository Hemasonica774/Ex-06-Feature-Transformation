# Ex-06-Feature-Transformation
# AIM:

To read the given data and perform Feature Transformation process and save the data to a file.'

# EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM:
# STEP 1

Read the given Data

# STEP 2

Clean the Data Set using Data Cleaning 

# STEP 3

Apply Feature Transformation techniques to all the features of the data set

# STEP 4

Save the data to the file

# CODE:

```

NAME:P.Hemasonica
Register number:22003246


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()

df1 = df.copy()
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df1['Highly Positive Skew'] = np.log(df1['Highly Positive Skew'])
sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')
plt.show()

df2 = df.copy()
df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df3 = df.copy()
df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)
sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')
plt.show()

df4 = df.copy()
df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])
sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_1'],line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))
sm.qqplot(df5['Moderate Negative Skew_2'],line='45')
plt.show()


```
# Output
## Data

![image](https://user-images.githubusercontent.com/118361409/236668907-096722f7-16b4-4c08-a531-dede6e291196.png)

![image](https://user-images.githubusercontent.com/118361409/236668917-7fd07aac-e8ba-42a1-aa07-584dda964f1f.png)

![image](https://user-images.githubusercontent.com/118361409/236668925-77d2564c-afb1-4d48-bade-d9ab9266429f.png)

![image](https://user-images.githubusercontent.com/118361409/236668947-fdbac777-3115-4803-8163-51c78a9f0f28.png)

![image](https://user-images.githubusercontent.com/118361409/236668953-3717b304-a70d-4ae8-a677-c1c1e58d823f.png)

## Before Transformation

![image](https://user-images.githubusercontent.com/118361409/236669102-332dd1d6-ddb5-48b5-ad58-1f0ceea66f78.png)

![image](https://user-images.githubusercontent.com/118361409/236669111-4455a75c-7fdd-4718-ac45-61d973374556.png)

![image](https://user-images.githubusercontent.com/118361409/236669126-5d91e7b1-eda6-40fe-ad44-b1c58eccf24e.png)



## Log Transformation

![image](https://user-images.githubusercontent.com/118361409/236669248-c7b29f19-72e2-4975-8114-944222e22839.png)

## Reciprocal Transformation

![image](https://user-images.githubusercontent.com/118361409/236669266-49b0bcad-72d2-4fe1-82e6-b2403366a4f2.png)

## Square root Transformation

![image](https://user-images.githubusercontent.com/118361409/236669330-f10b37bb-ebc2-4a23-89bf-10cd9132bac7.png)

![image](https://user-images.githubusercontent.com/118361409/236669343-f87ce5e6-4cbc-448f-be8f-45c05f9f700f.png)

## Power Transformation

![image](https://user-images.githubusercontent.com/118361409/236669358-95eb58a6-5ad8-44b4-8d37-86d99002e110.png)

##Quantile Transformation

![image](https://user-images.githubusercontent.com/118361409/236669405-b15a19fa-1b2c-465f-85ac-aa2da5150f22.png)

# RESULT:

Thus the Feature Transformation for the given datasets had been executed successfully.
