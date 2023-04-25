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

Name :P.Hemasonica
Register Number : 212222230048
Feature Transformation - Data_to_Transform.csv


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
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()

```
# Output

Feature Transformation - Data_to_Transform.csv

![image](https://user-images.githubusercontent.com/118361409/234178783-e702fec6-ae97-4514-8419-9e53c9ba63d7.png)

![image](https://user-images.githubusercontent.com/118361409/234178843-b0308ca5-51ef-4275-8c32-ab39d5d0c93e.png)

![image](https://user-images.githubusercontent.com/118361409/234178859-12f72d50-9397-4479-89e7-d982cd19c848.png)

![image](https://user-images.githubusercontent.com/118361409/234178899-8827d06c-8395-4ade-aea4-0c2623d3db64.png)

![image](https://user-images.githubusercontent.com/118361409/234178925-5d8faeb0-1110-4d12-881e-3e3a5ffb18d8.png)


Log Transformation

![image](https://user-images.githubusercontent.com/118361409/234178994-a976fc77-03d4-4ee6-b501-08fb3b02d2bc.png)


Reciprocal Transformation

![image](https://user-images.githubusercontent.com/118361409/234179020-3081c29d-d9f8-4351-bdb6-6299c0aa74ce.png)

SquareRoot Transformation

![image](https://user-images.githubusercontent.com/118361409/234179071-430a9902-eed1-4247-ad88-419297849eb2.png)

Power Transformation

![image](https://user-images.githubusercontent.com/118361409/234179108-ed592042-7a37-4389-9610-27ae58eb50d8.png)

![image](https://user-images.githubusercontent.com/118361409/234179136-a309e951-f13d-4ffe-ab0b-c84d543fc841.png)


Quantile Transformation

![image](https://user-images.githubusercontent.com/118361409/234179167-fd068fef-d1c8-4c51-8b46-4eff21a71f76.png)


# RESULT:

Thus the Feature Transformation for the given datasets had been executed successfully.
