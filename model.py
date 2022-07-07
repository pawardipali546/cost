import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df= pd.read_csv(r"C:/Users/acer/Desktop/dipali/edx_courses.csv",encoding = 'unicode_escape')
df=df.drop(['title','summary','instructors','subtitles'], axis=1)
df.describe()
df.info()
df.columns

df.isnull()
df.isnull().sum()
df.drop("n_enrolled",axis=1,inplace=True)
#dummies
df["course_type"]=pd.get_dummies(df.course_type,drop_first=True)

#labeleccoding
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df["institution"]=lb.fit_transform(df.institution)
df["Level"]=lb.fit_transform(df.Level)
df["language"]=lb.fit_transform(df.language)
df["subject"]=lb.fit_transform(df.subject)
#now defining the predictors and the target columns and doing the train_test split


predictors = df.loc[:, df.columns!="price"]
type(predictors)

target = df["price"]
type(target)
df_new.columns
# Train Test partition of the data and perfoming the adaboost regressor as it has given best result in automl by pycaret
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.20,random_state=2)

from sklearn.ensemble import AdaBoostRegressor as AR
from sklearn import metrics
regressor=AR(base_estimator=None,learning_rate=1.0,loss="linear",n_estimators=100,random_state=2)

regressor.fit(x_train,y_train)
#predicting a new result
y_pred=regressor.predict(x_test)
## accuracy score
from sklearn import metrics
r_square=metrics.r2_score(y_test, y_pred)
print(r_square)
mean_squared_log_error=metrics.mean_squared_log_error(y_test, y_pred)
print(mean_squared_log_error)


#plotting the actual price and the predicted price
plt.plot(y_test,color="blue",label="actual_price")
plt.plot(y_pred,color="red",label="predicted_price")
plt.title("Actual_price vs Predicted_price")
plt.xlabel("values")
plt.ylabel("price")
plt.legend()
plt.show()
#save the model_ar to the disk
filename="model_ar.pkl"
pickle.dump(regressor,open(filename,"wb"))
model_ar=pickle.load(open("model_ar.pkl","rb"))

