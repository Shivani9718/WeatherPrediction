#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


weather = pd.read_csv("local_weather.csv", index_col="date")
weather


# In[ ]:


from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()
weather['condition']=lc.fit_transform(weather['condition'])


# In[17]:


weather


# In[18]:


lc=LabelEncoder()
weather['wdirect']=lc.fit_transform(weather['wdirect'])


# In[19]:


weather


# In[20]:


weather.apply(pd.isnull).sum()


# In[21]:


core_weather = weather[["condition", "fog", "humidity", "pressure","rain","temp","thunder","wdegree","wdirect"]].copy()
core_weather.columns = ["condition", "fog", "humidity", "pressure","rain","temp","thunder","wdegree","wdirect"]


# In[22]:


core_weather


# In[23]:


weather.shape


# In[24]:


core_weather["temp"] = core_weather["temp"].fillna(method="ffill")


# In[25]:


core_weather["humidity"] = core_weather["humidity"].fillna(method="ffill")


# In[26]:


core_weather["pressure"] = core_weather["pressure"].fillna(method="ffill")


# In[33]:


core_weather["temp"] = core_weather["temp"].fillna(method="ffill")


# In[34]:


core_weather["wdegree"] = core_weather["wdegree"].fillna(method="ffill")


# In[35]:


core_weather.apply(pd.isnull).sum()


# In[36]:


core_weather.index


# In[37]:


core_weather[["temp"]].plot()


# In[38]:


core_weather[["pressure"]].plot()


# In[39]:


core_weather.plot(subplots=True, figsize=(25,20))


# In[40]:


core_weather.hist(bins=10,figsize=(15,15))


# In[41]:


core_weather["target"] = core_weather.shift(-1)["temp"]


# In[42]:


core_weather = core_weather.iloc[:-1,:].copy()


# In[43]:


core_weather.columns


# In[44]:


from sklearn.linear_model import Ridge
reg = Ridge(alpha=1.5)


# In[50]:


predictors = [ "temp","condition","humidity","pressure","rain","wdirect","fog"]


# In[51]:


train = core_weather.loc[:"31-01-2012"]
test = core_weather.loc["01-02-2012":] 


# In[52]:


train


# In[53]:


test


# In[54]:


reg.fit(train[predictors], train["target"])


# In[55]:


predictions = reg.predict(test[predictors])


# In[56]:


from sklearn.metrics import mean_squared_error

mean_squared_error(test["target"], predictions)


# In[57]:


combined = pd.concat([test["target"], pd.Series(predictions, index=test.index)], axis=1)
combined.columns = ["actual", "predictions"]


# In[58]:


combined


# In[59]:


combined.plot()


# In[60]:


mean_squared_error(test["target"], predictions)


# In[61]:


reg.coef_


# In[62]:


core_weather.corr()["target"]


# In[63]:


combined["diff"] = (combined["actual"] - combined["predictions"]).abs()


# In[100]:


combined.sort_values("diff").head(10)


# In[65]:


core_weather2 = core_weather[["condition", "fog", "humidity", "pressure","rain","temp","thunder","wdegree","wdirect"]].copy()
core_weather2.columns = ["condition", "fog", "humidity", "pressure","rain","temp","thunder","wdegree","wdirect"]


# In[66]:


core_weather2


# In[67]:


weather_y=core_weather2
weather_x=core_weather2


# In[68]:


train_X,test_X,train_y,test_y=train_test_split(weather_x,weather_y,test_size=0.2,random_state=4)


# In[69]:


train_X.shape


# In[70]:


train_y.shape


# In[71]:


train_y.head()


# In[72]:


from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(train_X,train_y)


# In[73]:


prediction2=regressor.predict(test_X)
np.mean(np.absolute(prediction2-test_y))


# In[74]:


print('Variance score: %.2f' % regressor.score(test_X, test_y))


# In[79]:


for i in range(len(prediction2)):
  prediction2[i]=np.round(prediction2[i],2)
pd.DataFrame({'Actual':test_y,'Prediction':prediction2,'diff':(test_y-prediction2)})


# In[80]:


regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(train_X,train_y)
print("Accuracy:{:.2f}%".format(regressor.score(test_X,test_y)*100))


# In[87]:


from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(max_depth=90,random_state=0,n_estimators=1000)
regr.fit(train_X,train_y)


# In[88]:


prediction3=regr.predict(test_X)
np.mean(np.absolute(prediction3-test_y))


# In[89]:


print('Variance score: %.2f' % regr.score(test_X, test_y))


# In[199]:


regr=RandomForestRegressor(max_depth=90,random_state=4,n_estimators=400)
regr.fit(train_X,train_y)
print("Accuracy:{:.2f}%".format(regr.score(test_X,test_y)*100))


# In[93]:


model=LinearRegression()
model.fit(train_X,train_y)


# In[94]:


prediction = model.predict(test_X)


# In[95]:


np.mean(np.absolute(prediction-test_y))


# In[96]:


print('Variance score: %.2f' % model.score(test_X, test_y))


# In[97]:


model=LinearRegression()
model.fit(train_X,train_y)
print("Accuracy:{:.2f}%".format(model.score(test_X,test_y)*100))


# In[110]:


core_weather3 = core_weather[["condition", "fog", "humidity", "pressure","rain","temp"]].copy()
core_weather3.columns = ["condition", "fog", "humidity", "pressure","rain","temp",]


# In[111]:


weather_Y=core_weather3.pop("temp")
weather_X=core_weather3


# In[173]:


x_train,x_test,y_train,y_test=train_test_split(weather_X,weather_Y,test_size=0.2,random_state=4)


# In[174]:


x_train.shape


# In[175]:


y_train.shape


# In[176]:


y_test.shape


# In[177]:


x_test.shape


# In[178]:


model=LinearRegression()
model.fit(x_train,y_train)


# In[179]:


prediction = model.predict(x_test)


# In[180]:


np.mean(np.absolute(prediction-y_test))


# In[181]:


print('Variance score: %.2f' % model.score(x_test, y_test))


# In[182]:


model=LinearRegression()
model.fit(x_train,y_train)
print("Accuracy:{:.2f}%".format(model.score(x_test,y_test)*100))


# In[183]:


regressor=DecisionTreeRegressor(random_state=4)
regressor.fit(x_train,y_train)


# In[184]:


prediction2=regressor.predict(x_test)
np.mean(np.absolute(prediction2-y_test))


# In[185]:


print('Variance score: %.2f' % regressor.score(x_test, y_test))


# In[186]:


for i in range(len(prediction2)):
  prediction2[i]=round(prediction2[i],2)
pd.DataFrame({'Actual':y_test,'Prediction':prediction2,'diff':(y_test-prediction2)})


# In[187]:


regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x_train,y_train)
print("Accuracy:{:.2f}%".format(regressor.score(x_test,y_test)*100))


# In[188]:


from sklearn.ensemble import RandomForestRegressor
regr=RandomForestRegressor(max_depth=90,random_state=4,n_estimators=200)
regr.fit(x_train,y_train)


# In[189]:


prediction3=regr.predict(x_test)
np.mean(np.absolute(prediction3-y_test))


# In[190]:


print('Variance score: %.2f' % regr.score(x_test, y_test))


# In[191]:


for i in range(len(prediction3)):
  prediction3[i]=round(prediction3[i],2)
pd.DataFrame({'Actual':y_test,'Prediction':prediction3,'diff':(y_test-prediction3)})


# In[196]:


regr=RandomForestRegressor(max_depth=90,random_state=4,n_estimators=1000)
regr.fit(x_train,y_train)
print("Accuracy:{:.2f}%".format(regr.score(x_test,y_test)*100))


# In[193]:


model=LinearRegression()
model.fit(x_train,y_train)
print("Accuracy:{:.2f}%".format(model.score(x_test,y_test)*100))


# In[194]:


model = Ridge(alpha=2.5)

model.fit(x_train, y_train)
print("Accuracy:{:.2f}%".format(model.score(x_test,y_test)*100))


# In[ ]:


import pandas as pd
from sklearn.linear_model import Ridge

# Assuming you have trained your Ridge Regression model and have new_data for forecasting

# Create an instance of the Ridge Regression model
model = RandomForestRegressor()

# Train the model with your training data (assuming your training data is stored in train_X and train_y)
model.fit(x_train, y_train)

# Make predictions on the new data
new_data = [[16,0,32,1010,1]]   # List with a single data point

predictions = model.predict(new_data)

# Create a DataFrame to store the predictions
predictions_df = pd.DataFrame(predictions, columns=["Forecast"])

# Export the predictions to an Excel file
predictions_df.to_excel("prediction1.xlsx", index=True)


# In[206]:


x=((core_weather.loc[:,core_weather.columns!='condition']).astype(int)).values[:,0:]
y=core_weather['condition'].values
     


# In[207]:


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.1,random_state=2)


# In[ ]:


core_weather4 = core_weather[["pressure", "fog", "humidity","rain","temp"]].copy()
core_weather4.columns = ["pressure", "fog", "humidity" ,"rain","temp",]


# In[ ]:




