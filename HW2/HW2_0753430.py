
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# ### 讀入資料
# - 取10～12月資料
# - 將NR轉換成0

# In[2]:


data = pd.read_excel('hsinchu.xls')
#10~12
data = data[data['日期'].between('2017/10/01','2017/12/31	')]
# NR->0
data.replace('NR',0, inplace=True)


# In[3]:


data.shape


# In[4]:


data.head(19) # 18 features


# ### 清理缺失值及有問題的數值

# In[5]:


# row 1656 column= 3~27
def f_element(x):
    return type(x)==str or math.isnan(x)
            
def before_element(i,j):
    if j-1==2:      
        i = i - 18
        j = 26
    else:
        j = j-1
    if f_element(data.iloc[i,j]):
        return before_element(i,j)
    else:
        return data.iloc[i,j]

def after_element(i,j):
    if j+1==27:      
        i = i + 18
        j = 3
    else:
        j = j+1
    if f_element(data.iloc[i,j]):
        return after_element(i,j)
    return data.iloc[i,j]

for i in range (1656):
    for j in range(3,27):
        if f_element(data.iloc[i, j]):
            data.iloc[i, j] = (before_element(i,j)+after_element(i,j))/2


# ### 資料分割
# #### 先以單一項目PM2.5來預測PM2.5值

# In[6]:


# 分成training data(10~11月) 及 testing data(12月)
train_data = data.iloc[:1098] # 61 day * 18 = 1098
test_data = data.iloc[1098:,]  # 31 day

#將數值(0~23)與測項分開，並將數值分成61等份(每個等份分別代表每一天的18個測項)
train_data_values = train_data.drop(['日期','測站','測項'],axis=1).values   
train_data_values = np.vsplit(train_data_values, 61)   #61 days

#保留測項名稱->index
train_data_18 = np.array(train_data.iloc[0:18,2]).reshape(18,1)   # index

#將每一天的18個測項加在index(row)後，形成18個index對應到 0~23, 0~23.....共61次(天)
for i in range(len(train_data_values)):
    train_data_18 = np.concatenate((train_data_18, train_data_values[i]), axis=1)

#轉換成DataFrame並設定index(row的名稱)
train_data_18 = pd.DataFrame(train_data_18)
train_data_18 = train_data_18.set_index(0)

#分割訓練集次數
train_num = train_data_18.shape[1] - 6 #1464-6

#取PM2.5
train_pm25= train_data_18.loc['PM2.5'].tolist()


#取前六筆資料預測第七筆，以此類推，可分出train_num組預測集
X_train = []
y_train = []
for i in range(train_num):
    X_train.append(train_pm25[i:i+6])   #前六筆
    y_train.append(train_pm25[i+6])     #第七筆

#將list轉成np.array性質
X_train, y_train = np.array([X_train]), np.array([y_train])
X_train, y_train = X_train.reshape(1458, 6), y_train.reshape(1458,1)


# In[7]:


# 測試集(12月)分法與訓練集相同
test_data_values = test_data.drop(['日期','測站','測項'],axis=1).values
test_data_values = np.vsplit(test_data_values, 31)
test_data_18 = np.array(test_data.iloc[0:18,2]).reshape(18,1)

for i in range(len(test_data_values)):
    test_data_18 = np.concatenate((test_data_18, test_data_values[i]), axis=1)

test_data_18 = pd.DataFrame(test_data_18)
test_data_18 = test_data_18.set_index(0)

test_num = test_data_18.shape[1] -6

test_pm25= test_data_18.loc['PM2.5'].tolist()

X_test = []
y_test = []
for i in range(test_num):
    X_test.append(test_pm25[i:i+6])
    y_test.append(test_pm25[i+6])

X_test, y_test = np.array([X_test]), np.array([y_test])
X_test, y_test = X_test.reshape(738, 6), y_test.reshape(738,1)


# In[8]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

lm = LinearRegression()
lm.fit(X_train, y_train)

y_train_pred = lm.predict(X_train)
y_test_pred = lm.predict(X_test)

print('LinearRegression(PM2.5)\n')
print('[MAE] train: %.2f, test: %.2f' %
      (mean_absolute_error(y_train, y_train_pred),
       mean_absolute_error(y_test, y_test_pred)))
print('[MSE] train: %.2f, test: %.2f' %
      (mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))
print('[R^2] train: %.2f, test: %.2f' % (r2_score(y_train, y_train_pred),
                                                           r2_score(y_test, y_test_pred)))


# In[9]:


lm.coef_


# In[10]:


from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(
    n_estimators=100, criterion='mse', random_state=1, n_jobs=-1, max_depth=10)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

print('RandomForestRegressor(PM2.5)\n')
print('[MAE] train: %.2f, test: %.2f' %
      (mean_absolute_error(y_train, y_train_pred),
       mean_absolute_error(y_test, y_test_pred)))
print('[MSE] train: %.2f, test: %.2f' %
      (mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))
print('[R^2] train: %.2f, test: %.2f' % (r2_score(y_train, y_train_pred),
                                       r2_score(y_test, y_test_pred)))


# In[11]:


forest.feature_importances_


# ### 使用18種測項預測PM2.5

# In[12]:


#18種測項
X_train_all = [ ]
for i in range(train_num):
    X_train_all.append(train_data_18.iloc[:,i:i+6].values.ravel().tolist())

X_train_all = np.array([X_train_all])
X_train_all = X_train_all.reshape(1458, 108)

X_test_all = []
for i in range(test_num):
    X_test_all.append(test_data_18.iloc[:,i:i+6].values.ravel().tolist())

X_test_all = np.array([X_test_all])
X_test_all = X_test_all.reshape(738, 108)


# In[13]:


# 將不同測項標準化
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train_all)
X_train_all = sc.transform(X_train_all)
X_test_all = sc.transform(X_test_all)


# In[14]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

lm = LinearRegression()
lm.fit(X_train_all,y_train)

y_train_pred = lm.predict(X_train_all)
y_test_pred = lm.predict(X_test_all)

print('LinearRegression(18 features)\n')
print('[MAE] train: %.2f, test: %.2f' %
      (mean_absolute_error(y_train, y_train_pred),
       mean_absolute_error(y_test, y_test_pred)))
print('[MSE] train: %.2f, test: %.2f' %
      (mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))
print('[R^2] train: %.2f, test: %.2f' % (r2_score(y_train, y_train_pred),
                                                           r2_score(y_test, y_test_pred)))

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(
    n_estimators=100, criterion='mse', random_state=1, n_jobs=-1)
forest.fit(X_train_all, y_train)
y_train_pred = forest.predict(X_train_all)
y_test_pred = forest.predict(X_test_all)

print('RandomForestRegressor(18 features)\n')
print('[MAE] train: %.2f, test: %.2f' %
      (mean_absolute_error(y_train, y_train_pred),
       mean_absolute_error(y_test, y_test_pred)))
print('[MSE] train: %.2f, test: %.2f' %
      (mean_squared_error(y_train, y_train_pred),
       mean_squared_error(y_test, y_test_pred)))
print('[R^2] train: %.2f, test: %.2f' % (r2_score(y_train, y_train_pred),
                                       r2_score(y_test, y_test_pred)))


# In[15]:


get_ipython().system('jupyter nbconvert --to script hw3.ipynb')


# In[16]:


get_ipython().run_line_magic('ls', '')

