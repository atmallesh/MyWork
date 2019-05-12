#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df=pd.read_csv("E:\\01.ATM\\RDSP\\RDSP_Txn.csv")
print(df.head())
print("Dimension", df.ndim)
print(df.shape)


# In[2]:


# To check correlation
df.axes
newdf=df.loc[:,['OB_Suc', 'WhatsApp_Del', 'Meeting_Atten', 'Txn_Cnt']]
print(newdf.head())


# In[3]:


newdf.corr()


# In[11]:


import matplotlib.pyplot as plt
import statsmodels.graphics.api as smg
corr_matrix = np.corrcoef(newdf.T)
smg.plot_corr(corr_matrix, xnames=newdf.columns)
plt.show()


# In[12]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# OLS = Ordinary Least Squares - Regression
# smf = stats model formula

results_cal_sug_pro_fib = smf.ols('Txn_Cnt ~ OB_Suc', data=newdf).fit()
print(results_cal_sug_pro_fib.summary())


# In[16]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# OLS = Ordinary Least Squares - Regression
# smf = stats model formula

results_cal_sug_pro_fib = smf.ols('Txn_Cnt ~ OB_Suc + WhatsApp_Del', data=newdf).fit()
print(results_cal_sug_pro_fib.summary())


# In[17]:


import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# OLS = Ordinary Least Squares - Regression
# smf = stats model formula

results_cal_sug_pro_fib = smf.ols('Txn_Cnt ~ OB_Suc + WhatsApp_Del + Meeting_Atten', data=newdf).fit()
print(results_cal_sug_pro_fib.summary())


# In[18]:


results_sug_fib_fat_fib = smf.ols('Txn_Cnt ~ OB_Suc + WhatsApp_Del + Meeting_Atten', data=newdf).fit()
print(results_sug_fib_fat_fib.summary())

fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(results_sug_fib_fat_fib, fig=fig)


# In[ ]:




