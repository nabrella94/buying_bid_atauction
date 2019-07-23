#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib
import warnings

warnings.filterwarnings("ignore")


# In[154]:


#reading the auction list file which is avaialb;e for public and dealer for different brnches on https://www.iaai.com/LiveAuctions

automobile = pd.read_csv(r'C:\Users\Dell\Downloads\auction\BranchSalesListItems_07082019.csv')
automobile.head()


# In[155]:



 #removing unneccessary coloumns and features are less important for buyer to buy a car . 
automobile.drop(['Item#','Cylinders','Fuel Type','Public','Lane', 'Vehicle Location Info','Auction Date','Buy Now Price'],axis=1 , inplace=True)
#replace null value with zero value 

automobile.head()


# In[156]:


#replacing odemeters which has nan value with zero
automobile.Odometer=automobile.Odometer.fillna(0)


# In[157]:


#classified cars in 4 segments , we are looking to analyze the price of cars in Economy segment 

automobile.insert(4, "segment",  True)
def Classifysegment(make):
 
    Midlevel = ['Alfa Romeo', 'RAM','Chrysler', 'Infiniti', 'MINI', 'Volkswagen' , 'JEEP','CADILLAC','LINCOLN']
    Economy = ['TOYOTA' , 'HONDA','MAZDA' , 'CHEVROLET' , 'MERCURY','BUICK','SUBARU','VOLVO' ,'GMC','DODGE','KIA','Saab', 'PONTIAC' , 'NISSAN' ,'JEEP']
    Laxury=['BMW','LEXSUS', 'MERCEDES-BENZ','LEXUS','PORSCHE','JAGUAR']
    SUPERLUXERY=['BENTLEY' , 'MASARATI']
    if make in Midlevel:
        return 'Midlevel'
    elif make in Economy:
        return 'Economy'
automobile['segment'] = [Classifysegment(make) for make in automobile.Make ]#list comprehension
#Rename Some column of dataset
automobile=automobile.rename(columns = {'Run & Drive':'Run'})

#This bar char below shows destribution of car by make in the IAA auction
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
automobile.Make.value_counts().nlargest(10).plot(kind='bar', figsize=(15,5))
plt.title("Number of vehicles by make")
plt.ylabel('Number of vehicles')
plt.title('Disdtribution of car for sales in auction by brand', color='green', fontsize=14)
plt.show()


# In[158]:



#conver Milage DataType from string to Intiger
automobile.Odometer=automobile.Odometer.astype(str).str.replace(",", "").astype(int)

#filtering cars for sales in the auction to those with these condition . IN my experince these cars can be sell in tax season
#with highest profit with less amount of time

automobile=automobile.rename(columns = {'Sale Document':'SaleDoc','Primary Damage':'PrimaryDamage','Secondary Damage':'SecondaryDamage'})

#filtering cars with the condition that with my experince I can sell buy and repire them less than $3000 and make $1000 profit
df_filtered =automobile[(automobile.Odometer<= 160000 )&( automobile.Odometer > 80000 ) & (automobile.Starts == "YES") & (automobile.Run == "YES")
                        & (automobile.segment == "Economy") & (automobile.Year<= 2011 ) & (automobile.Year>= 2005 )
                       & (automobile.SaleDoc.str.contains('CLEAR')) & (pd.notnull(automobile['Provider']))  & (automobile.PrimaryDamage !="LEFT REAR") ] 
#I droped agifn unessesary colloumns to reach the coloumns which make the most infuence in buyer bid price , I want to 
#give these data to machine to predict the price
df_filtered.drop(['segment','Starts','Run','Odometer Status', 'Transmission','SaleDoc' ,'VIN','Loss Type' ,'Provider'],axis=1 , inplace=True)
df_filtered.insert(5, "$BidSalesPrices",0)
#adding the price of car that sold in auction at bid price in the BidSalesPrices coloumn 

df_filtered.loc[:, "$BidSalesPrices"] = [ 1050, 1600,7000, 1500, 975, 1700, 1250, 3450,1100,1000,1850]
#creating dictionary for dameges of the cars which get to the accident , I rated the type accident from low to high regarding what
#part of car get to accident . IF it doesnot have an accident I rated that 0 
df_filtered.SecondaryDamage = df_filtered.SecondaryDamage.replace(np.nan, 'nothing')
PrimaryDamage = {'FRONT END': 1.0,'LEFT SIDE': 3.0,'FRONT & REAR':4.0 ,'LEFT & RIGHT':5.0}
SecondaryDamage = {'REAR': 2,'RIGHT FRONT': 2.5,'LEFT SIDE':3,'FRONT & REAR':4 ,'LEFT & RIGHT':5 ,'nothing':0}
df_filtered.loc[:, "PrimaryDamage"] = [PrimaryDamage[item] for item in df_filtered.PrimaryDamage]
df_filtered.loc[:, "SecondaryDamage"] = [SecondaryDamage[item] for item in df_filtered.SecondaryDamage]
warnings.filterwarnings("ignore", category=FutureWarning)

df_filtered.head()


# In[159]:


#import the necessary modules
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

# 1. Encoding
svc_model = LinearSVC(random_state=0)
le = preprocessing.LabelEncoder()
# 1. A. Encoding Make
ls = le.fit(df_filtered['Make'].values)
df_filtered['Make'] = ls.transform(df_filtered['Make'].values) 
# 1. B. Encoding Model
le = le.fit(df_filtered['Model'].values)
df_filtered['Model'] = le.transform(df_filtered['Model'].values) 

# 2. Training
pred = svc_model.fit(df_filtered, df_filtered['$BidSalesPrices'])

# 3. Testing
pred = pred.predict(df_filtered)
print("LinearSVC accuracy : ",accuracy_score(df_filtered['$BidSalesPrices'], pred, normalize = True))
pred


# In[ ]:




