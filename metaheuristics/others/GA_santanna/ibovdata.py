import pandas as pd
import numpy as np
import os

def get_var():
    return prices,returns

def change_ind(list):       #take the index in the last position and bring it to the first, maintaining the assets orders
    list=list.copy()
    aux=[]
    for i in range(len(list)-1,0,-1):
        aux=list[i-1].copy()
        list[i-1]=list[i].copy()
        list[i]=aux.copy()
        aux=[]
    return list.copy()

curpath=os.getcwd()+"/"
name="ibov.csv"
start,end=2,229                  #insert the data index you want to start the sample and the data you want to end
timedata=end-start
data=pd.read_csv(curpath+name)
print(data)
data=data.drop(["date","OEX","UKX","DAX","Dolar PTAX"],axis=1)
data=data.drop([i for i in range(start-1)])
names=list(data.columns) 
prices=data.to_numpy()
prices=np.transpose(prices)
#prices=np.array(prices)
#for i in names:
    #prices.append(data.loc[:,i].values.tolist())
price=prices[:,start-1:end-1]
prices=change_ind(prices)
returns=np.zeros([len(prices),timedata-1])
for t in range(1,len(prices[0])):
    returns[:,t-1]=(prices[:,t]-prices[:,t-1])/prices[:,t-1]



