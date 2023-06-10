import numpy as np

length,timedata=0,0
filename="indtrack6.txt"

f=open(filename,"r")
data=f.readlines()
prices=[]
aux=[]
for i in data:
    timedata+=len(i.split())
    aux+=[float(r) for r in i.split()]
    if len(i.split())==1:
        prices.append(aux)
        aux=[]
lenght=len(prices)
timedata=int(timedata/lenght) 
prices=np.array(prices)
returns=np.zeros((lenght,timedata-1))
print("this archive have %d assets (including index) and a timedata of %d "%(lenght,timedata))

for i in range(timedata-1):
    returns[:,i]=(prices[:,i+1]-prices[:,i])/prices[:,i]    
        