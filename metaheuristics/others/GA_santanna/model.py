#This file will pick the objective value for the GA in the exact model 
from ast import literal_eval as liteval
import os

#print("----------REMEMBER TO CHECK IF THE VALUES OF unisize("+str(unisize)+") AND k_size("+str(k_size)+") ARE THE SAME OF THE ONES GIVEN IN INTERFACE.----------\n\n")
#input("Enter anything to if you checked and want to continue.")
class Model:
   
    def __init__(self,ksize,usize,id,ibov=False):
        self.id=id  #1 for wang
        self.currpath=os.getcwd()   
        if id==0:
            print("Acessing results for torrubiano.")
            os.chdir('torrubiano_latest_solutions')
        if id==1:
            print("Acessing results for wang.")
            os.chdir('wang_solutions')
        self.newpath=os.getcwd()
        self.k_size=ksize
        self.unisize=usize
        if ibov==False:
            if id==0:
                self.name=self.newpath+'//torrubiano_solution_track'+str(self.unisize*10)+str(self.k_size)+".txt"
            if id==1:
                self.name=self.newpath+'//wang_solution_track'+str(self.unisize*10)+str(self.k_size)+'.txt'
        if ibov==True:
            if id==0:
                self.name=self.newpath+'//torrubiano_solution_track_ibov_'+str(self.k_size)+".txt"
            if id==1:
                self.name=self.newpath+'//wang_solution_track_ibov_'+str(self.k_size)+".txt"
    def get_objvalue(self):
        if self.k_size==0 or self.unisize==0:
            print("Before using any function in 'model.py', define the 'k' and unisize parameters using the setfile() method.")
            #os.chdir("../")
            return -1
        print('File opened path: '+self.name)
        print("\n")
        with open(self.name,"r") as f:
            line1=f.readline()
            if line1=='Objective Value: \n':
                line2=f.readline()
                #os.chdir("../")
                return float(line2)
            else:
                print("Error getting the objective value. Check if the reading file is in the modelname_solution_trackXX0YY.txt layout.")
            f.close()
            #os.chdir("../")

    def get_optcomp(self):
        if self.k_size==0 or self.unisize==0:
            print("Before using any function in 'model.py', define the 'k' and unisize parameters using the setfile() method.")
            #os.chdir("../")
            return -1
        print('Getting the composition vector in: '+self.name)
        print("\n") 
        with open(self.name,'r') as f:
            for i in range(5):
                f.readline()
            comp=[]
            s=f.readline()
            comp=liteval(s)
            #print(comp)
            print('Got the optimal composition vector!')
            #os.chdir("../")
            return comp

    def get_optparent(self):
        print("Ksize: %d\n Unisize: %d"%(self.k_size,self.unisize))
        if self.k_size==0 or self.unisize ==0:
            print("Before using any function in 'model.py', define the 'k' and unisize parameters using the setfile() method.")
            #os.chdir("../")
            return -1
        print("Getting the optimal parent.")
        print("\n")
        with open(self.name,'r') as f:
            for i in range(5):
                f.readline()
            comp=[]
            s=f.readline()
            comp=liteval(s)
            #print(comp)
            print('Got the optimal composition vector. Transforming it in the equivalent parent.')
        parent=[]
        for i in comp:
            if i==0:
                parent.append(0)
            elif i!=0:
                parent.append(1)
        #os.chdir("../")
        return parent
   




