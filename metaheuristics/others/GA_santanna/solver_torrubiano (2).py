import numpy as np
import cplex as cp
import itertools as it

'''
Solver file for GA_santanna using the Torrubiano model
'''
print("Using Torrubiano Encoding.")
class Solve:
    def __init__(self,filename,timedata,ibov=False):
        self.filename=filename
        self.modelname="torrubiano"
        self.lenght,self.n,self.t,self.ub,self.lb=0,0,0,1,0.001  
        self.insample=int((timedata-1)/2)
        self.outsample=int((timedata-1)/2)#assets counter, it includes the index
        self.n=0
        if ibov==False:
            print("Reading the indtrack file")
            self.readfile()
        if ibov==True:
            print("Getting data from the .csv file")
            import ibovdata
            x,r=ibovdata.get_var()
            print("The csv file have %d assets (including index) and %d time info"%(len(x),len(x[0])))
            self.lenght=len(x)
            self.var(x,r)
            


    def readfile(self):
        self.timedata=291
        counter,n=0,0
        print("Reading the file "+str(self.filename))
        f=open(self.filename,"r")
        data=f.readlines()
        for i in data:                                                  #structure to count assets and define vectors lenght
            if len(i.split()) ==3:
                self.lenght+=1
        print("The file have :" + str(self.lenght) +" assets:")
        x=np.zeros([self.lenght,self.timedata])                                        #prices
        r=np.zeros([self.lenght,self.timedata-1])                                        #returns
        for i in data:
            for j in i.split():
                x[counter][n]=float(j)                                  #basic reading structure
                n=n+1
            if len(i.split())==3:
                counter=counter+1
                n=0
        f.close()
        print("Information read, .txt file closed")    
        for t in range(1,291):                                          #290 periods
            r[:,t-1]=(x[:,t]-x[:,t-1])/x[:,t-1]
        self.var(x,r)
   
    def var(self,x,r):
        H=np.zeros([self.lenght-1,self.lenght-1])
        G=np.zeros([self.lenght-1])
                    #calculate matrix H
        for i in range(1,self.lenght):
            for j in range(1,self.lenght):
                s=0                                                     #auxiliary variable to sum the time series
                for t in range(self.insample):                                    #the optimization time is untill T=145
                    s=s+(1/self.insample)*r[i,t]*r[j,t]
                H[i-1,j-1]=s
                        #calculate matrix g
        for i in range(1,self.lenght):
            s=0
            for t in range(self.insample):
                s+=(1/self.insample)*r[i,t]*r[0,t]
            G[i-1]=s
        self.x,self.r,self.H,self.G=x,r,H,G
        #h=H.tolist()
        #g=G.tolist()

    def opt(self):
        print("Starting optimization.")
        c=cp.Cplex()
        time=[i for i in range(self.insample,self.insample+self.outsample)]
        gg=[(i,-self.G[i])for i in range(self.lenght-1)]
        square=[(i,j,self.H[i][j]) for i,j in it.product(range(self.lenght-1),range(self.lenght-1))]
        var=["w"+ str(i) for i in range(self.lenght-1)]       #in this model, we only have the given w by the population
        c.parameters.timelimit.set(5400)
        indices=c.variables.add(names=var)
        c.objective.set_sense(c.objective.sense.minimize)
        c.objective.set_quadratic_coefficients(square)
        c.objective.set_linear(gg)
        rows=[[var,[1 for i in range(self.lenght-1)]]]+[                  #Porfolio allocation constraint
                [var,[0 for i in range(j)]+[1]+[0 for i in range(self.lenght-2)]]for j in range(self.lenght-1)]+[          #lower bound
                [var,[0 for i in range(j)]+[1]+[0 for i in range(self.lenght-2)]]for j in range(self.lenght-1)]            #upper bound

        indices=c.linear_constraints.add(lin_expr=rows,
                                            senses=["E"]+["G" for i in range(self.lenght-1)]+["L" for i in range(self.lenght-1)],
                                            rhs=[1]+[self.lb for i in range(self.lenght-1)]+[self.ub for i in range(self.lenght-1)],
                                            names=["c"+str(i) for i in range(2*(self.lenght-1)+1)])
        print('Initalizing optimization \n \n')
        t0=c.get_time()
        c.solve()
        t1=c.get_time()
        obj=c.solution.get_objective_value()
        print(c.solution.get_objective_value())
        self.sol_composition=c.solution.get_values()
        return obj,self.sol_composition

    def solving(self,parent1,ctrl=0):      
        if len(parent1)!=self.lenght-1:
            print("Parent given in solving method doesn't use binary encoding. Fix this before using Solver.solving() method.")
            return -1
        parent1=parent1.copy()
        returns,h,g=self.adjust(parent1)
        #h=h.tolist()
        c=cp.Cplex()
        print("Setting CPLEX model parameters.")
        time=[i for i in range(self.insample,self.insample+self.outsample)]
        gg=[(i,-g[i])for i in range(len(returns))]
        square=[(i,j,h[i][j]) for i,j in it.product(range(len(returns)),range(len(returns)))]
        var=["w"+ str(i) for i in range(len(returns))]
        c.parameters.timelimit.set(5400)
        print("Adding constraints to the problem.")
        indices=c.variables.add(names=var)
        c.objective.set_sense(c.objective.sense.minimize)
        c.objective.set_quadratic_coefficients(square)
        c.objective.set_linear(gg)
        rows=[[var,[1 for i in range(len(returns))]]]+[                    #Porfolio allocation constraint                                                                                                #cardinality constraint
              [var,[0 for i in range(j)]+[1]+[0 for i in range(len(returns)-1-j)]]for j in range(len(returns))]+[          #lower bound
              [var,[0 for i in range(j)]+[1]+[0 for i in range(len(returns)-1-j)]]for j in range(len(returns))]           #upper bound

        indices=c.linear_constraints.add(lin_expr=rows,
                                         senses=["E"]+["G" for i in range(len(returns))]+["L" for i in range(len(returns))],
                                         rhs=[1]+[self.lb for i in range(len(returns))]+[self.ub for i in range(len(returns))],
                                         names=["c"+str(i) for i in range(len(rows))])
        print('All set up! Starting optimization \n \n')
        t0=c.get_time()
        c.solve()
        t1=c.get_time()
        print("Done!")
        solutiontime=t1-t0
        solution=c.solution.get_objective_value()
        sol_composition=c.solution.get_values()
        if ctrl==0:
            return sol_composition             #the sol_composition is a vector [a1,a2,a3,a4...], where a are the assets in parent1
        if ctrl==1:
            return solution

    def adjust(self,parent):
        parent=parent.copy()
        if len(parent)!=self.lenght-1:
            print("Given parent lenght is not equal to the units of the assets.Check if the parent is using the binary enconding.\n")
            return -1
        control=0
        zeros=0
        for i in parent:
            if i==1:
                control+=1
        returns=np.zeros([control,self.insample+self.outsample])
        H=np.zeros([control,control])
        G=np.zeros([control])
        #adjusts the r matrix
        if len(parent)==self.lenght-1:
            print("Given Parent have the same lenght of the class.")
            for i in range(len(parent)):                 #remember here that self.r[0,i] is the index data
                if parent[i]==0:
                    zeros+=1
                elif parent[i]==1:
                    returns[i-zeros,:]=self.r[i+1,:]
        #with the new returns matrix, calculates the H and G matrices.
        print('Returns vector lenght: '+str(len(returns)))
        for i in range(len(returns)):
            for j in range(len(returns)):
                s=0                                                     #auxiliary variable to sum the time series
                for t in range(self.insample):                                    #the optimization time is untill T=145
                    s=s+(1/self.insample)*returns[i,t]*returns[j,t]
                H[i,j]=s
        for i in range(len(returns)):
            s=0
            for t in range(self.insample):
                s+=(1/self.insample)*returns[i,t]*self.r[0,t]
            G[i]=s
        print("Adjust function executed with success.")
        return returns,H,G

    def get_r(self):
        return self.r.copy()


    def get_lenght(self):
        return self.lenght
    def get_id(self):
        return 0
