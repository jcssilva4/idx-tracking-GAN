import time 
import random as rand
import solver_wang as solver

class Population:
    def __init__(self,filename,popsize,card,num_mutation):
        self.num_mutation=num_mutation
        self.filename=filename
        self.popsize=popsize
        self.k=card   #cardinality
        self.Solver=solver.Solve(self.filename)
        self.id=self.Solver.get_id()  #1 for wang
        if self.id==0:
            self.lenght=self.Solver.get_lenght()-1  #ONLY ON TORRUBIANO -1 to remove the index counting
        if self.id==1:
            self.lenght=self.Solver.get_lenght()
        self.universe=[i for i in range(1,self.lenght+1)]       
        self.gen_pop()

    def get_id(self):
        return self.id

    def gen_pop(self):                         #generates random population
        self.population=[]
        for i in range(self.popsize):     
            parent=rand.sample(self.universe,self.k)
            parent.sort()
            self.population.append(parent)

    def get_parents(self,populatio):
        population=populatio.copy()
        parent1,parent2=rand.sample(population,2)
        return parent1,parent2

    def solve(self,parent):
        parent=parent.copy()
        sol_comp=self.Solver.solving(parent)
        return sol_comp
        

    def crossover(self,parent1,parent2):           #this function performs the crossover method in Sant'Anna2017 an return the childs in Torrubiano encoding
        if len(parent1) and len(parent2) != len(self.universe):
            print("Inserted parents to the crossover do not use Torrubiano encoding.")
            parent1,parent2=parent1.copy(),parent2.copy()
            parent1=self.bin(parent1)
            parent2=self.bin(parent2)
        if len(parent1) and len(parent2)==len(self.universe):
            binparent1,binparent2=parent1.copy(),parent2.copy()
            #print("Parent1: %s \nParent2: %s" %(binparent1,binparent2))
            cut=rand.randint(0,len(binparent1)-1)        #sort the cutoff point
            #print("Cutoff point: %d" %cut)
            binchild1,binchild2=[],[]
            binchild1=binparent1[:cut]+binparent2[cut:]
            binchild2=binparent2[:cut]+binparent1[cut:]
            #print("Childs before correcting cardinality: \n%s\n%s"%(binchild1,binchild2))
            binchild1,binchild2=self.adjustchild(binchild1),self.adjustchild(binchild2)
            #print("Childs after correcting cardinality: \n%s\n%s"%(binchild1,binchild2))
            child1,child2=self.tencode(binchild1),self.tencode(binchild2)
            return child1,child2

    def adjustchild(self,childd):                  #this function takes a binary child and adjust it cardinality to 'size'
        if len(childd)!=len(self.universe):
            print("Child does not use the binary encoding. Fix it before using this function.")
            return -1
        child=childd.copy()
        #print(child)
        indexes=[]
        aux=[i for i in range(len(self.universe))]                              
        a=0
        for i in range(len(child)):
            if child[i]==1:
                indexes.append(i)   
                aux.pop(i-a)
                a+=1
        #print(indexes)            
        #print(aux)
        while len(indexes)!=self.k:
            if len(indexes)<self.k:
                print("Adding asset in child.")
                ind=aux.pop(rand.randint(0,len(aux)-1))          #the number popped is the index for substitution
                #print(ind)
                child[ind]=1
                indexes.append(ind)
                #print(child)
            elif len(indexes)>self.k:
                print("Removing assets from child.")
                ind=indexes.pop(rand.randint(0,len(indexes)-1))
                #print(ind)
                child[ind]=0
                aux.append(ind)
                aux.sort()
        return child

    def mutation(self,parent):
        parent=parent.copy()
        indexes,aux=[],[]
        if len(parent)!=len(self.universe):
            print("Parent used to the mutation doesn't use binary encoding, translate it before using mutation method.")
            return -1
        for i in range(len(parent)):
            if parent[i]==1:
                indexes.append(i)
            if parent[i]==0:
                aux.append(i)
        if self.num_mutation==1:                    
            print("Applying single mutation.")
            ind1,aux1=indexes[rand.randint(0,len(indexes)-1)],aux[rand.randint(0,len(aux)-1)]
            print("Index changed for 0: %d\nIndex changed for 1: %d" %(ind1,aux1))
            parent[ind1],parent[aux1]=0,1
            return parent

        if self.num_mutation==2:
            print("Applying double mutation.")
            ind,aux=rand.sample(indexes,2),rand.sample(aux,2)
            ind1,ind2,aux1,aux2=ind[0],ind[1],aux[0],aux[1]
            print("Index changed for 0: %d,%d\nIndex changed for 1: %d,%d" %(ind1,ind2,aux1,aux2))
            parent[ind1],parent[ind2],parent[aux1],parent[aux2]=0,0,1,1
            return parent               
            
    def get_mse(self,comp_vector): #calculates the mse to a given composition vector
        if len(comp_vector)!=self.lenght:
            print("Parent in get_mse() is no binary. Fix this before using the method. ")
        if len(comp_vector)==self.lenght:
            aux=0
            for i in comp_vector:
                aux+=i
            if abs(aux-1)>=10**-6:
                print("This composition vector violates capital allocation constraint.")
            r=self.Solver.get_r()
            #print('Calculating the MSE for the compositon:'+ str(comp_vector))
            mse=0
            for t in range(145): #for this porpuse, we consider only the in-sample period
                sum=0
                for j in range(self.lenght-1):
                    sum+=comp_vector[j]*r[j+1,t]
                err=(sum-r[0,t])**2
                mse+=err
            mse=mse/145
            return mse

    def fit_all(self,population,id):
        fitness=[]
        population=population.copy()
        if id==0:
            print('Fitting acording to torrubiano solver')
            for i in population:
                i=self.bin(i)
                comp=self.Solver.solving(i)
                composition=self.btranslate(comp,i)
                mse=float(self.get_mse(composition))
                fitness.append(mse)
            return fitness.copy()
        if id==1:
            for i in population:
                i=self.bin(i)
                fit=self.Solver.solving(i,1)
                fitness.append(fit)
            return fitness.copy()

    def fit(self,parent,id,control=0): #control==0 return the mse, ==1 return the composition
        parent=parent.copy()
        if id==0:    
            if len(parent)!=self.lenght:
                print("Parent for fitness calculation does not use binary encoding. Fix the given parameter for fit() method.")
                return -1
            comp=self.Solver.solving(parent)
            if control==0:
                composition=self.btranslate(comp,parent)
                print(composition)
                mse=float(self.get_mse(composition))
                return mse
            elif control==1:
                return comp
        if id==1:
            if len(parent)!=self.lenght:
                print("Parent for fitness calculation does not use binary encoding. Fix this parameter befor using fit() method")
                return -1
            else:
                comp=self.Solver.solving(parent)
                if control==0:
                    fit=self.Solver.solving(parent,1)
                    return fit
                elif control==1:
                    return comp            

    def btranslate(self,vector,parent):
        vector=vector.copy()
        parent=parent.copy()
        comp=[]
        j=0
        for i in range(len(parent)):
            if parent[i]==0:
                comp.append(0)
            elif parent[i]==1:
                comp.append(vector[j])                                               
                j+=1
            else:
                print("Compared element: "+str(parent[i]))
                print("Vector: "+str(vector))
                print("Parent: "+str(parent))
                print("Problem iterating the btranslate() function. Check here.")
                if len(parent)==self.size:
                    print("Parent is in the Torrubiano encoding. Please change it to binary encoding before giving it here.")
                    return -1
        return comp

    def bin(self,parent):
        count=[0 for i in range(self.lenght)]
        for i in parent:
            count[i-1]=1
        return count
    
    def tencode(self,list):                #this functions transform a binary vector in the Torrubiano encoding
        tlist=[]
        for i in range(len(list)):
            if list[i]==1:
                tlist.append(i+1)
        tlist.sort
        return tlist

           