import model
import population as populationn
import os
import random as rand
import time
fileversion="v8.0"
gatype='Santanna'
os.chdir('solutions')
mutation_rate,popsize=85,15
it=1       #number of iterations performed by GA
timelimit=300
filename="indtrack1.txt"
ibov=True
timedata=227
kvalues=[9]  
num_mutation=1
objerror=10**-6 #error to the GA says a solution is equals to cplex solution
def maxx(list):                      #returns the index of the two worst fitnesss parents
    i1,i2=None,None
    w1,w2=0,0
    #print("len(list): %d"%len(list))
    #print("Max given list: %s"%list)
    for j in range(len(list)):
        if list[j]>w1:
            w2,w1,i2,i1=w1,list[j],i1,j
        elif list[j]>w2:
            w2,i2=list[j],j
    if i1>i2:
        return i1,i2
    elif i2>i1:
        return i2,i1

def worstfit(list,control=0):
    aux,ind=0,None
    for i in range(len(list)):
        if list[i]>aux:
            ind=i
            aux=list[i]
    if control==0:
        return aux
    if control==1:
        return ind

def bestfit(list,control=0):
    fit,ind=float('inf'),None
    for i in range(len(list)):
        if list[i]<fit:
            fit=list[i]
            ind=i
    if control==0:
        return fit
    if control==1:
        return ind
    if control==2:
        return fit,ind

def checkl(element,list):            #check if this element is already on a list
    for i in list:
        if element==i:
            return 1
    if len(element)==0:
        return 2
    return 0

def check(element1,element2):
    element1,element2=round(element1,12),round(element2,12)
    #if abs(element1-element2)<=objerror:
    if element1==element2:
        return 1
    else:                                             
        return 0

for k in kvalues:
    os.chdir('../')
   
    pop=populationn.Population(filename,popsize,k,num_mutation,timedata,ibov)
    id=pop.get_id()                       #0 for torrubiano, 1 for wang
        
    mod=model.Model(k,pop.lenght,id,ibov=ibov)
    optparent=mod.get_optparent()                           
    optcomp=mod.get_optcomp() 
    if id==0:                            
        cplex="torrubiano"
        optmse=float(pop.get_mse(optcomp))
        optmse=round(optmse,12)
    if id==1:
        cplex="wang"
        optmse=mod.get_objvalue()
    
    times=[] #vector with all the optimization times
    relfit=[]
    bfitness=[]
    parents,compositions,iterations=[],[],[]
    count=0

    os.chdir("../solutions")
    if ibov==False:
        ar=open('s_'+str(filename[:-4])+"_"+str(k)+"_"+str(timelimit)+"_results_"+cplex[0]+".txt","w") #[:-4] is to avoid the '.txt' in filename
    if ibov==True:
        ar=open("s_ibov_"+str(k)+"_"+str(timelimit)+"_results_"+cplex[0]+".txt","w") 
    for ro in range(it):
        nit=0
        pop.gen_pop()
        population=pop.population.copy()    ##
        fitness=pop.fit_all(population,id)
        print("------------------------------------------------------------------------------------------------------------------")
        t0=time.time()
        t1=time.time()
        c,dontrepeat=0,0
        while abs(t0-t1)<timelimit:
            parent1,parent2=pop.get_parents(population)
            binparent1,binparent2=pop.bin(parent1),pop.bin(parent2)
            child1,child2=[],[]
            while checkl(child1,population)!=0 or checkl(child2,population)!=0:#remove children already included
                child1,child2=pop.crossover(binparent1,binparent2)
                binchild1,binchild2=pop.bin(child1),pop.bin(child2)
                m1,m2=rand.randint(1,100),rand.randint(1,100)
                if m1<=mutation_rate:
                    print("mutation applied!")
                    binchild1=pop.mutation(binchild1)
                if m2<=mutation_rate:
                    print("mutation applied to child2.")
                    binchild2=pop.mutation(binchild2)
                child1,child2=pop.tencode(binchild1),pop.tencode(binchild2)
                fitchild1,fitchild2=pop.fit(binchild1,id),pop.fit(binchild2,id)
            population.extend([child1,child2])
            fitness.extend([fitchild1,fitchild2])
            w1,w2=maxx(fitness)
            population.pop(w1),population.pop(w2)
            fitness.pop(w1),fitness.pop(w2)
            t1=time.time()
            nit+=1
            print("---------Instance number: %d------------"%ro)
           
            for i in range(len(population)):
                if check(fitness[i],optmse)==1:
                    times.append(t1-t0)
                    bestfi,bi=bestfit(fitness,2)
                    bfitness.append(bestfi)
                    iterations.append(nit)
                    parents.append(population[bi])
                    compositions.append(pop.fit(pop.bin(population[bi]),1))
                    c=1
                    break
            if c==1:
                print("CONVERGENCE!")
                count+=1
                break
        
        if t1-t0>timelimit:
            bestfi,bi=bestfit(fitness,2)
            bfitness.append(bestfi)
            parents.append(population[bi])
            iterations.append(nit)
            compositions.append(pop.fit(pop.bin(population[bi]),1))
            times.append(t1-t0)

    mtime=0
    for i in times:
        mtime+=i
    mtime=mtime/len(times)
    mrelfit=0
    for i in bfitness:
        i=round(i,12)
        aux=(i-optmse)/optmse
        relfit.append(aux)
        mrelfit+=aux
    mit=0
    for i in iterations:
        mit+=i
    mit=int(mit/len(iterations))

    wind,bind=worstfit(bfitness,1),bestfit(bfitness,1)
    wparent,bparent=parents[wind],parents[bind]
    wcomp,bcomp=compositions[wind],compositions[bind]

    mrelfit=mrelfit/len(relfit)
    ar.write("Mean time: %.12f   |   Iterations that met the CPLEX solution: %d\n"%(mtime,count))
    ar.write("Times:\n%s\n"%times)
    ar.write("Best Fitness of each iteration: \n%s\n"%bfitness)
    ar.write("Mean Relative fitness:\n%.12f\n"%mrelfit)
    ar.write("Relative fitness:\n%s\n"%relfit)
    ar.write("Worst Parent\Composition: %s\%s\n"%(wparent,wcomp))
    ar.write("Best  Parent\Composition: %s\%s\n"%(bparent,bcomp))
    ar.write("Mean Iterations:\n%d\n"%mit)
    ar.write("Iterations:\n%s\n"%iterations)
    ar.write("Filename: %s |Time limit: %d |Instances: %d |Mutation Rate=%d |k=%d |Population Size=%d |Objective Value= %.12f |File Version: %s |\n"%(filename,timelimit,it,mutation_rate,k,popsize,optmse,fileversion))
    ar.write("GA type: %s | CPLEX model utilized: %s\n"%(gatype,cplex))
    ar.close()
    print("File created: "+str(filename[:-4])+"_"+str(k)+"_"+str(timelimit)+"_results_"+cplex[0]+".txt")
