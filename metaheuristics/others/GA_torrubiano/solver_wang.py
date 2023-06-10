import numpy as np 
import cplex as cp
class Solve:
    def __init__(self,T,r):
        self.assets,self.t,self.ub,self.lb=r.shape[0],0,1,0.001  
        self.insample = T
        self.outsample = 290 - T #assets counter, it includes the index
        self.r = r
        

    def opt(self):
        rx=self.r.copy()
        rx=rx.tolist()
        print("\n Configuring the relaxed model optimization.")
        c=cp.Cplex()
        var=["1q"+str(i) for i in range(self.insample)]+["2q"+str(i) for i in range(self.insample)]+["x"+str(i) for i in range(self.assets-1)]       #declaring variables
        indices=c.variables.add(names=var)                              #remember that i[1] is refered by i[0]
        c.objective.set_sense(c.objective.sense.minimize)
        c.parameters.timelimit.set(5400)
        c.objective.set_linear((i,1/self.insample) for i in range(2*(self.insample)))           #in the objective function,we add only q+ and q-


        rows=[[var,[0 for i in range(1+t)]+[1]+[0 for i in range(self.insample-1)]+[-1]+[0 for i in range(self.insample-1-t)]+[rx[i+1][t] for i in range(self.assets-1)]]for t in range(self.insample)]+[
              [var,[0 for i in range(2*self.insample)]+[1 for i in range(self.assets-1)]]]+[                                                            #percentage constraint
              [var,[0 for i in range(2*self.insample+j)]+[1]+[0 for i in range(self.assets-j-2)]]for j in range(self.assets-1)]+[                            #lower bounds
              [var,[0 for i in range(2*self.insample+j)]+[1]+[0 for i in range(self.assets-j-2)]]for j in range(self.assets-1)]                              #upper bounds
 
        senses=["E" for i in range(self.insample+1)]+["G" for i in range(self.assets-1)]+["L" for i in range(self.assets-1)]
        rhs=[self.r[0][t-1]for t in range(self.insample)]+[1]+[self.lb for i in range(self.assets-1)]+[self.ub for i in range(self.assets-1)]
        names=["c"+str(i) for i in range(len(rows))]

        indices=c.linear_constraints.add(lin_expr=rows,senses=senses,rhs=rhs,names=names)

        c.set_log_stream(None)
        c.set_error_stream(None)
        c.set_warning_stream(None)
        c.set_results_stream(None)
        t0=c.get_time()
        c.solve()
        t1=c.get_time()
        sol=c.solution.get_objective_value()
        comp=c.solution.get_values()
        sol_comp=comp[2*self.insample:]
        solutiontime=t1-t0
        return sol,sol_comp         

    def solving(self,parent,ctrl=0):            #0 returns the solution compoition, 1 returns the objective value
        tparent=parent.copy()
        parent=self.tencode(parent)
        rx=self.adjust(tparent)
        #print("\n Configuring the relaxed model optimization.")
        c=cp.Cplex()
        var=["1q"+str(i) for i in range(self.insample)]+["2q"+str(i) for i in range(self.insample)]+["x"+str(i) for i in range(len(parent))]       #declaring variables
        indices=c.variables.add(names=var)                             
        c.objective.set_sense(c.objective.sense.minimize)
        c.parameters.timelimit.set(5400)
        c.objective.set_linear((i,1/self.insample) for i in range(2*(self.insample)))           #in the objective function,we add only q+ and q-
        rows=[[var,[0 for i in range(t)]+[1]+[0 for i in range(self.insample-1)]+[-1]+[0 for i in range(self.insample-1-t)]+[rx[i][t] for i in range(len(parent))]]for t in range(self.insample)]+[
              [var,[0 for i in range(2*self.insample)]+[1 for i in range(len(parent))]]]+[                                                            #percentage constraint
              [var,[0 for i in range(2*self.insample+j)]+[1]+[0 for i in range(len(parent)-j-1)]]for j in range(len(parent))]+[                            #lower bounds
              [var,[0 for i in range(2*self.insample+j)]+[1]+[0 for i in range(len(parent)-j-1)]]for j in range(len(parent))]                              #upper bounds
        senses=["E" for i in range(self.insample+1)]+["G" for i in range(len(parent))]+["L" for i in range(len(parent))]
        rhs=[self.r[0][t]for t in range(self.insample)]+[1]+[self.lb for i in range(len(parent))]+[self.ub for i in range(len(parent))]
        names=["c"+str(i) for i in range(len(rows))]
        indices=c.linear_constraints.add(lin_expr=rows,senses=senses,rhs=rhs,names=names)

        c.set_log_stream(None)
        c.set_error_stream(None)
        c.set_warning_stream(None)
        c.set_results_stream(None)
        t0=c.get_time()
        c.solve()
        t1=c.get_time()
        sol=c.solution.get_objective_value()
        comp=c.solution.get_values()
        sol_comp=comp[2*self.insample:]
        solutiontime=t1-t0
        if ctrl==0:
            return sol_comp
        if ctrl==1:
            return sol

    def tencode(self,parent):
        control=[]
        for i in range(len(parent)):
            if parent[i]!=0:
                control.append(i+1)
        return control

    def adjust(self,parent):
        if len(parent)!=self.assets-1:
            print("Parent in adjust method not using binary encoding.The parent have %d and the file have %d"%(len(parent),self.assets-1))
            return -1
        else:
            control=[]
            for i in range(len(parent)):
                if parent[i]!=0:
                    control.append(i+1)
            r,x=[],[]
            for i in control:
                r.append(self.r[i])
            return r

    def get_r(self):
        return self.r.copy()


    def get_lenght(self):
        return self.assets-1

    def get_id(self):
        return 1





