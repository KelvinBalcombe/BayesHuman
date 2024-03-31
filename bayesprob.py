import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from copy import copy
from scipy.stats import norm,beta,binom,poisson,nbinom,gamma,loggamma,lognorm,uniform,rankdata
ln=np.log; exp=np.exp; frame=pd.DataFrame; shape=np.shape; concatenate=np.concatenate
from scipy.special import gammaln as lgm

symbols=[r'$a$',r'$n$',r'$N$',r'$x$',r'$x/N$',r'$λ$',r'$π_{r}$',r'$π_{0}$',r'$a_{0}$',r'$δ$',r'$p$' + '-' + r'$π_{s}$']
symbols2=['p',r'$a_{0}$',r'$n_{0}$',r'$λ_{0}$',r'$θ$', r'$δ_{0}$',r'$δ_{1}$',r'$ϕ$', r'$n_r$',r'$a_r$',r'$π_{0}$']
symbols3=[r'$a$',r'$n$',r'$N$',r'$x$',r'$x/N$',r'$λ$',r'$p$']
symbols4=[r'$a$',r'$n$',r'$λ$',r'$lnN$',r'$π_{r}$','loglik']

#This is a directory if to be saving plots otherwie it should be hashed out
direcout='C:\\Users\\aes05kgb\\Kelvin\\Kel1\\Risk\\warping\\plots\\heuristic\\'

#-------------------------------------------------------------------------------------------------       
    
#Some general functions that make things easier

def cc(ob):
    if type(ob[0])==pd.core.frame.DataFrame or type(ob[0])==pd.core.series.Series:
        return pd.concat(ob,axis=1)
    else:
        return concatenate(ob,axis=1)

def pltsize(x1=10,x2=16):
    plt.rcParams['figure.figsize'] = [x2, x1]
    return

def rows(x):
    return shape(f(x))[0]

def f(x):
    if type(x) !=np.ndarray:
        x=np.array(x)
    x=twodm(x)
    return x

def twodm(x): 
    s=shape(x)
    if len(s)>2:
        x=np.squeeze(x)
        s=shape(x)    
    if len(s)==1:
        x=x.reshape(s[0],1)
    if len(s)==0:
        x=x.reshape(1,1)
    return x

def ones(ob):
    return twodm(np.ones(ob))


#------------------------------------------------------------------------------------------------- 

#An object that gives the quantities given a singular values. These wil depende on the functions below
class bayes():
    def __init__(self,p=1/2,a_0=1/2,n_0=2,lamda_0=1/2,theta=1/2,phi=1/2,delta=[0.5,0],n_r=2,a_r=1/2,iters=100,kappa=0.1):
        self.p=p
        self.a_0=a_0
        self.n_0=n_0
        self.lamda_0=lamda_0
        if self.lamda_0==1:
           self.lamda_0=1-10**-16
        if self.lamda_0==0:
           self.lamda_0=10**-16
        self.theta=theta
        self.delta=delta
        self.phi=phi
        self.n_r=n_r
        self.a_r=a_r
        self.iters=iters
        self.kappa=0.1
        
    #The receiver's prior expected value of the stated probability  
    def pie_0(self):
        return self.lamda_0*self.a_0+(1-self.lamda_0)*self.a_r
    
    #The perceived value of the stated probability under potential dishonesty    
    def pie_s(self):
        pies,sup,delta=disc(self.p,self.delta,self.pie_0(),self.kappa,self.iters)
        return pies
    
    def delta_(self):
        pies,sup,delta=disc(self.p,self.delta,self.pie_0(),self.kappa,self.iters)
        return delta
    
    #The degree of surprise about the stated probability
    def surprise(self):
        return self.pie_s()-self.pie_0()
    
    #The posteror solution for the prior mean of the Source
    def a(self):
        a=find_a(self.a_0,self.theta,self.pie_s(),self.pie_0(),self.surprise()<0)
        return a
    
    #The posterior solution for the value of lamda (n/ (n+N))
    def lamda(self):
        lam=find_l(self.lamda_0,self.theta,self.pie_s(),self.pie_0(),self.surprise()<0)
        return lam
    
    #This is an alternative calculation just as a check
    def lAmda(self):
        vl,vu,v=R_limits(self.lamda_0,self.a_0,self.pie_0(),self.surprise(),self.surprise()<0)
        s=self.surprise()<0
        vl_=v*self.pie_s()/self.a()
        vu_=v*(1-self.pie_s())/(1-self.a())
        lam=s*vl_+(1-s)*vu_
        return float(np.squeeze(lam))
    
    #The posterior solution for the value of n (the strength of the prior of the Source)
    def n(self):
        n,N=Nandn(self.lamda(),self.lamda_0,self.phi,self.n_0)
        return n
    
    #N is the total evidence of the Source (the ex-post perception of the receiver)
    def N(self):
        n,N=Nandn(self.lamda(),self.lamda_0,self.phi,self.n_0)
        return N
    
    #X is the favourable evidence of the Source (the ex-post perception of the receiver)
    def X(self):
        x=fx(self.N(),self.pie_s(),self.n(),self.a())
        return x
    
    #XoN is the ratio of favourable evidence (the ex-post perception of the receiver)
    def XoN(self):
        return float(self.X()/self.N())
    
    #prob is the expected probability of the receiver (the ex-post expectation)
    def prob(self):
        return men(self.a_r,self.n_r,self.N(),self.X())
    
    #The beta parameters ex-post distribution of the reciever 
    #If r is true it is about their own ex-post distribution about the probability
    #If r is false it is about their expost belief of the distribution of the source posterior
    #post signifies ex-post as opposed to ex-anted
    def alpha_beta(self,r=True,post=True):
        if r and post:
            return alpha_beta(self.a_r,self.n_r,self.N(),self.X())
        
        if r and post==False:
            return alpha_beta(self.a_r,self.n_r) 
            
        if r==False and post:
            return alpha_beta(self.a(),self.n(),self.N(),self.X())
        
        if r==False and post==False:
            return alpha_beta(self.a_0,self.n_0)
        
    
    #The beta distribution distributions of the receiver
    def beta(self,r=True,post=True):
        a,b=self.alpha_beta(r=r,post=post)
        return beta(a,b)
    
    #create a subjective opinion from the beta distribution
    def opinion(self,r=True,post=True,base=0.5):
        return opinionf(self.alpha_beta(r=r,post=post),base=base)
    
    #A graph of the distributions of the receiver
    def graph(self,r=True,post=True,ax=0,lab='Ev'):
        if r==True and post==True:
            title="Receiver's Ex-Post Posterior"
        if r==False and post==True:
            title='Ex-post Posterior attributed to the Source'
        if r==True and post==False:
            title="Receiver's Prior"
        if r==False and post==False:
            title='Prior attributed to the Source'
        df=graph_beta(self.alpha_beta(r=r,post=post),lab=lab)
        s=float(df.var().iloc[0])
        if ax==0:
            if s>10**-3:
                return df.plot(grid=True,title=title)
            else:
                return df.plot(grid=True,title=title,ylim=(0,2))
        else:
            if s>10**-3:
                return df.plot(grid=True,title=title,ax=ax)
            else:
                return df.plot(grid=True,title=title,ylim=(0,2),ax=ax)
    
    #The densities of the log-ddds        
    def graph_lodds(self,r=True,post=True,ax=0,lab='Ev'):
        if r==True and post==True:
            title="Receiver's Ex-Post Posterior"
        if r==False and post==True:
            title='Ex-post Posterior attributed to the Source'
        if r==True and post==False:
            title="Receiver's Prior"
        if r==False and post==False:
            title='Prior attributed to the Source'
        df=graph_lodds(self.alpha_beta(r=r,post=post),lab=lab)
        s=float(df.var().iloc[0])
        if ax==0:
            if s>10**-3:
                return df.plot(grid=True,title=title)
            else:
                return df.plot(grid=True,title=title,ylim=(0,2))
        else:
            if s>10**-3:
                return df.plot(grid=True,title=title,ax=ax)
            else:
                return df.plot(grid=True,title=title,ylim=(0,2),ax=ax)
    
    #Put all 4 key beta plots in one graph
    def graphs(self):
        labs=[r'$a_{0}$',r'$π_{s}$',r'$a_{r}$',r'$π_{r}$']
        fig,ax=plt.subplots(2,2,sharex=True);
        fig.suptitle('Beta Distributions')
        g=[]
        i,k=0,0
        for r in [False,True]:
            j=0
            for post in [False,True]:
                self.graph(r=r,post=post,ax=ax[i,j],lab=labs[k])
                j+=1
                k+=1
            i=i+1
        return
    
    def graphs_lodds(self):
        labs=[r'$a_{0}$',r'$π_{s}$',r'$a_{r}$',r'$π_{r}$']
        fig,ax=plt.subplots(2,2,sharex=True);
        fig.suptitle('Log-Odds Distributions')
        g=[]
        i,k=0,0
        for r in [False,True]:
            j=0
            for post in [False,True]:
                self.graph_lodds(r=r,post=post,ax=ax[i,j],lab=labs[k])
                j+=1
                k+=1
            i=i+1
        return
    
    #Produces a summary dataframe of the ex-post quantities
    def expost(self):
        out=[self.a(),self.n(), self.N(), float(self.X()), self.XoN(), self.lamda(),self.prob(),self.pie_s(),self.delta_()]
        out=frame(out).T
        symb=symbols[:7]+[r'$π_{s}$']+[r'$δ$']
        out.columns=symb
        
        out.index=['ex-post']
        return out
    
    #Produces a summary dataframe of the intial quantities- the ex-ante priors and the parameters of the receiver
    def initial(self):
        out=[self.p,
        self.a_0,
        self.n_0,
        self.lamda_0,
        self.theta,
        self.delta[0],
        self.delta[1],     
        self.phi,
        self.n_r,
        self.a_r,
        self.pie_0()]
        out=frame(out).T
        out.columns=symbols2
        out.index=['parameters']
        return out
    
    
    
#-------------------------------------------------------------------------------------------------   
            
#The following produces a dataframe with the important quantities for the graphs in the paper. 
#fg is a dictionary with the parameters
def bayesprob(fg,e=3):
    lamda_0=fg['lamda_0'];
    a_0=fg['a_0'];n_0=fg['n_0'];theta=fg['theta'];delta=fg['delta'];phi=fg['phi'];a_r=fg['a_r'];n_r=fg['n_r']
    pie=f(np.linspace(10**-e,1-10**-e,1000))
    pie_0=lamda_0*a_0+(1-lamda_0)*a_r
    pies,sup,prob=disc(pie,delta,pie_0)
    surprise=abs(pies-pie_0)
    s=pies<pie_0 
    vl,vu,v=R_limits(lamda_0,a_0,pie_0,surprise,s)
    a=find_a(a_0,theta,pies,pie_0,s)
    vl_=v*pies/a
    vu_=v*(1-pies)/(1-a)
    lamda=s*vl_+(1-s)*vu_
    n,N=Nandn(lamda,lamda_0,phi,n_0)
    x=fx(N,pies,n,a)
    XoN=x/N 
    mypie=men(a=a_r,n=n_r,N=N,X=x)
    V=frame(cc([f(a),f(n),f(N),f(x),f(XoN),f(lamda),mypie,pie_0*ones(1000),a_0*ones(1000),prob,mypie-pie]))
    V.columns=symbols
    V.index=np.linspace(10**-e,1-10**-e,1000)
    return V

def labels(fg):
    lamda_0=fg['lamda_0'];    a_0=fg['a_0'];n_0=fg['n_0'];theta=fg['theta'];delta=fg['delta'];phi=fg['phi'];a_r=fg['a_r'];n_r=fg['n_r']
    settings=r'$λ_0$' +'='+ str(lamda_0)+' , ' +r'$n_{0}$'+'='+ str(n_0) +' , ' +r'$a_{0}$'+'='+ str(a_0) +' , ' +r'$θ$'+'='+ str(theta) 
    settings+=' , '+ r'$ϕ$'+'='+ str(phi)   +' , '+ r'$δ$'+'='+ str(delta)+' , ' +  r'$n_r$'+'='+ str(n_r)+' , '  r'$a_r$'+'='+ str(a_r)
    return settings

def graphout(V,fg,fig,ax):
    pie_=r'$π_{s}$'
    lamda_0=fg['lamda_0'];
    a_0=fg['a_0'];n_0=fg['n_0'];theta=fg['theta'];delta=fg['delta'];phi=fg['phi'];a_r=fg['a_r'];n_r=fg['n_r']
    V[[r'$x/N$', r'$π_{r}$']].plot(grid=True,ax=ax[0],xlabel=pie_,fontsize=12)
    V[[r'$N$']].plot(grid=True,ax=ax[1],xlabel=pie_,fontsize=12)
    V[[r'$n$']].plot(grid=True,ax=ax[2],xlabel=pie_,fontsize=10)
    V[[r'$a$']].plot(grid=True,ax=ax[3],xlabel=pie_,fontsize=12,ylim=(0,1))
    V[[r'$a_{0}$']].plot(grid=True,ax=ax[3],xlabel=pie_,fontsize=12,ylim=(0,1))
    V[[ r'$p$'+ '-' + r'$π_{s}$']].plot(grid=True,ax=ax[4],xlabel=pie_,fontsize=12,ylim=(-.5,.5))
    settings=labels(fg)
    fig.suptitle(settings, y=1.1, fontsize=18  )
    return


#------------------------------------------------------------------------------------------------- 
#Below are the functions that are used in the object and the function to produce the graphs

def R_limits(lamda0,a0,pie_0,surprise,s):
    T=ones(rows(surprise))
    vl=T*lamda0*a0/(pie_0)            
    vu=T*lamda0*(1-a0)/(1-pie_0)  
    v=s*vl+(1-s)*vu
    return vl,vu,v

#The decomposition to find a
def find_a(a0,theta,pies,pie_0,s):
    a_l=a0*(pies/pie_0)**theta
    a_u=1-(1-a0)*((1-pies)/(1-pie_0))**theta
    a=s*a_l+(1-s)*a_u
    return a 

#The decomposition to find lamda
def find_l(lamda_0,theta,pies,pie_0,s):
    l_l=lamda_0*(pies/pie_0)**(1-theta)
    l_u=lamda_0*((1-pies)/(1-pie_0))**(1-theta)
    lam=s*l_l+(1-s)*l_u
    return lam 

#The decomposition between n and N
def Nandn(lamda1,lamda0,phi,n0):
    r=(1/lamda1-1)
    tau0=(1/lamda0-1)
    k=n0*(tau0**(1-phi))
    N=k*r**phi
    n=k*r**(phi-1)
    return n,N

#The favourable evidence based on the mean prob
def fx(N,pie,n=2,a=0.5):
    return (n+N)*pie-n*a

#The expected probability
def men(a,n,N,X):
    return (n*a+X)/(n+N)


#---------------------------------------------------------------------------------------------
#The following functions relate to working out the posterior mean attributed to the source under dishonesty
def disc(pie,delta,pie_0,kappa=0.1,iters=100): 
    sup=pie-pie_0
    #sup=ln(pie/pie_0)
    h_kappa=3.99*(1+kappa)/(1-kappa)                                    #Estimation of delta      
    delta_=logit(delta[0],delta[1],sup,b=h_kappa) 
    pie_bar_0=pie_bar(pie_0=pie_0,delta_=delta_,kappa=kappa)  #The intial ex-ante PMAS 
   
    #The following iteration recomputes delta based on the updated suprise
    nits=0
    if delta[1] !=0:
        for i in range(iters):
            sup=pie-pie_bar_0                                                       #Surprise
            #sup=ln(pie/pie_bar_0) 
            deltav=delta_
            delta_=logit(delta[0],delta[1],sup,b=h_kappa)                  #Estimation of delta 
   
            pie_bar_0=pie_bar(pie_0=pie_0,delta_=delta_,kappa=kappa)       #The reciever's ex-ante expectation of the dishonest view
            nits+=1
        
     #If dealing with a vector values for the graphs do the full 100 iterations otherwise check for convergence earlier
     #In most cases less than 25 iterations needed reach tolerance below
            try:
                length=len(delta_)
            except:
                length=1
            if length==1:
                if abs(deltav-delta_)<10**-4:
                   break
     
    
    pie_i, pie_d,pie_s=pie_actual(pie,delta_=delta_,kappa=kappa)
    #print(nits)
    return pie_s,sup,delta_

#This calculates the posterior mean attributed to the source based on their stated probability under dishonesty, for a given delta
#This is used in disc above recursively with updated deltas from that recursion
def pie_bar(pie_0,delta_=0.5,kappa=0.1):
    a=delta_+(1-delta_)*kappa
    b=1-delta_*(1-kappa)
    c=(1-kappa)*(1-2*delta_)
    return a*pie_0 /(b-c*pie_0)

def logit(v0,v1,x,b=1):
    c=ln(v0)-ln(1-v0)
    y=exp(c+b*v1*x)
    return y/(1+y)

#The inflated, deflated probabilities along with their "average"
#delta is a derived from a logit probability in the paper
#pie_s is the true ex-post mean attributed to the Source (pie_s)
def pie_actual(pie,delta_=0.5,kappa=0.1):
    om_i, om_d=omega(pie,kappa)
    pie_i=pie/om_i+(om_i-1)/om_i
    pie_d=pie/om_d
    pie_s=(pie-delta_*(1-om_i))/(delta_*om_i+(1-delta_)*om_d)
    return pie_i, pie_d,pie_s

#pie is the stated probability.
#kappa is generally set at a point such as 0.1 and fixed from the point of view of the analysis
def omega(pie,kappa):
    om_i=kappa+(1-kappa)*(1-pie)
    om_d=kappa+(1-kappa)*pie
    return om_i, om_d



#-------------------------------------------------------------------------------------------

def sequential(lst1,a):
    V=[]
    for i in lst1:
        a.p=i
        V+=[[a.a(),a.n(), a.N(), a.X(), a.XoN(), a.lamda(),a.prob()]] 
        p=a.prob()
        a.n_r+=a.N()
        a.a_r=a.prob()
    V=frame(V)
    V.columns=symbols3
    return V,p      

def alpha_beta(a,n,N=0,x=0):
    return a*n+x,(1-a)*n+N-x

def graph_beta(beta_param,increments=1000,scale=1,lab='Ev',standard=True):
    alp=beta_param[0]
    bet=beta_param[1]
    x=np.linspace(.001,.999,increments)
    f=frame(beta(alp,bet).pdf(x))
    f.index=x*scale
    m=round(alp/(alp+bet),5)
    if standard:
        f.columns=[r'$α$'+'='+str(alp)[0:6]+ ', '  + r'$β$'+'='+str(bet)[0:6]+', ' + lab + '='+ str(m)[0:6]]
    else:
        f.columns= ['mean_odds=' +str(m/(1-m))[0:6]]
    return f

def graph_lodds(beta_param,increments=1000,scale=1,lab='Ev'):
    fg=graph_beta(beta_param,lab=lab,standard=False)
    c=fg.columns
    fg.columns=['a']
    x=np.linspace(.001,.999,increments)
    
    odds=np.log(x)-np.log(1-x)
 
    fg.index=odds
    z=f(1-x)
    #print(np.shape(z),np.shape(f(fg['a'])))
    fg['a']=f(fg['a'])*z*(1-z)
    fg.columns=c
    
    return fg
                 

def opinionf(beta_param,W=2,base=0.5):
    a=base
    alp,bet=beta_param[0],beta_param[1]
    B=alp/W-a
    D=bet/W-(1-a)
    rB=B/(1+B)
    rD=D/(1+D)
    p=rB*rD
    b=(rB-p)/(1-p)
    d=rD*(1-b)
    u=1-b-d
    return [b,d,u]


#-------------------------------------------------------------------------------------
#The following is the likelihood for the MCMC simulations
def likelihood(x,N,n,n_r,a_r):
    a=n_r*a_r
    b=n_r*(1-a_r)
    f1=np.log(n+N)
    f2=lgm(N+1)
    f3=lgm(N-x+1)
    f4=lgm(x+1)
    f5=lgm(a+b)
    f6=lgm(a)
    f7=lgm(b)
    f8=lgm(N-x+b)
    f9=lgm(x+a)
    f10=lgm(N+a+b)
    f=f1+f2-f3-f4+f5-f6-f7+f8+f9-f10
    return f

def llf(p,lamda,n,a,n_r,a_r):
    N=n*(1-lamda)/lamda
    x=(n+N)*p-n*a
    if x<0 or (x/N)>1:
        lik=-10**16
    else:
        lik=likelihood(x,N,n,n_r,a_r)
    return lik

