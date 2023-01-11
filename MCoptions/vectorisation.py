import numpy as np
from scipy.linalg import cholesky 

class rainbow_option:
    
    def __init__(self, S_0, K, r, T, sigma, corr_mat, Type):
        '''
        S_0: array or np array of underlying prices
        corr_mat: array or np array of correlation matrix of assets
        '''
        self.S_0=np.array(S_0) # Initial price array
        self.K=K # Strike
        self.T=T # Time to expiration
        self.r=r # risk free interest rate
        self.sigma=sigma # volatility initial volatility if stochastic
        if Type.lower()!='call' or Type.lower()!='put':
            raise Exception("Invalid option type, try 'put' or 'call'.")
        self.Type=Type.lower()
        self.corr=np.array(corr_mat)
        
    def MC_price(self, sim_no=500, steps=200):
        ''' Vectorised implementation'''
        tot_payoff=0
        dt=self.T/steps
        S_T=np.ones((sim_no, len(self.S_0)))*self.S_0.T 
        for i in range(steps):
            phis=np.random.multivariate_normal(np.zeros(len(self.S_0)), self.corr, size=sim_no)
            S_T=S_T*np.exp((self.r-0.5*self.sigma**2)*dt+(self.sigma*dt**0.5*phis))
        for path in range(sim_no):
            maximum=np.max(S_T[path])
            if self.Type=='call':
                tot_payoff+=max(maximum-self.K,0)
            else:
                tot_payoff+=max(self.K-maximum,0)
                    
        return np.exp(-self.r*self.T)*(tot_payoff/sim_no) 
             
        


class option_basket:
    '''payoff determined by weighted sum of stocks difference from strike'''
    
    def __init__(self, S_0, K, r, T, sigma, corr_mat, weights, Type):
        '''
        S_0: array or np array of underlying prices
        corr_mat: array or np array of correlation matrix of assets
        weigths: array or np array of weights of each underlying
        '''
        
        self.S_0=np.array(S_0) # Initial price array
        self.K=K # Strike
        self.T=T # Time to expiration
        self.r=r # risk free interest rate
        self.sigma=sigma # volatility initial volatility if stochastic
        if Type.lower()!='call' and Type.lower()!='put':
            raise Exception("Invalid option type, try 'put' or 'call'.")
        self.Type=Type.lower()
        self.weights=np.array(weights)
        self.corr=np.array(corr_mat)
        
    
    def MC_price_cholesky(self, sim_no=500, steps=200):
        ''' Will vectorise this in the future'''
        tot_payoff=0
        dt=self.T/steps
        M=cholesky(self.corr, lower=True)
        for i in range(sim_no):
            S=self.S_0
            for j in range(steps):
                epsilon=np.random.normal(0,1,size=(len(self.S_0),))
                phis=M@epsilon
                S=self.S_0*np.exp((self.r-0.5*self.sigma**2)*dt+(self.sigma*dt**0.5*phis))
            if self.Type=='call':
                tot_payoff+=max(np.sum(S*self.weights)/np.sum(self.weights)-self.K,0)
            else:
                tot_payoff+=max(self.K-np.sum(S*self.weights/np.sum(self.weights)),0)
        mean=tot_payoff/sim_no
        return np.exp(-self.r*self.T)*mean
    
    def MC_price(self, sim_no=500, steps=200):
        ''' Vectorised implementation using inbuilt  np multivariate normal dist'''
        tot_payoff=0
        dt=self.T/steps
        S_T=np.ones((sim_no, len(self.S_0)))*self.S_0.T
        for i in range(steps):
            phis=np.random.multivariate_normal(np.zeros(len(self.S_0)), self.corr, size=sim_no)
            print(phis.shape)
            S_T=S_T*np.exp((self.r-0.5*self.sigma**2)*dt+(self.sigma*dt**0.5*phis))
        weights2=np.array([self.weights for i in range(sim_no)])
        prod=S_T*weights2
        prod/=np.sum(self.weights)
        sums=np.sum(prod, axis=1)
        for path in range(sim_no):
            if self.Type=='call':
                tot_payoff+=max(sums[path]-self.K,0)
            else:
                tot_payoff+=max(self.K-sums[path],0)
        mean=tot_payoff/sim_no
        return np.exp(-self.r*self.T)*mean
                           
        
s=option_basket([100,150], 90, 0.02, 1, 0.05, [[1,0.7],[0.7,1]], [1,1], 'call')
print(s.MC_price_cholesky())

