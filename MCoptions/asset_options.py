import numpy as np
from scipy.stats import norm
from scipy.linalg import cholesky
from  sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

class option_vanilla:
    
    def __init__(self, S_0, K, T, r, sigma, Type):
        self.S_0=S_0 # Initial price
        self.K=K # Strike
        self.T=T # Time to expiration
        self.r=r # risk free interest rate
        self.sigma=sigma # volatility initial volatility if stochastic
        if Type!='call' and Type!='put':
            raise Exception("Invalid option type, try 'put' or 'call'.")
        self.Type=Type
        if self.Type=='call':
            self.payoff=self.call_payoff
        elif self.Type=='put':
            self.payoff=self.put_payoff
        # Following quantities are useful for black scholes
        self.d1=np.log(self.S_0/self.K)+(self.r+(self.sigma**2/2)*self.T)/(self.sigma*self.T**0.5)
        self.d2=self.d1-self.sigma*self.T**0.5
    
    def call_payoff(self, S):
        return max(0, S-self.K)
    
    def put_payoff(self, S):
        return max(0, self.K-S)
    
    def MC_price(self, sim_no=1000):
        tot_payoff=0
        a=(self.r-0.5*(self.sigma**2))*self.T # Solution to geometric brownian motion
        b=self.sigma*(self.T)**0.5
        for i in range(sim_no):
            S_T=self.S_0*np.exp(a+(b*np.random.normal()))
            tot_payoff+=self.payoff(S_T)
        mean=tot_payoff/sim_no
        return np.exp(self.r*self.T)*mean
    
    def Black_Scholes_price(self):
        call_price=self.S_0*norm.cdf(self.d1)-norm.cdf(self.d2)*self.K*(np.exp(-self.r*self.T))
        if self.Type=='call':
            return call_price
        return call_price-self.S_0+self.K # using put call parity to price put
    
    def Heston_path(self, steps):
        dt=self.T/steps
        corr_mat=np.array([[1,self.rho2],[self.rho2,1]])
        M=cholesky(corr_mat, lower=True)
        V_t=self.V_0 # Volatility which is stochastic
        S_T=self.S_0
        for i in range(steps):
            epsilon=np.random.normal(0, 1, size=(2,))
            phis=M@epsilon
            sigma=abs(V_t)**0.5
            S_T=S_T*np.exp((self.r-0.5*sigma**2)*dt+(sigma*dt**0.5*(phis[0])))
            V_t=abs(V_t)*self.kappa*(self.theta-abs(V_t))*dt+(abs(V_t)**0.5)*phis[1]*dt**0.5
        return S_T   

    def stochastic_vol(self, theta, rho2, kappa, V_0):
        # If we want to use stochastic vol we call this function on object to initialis parameters
        self.theta=theta
        self.rho2=rho2
        self.kappa=kappa
        self.V_0=V_0
        self.sigma=None # We remove notion of constant sigma
        
    def stochastic_MC_price(self, sim_no=1000, steps=300):
        tot_payoff=0
        for i in range(sim_no):
            S_T=self.Heston_path(steps)
            tot_payoff+=self.payoff(S_T)
        mean=tot_payoff/sim_no
        return mean*np.exp(-self.r*self.T)
    
    '''
    We also include functionality for common greeks, implementing analytic methods for black scholes form
    and numerical methods for stochastic volatility
    '''
    def delta(self):
         if self.Type=='call':
             return norm.cdf(self.d1)
         return norm.cdf(self.d1)-1
     
    def gamma(self):
        N_dash=np.exp(-self.d1**2/2)/np.sqrt(2*np.pi)
        return N_dash/(self.S_0*self.sigma*self.T**0.5)
    
    def vega(self):
        N_dash=np.exp(-self.d1**2/2)/np.sqrt(2*np.pi)
        return self.S_0*np.sqrt(self.T)*N_dash
    
    def theta(self):
        N_dash=np.exp(-self.d1**2/2)/np.sqrt(2*np.pi)
        if self.Type=='call':
            a=self.S_0*N_dash*self.sigma/(2*self.T**0.5)
            b=self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(self.d2)
            c=self.r*self.K*np.exp(-self.r*self.T)*norm.cdf(-self.d2)
            if self.Type=='call':
                return -a-b
            else:
                return -a+c
    def rho(self):
        if self.Type=='call':
            return self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(self.d2)
        else:
            return -self.K*self.T*np.exp(-self.r*self.T)*norm.cdf(-self.d2)
        
        
    '''
    Numerical finite difference approximations of the greeks
    for stochastic volatility
    '''
     
    
    def stoch_delta(self, ds=1, sim_no=2000):
        ''' 
        Numerically calculates delta of stochastic vol model
        using finite difference
        '''
        self.S_0+=ds
        a=self.stochastic_MC_price(sim_no, steps=1000) # long simulations to get accurate estimates
        self.S_0-=ds
        b=self.stochastic_MC_price(sim_no, steps=1000)
        return (a-b)/ds
    
    def stoch_gamma(self, ds=1, sim_no=2000):
        self.S_0+=ds
        a=self.stochastic_MC_price(sim_no, steps=1000)
        self.S_0-=ds
        b=self.stochastic_MC_price(sim_no, steps=1000)
        self.S_0-=ds
        c=self.stochastic_MC_price(sim_no, steps=1000)
        return (a-2*b+c)/ds**2
    
    def stoch_theta(self, dt=1, sim_no=2000):
        self.T+=dt
        a=self.stochastic_MC_price(sim_no, steps=1000)
        self.T-=dt
        b=self.stochastic_MC_price(sim_no, steps=1000)
        return (b-a)/dt
    
    def stoch_rho(self, dr=1, sim_no=2000):
        self.r+=dr
        a=self.stochastic_MC_price(sim_no, steps=500)
        self.r-=dr
        b=self.stochastic_MC_price(sim_no, steps=500)
        return (b-a)/dr
    
    
    def BS_summary(self):
        print('\n')
        print('########################################\n')
        print('Price: ', self.Black_Scholes_price())
        print('Delta: ', self.delta())
        print('Gamma: ', self.gamma())
        print('Theta: ', self.theta())
        print('Vega: ', self.vega())
        print('Rho: ', self.rho())
        print('########################################\n')
        print('\n')
    '''
    def add(self, other):
        
        '''''' Can add two options to create a new option spread ''' '''
        new_option=option_vanilla(self.S_0, None, self.r, self.T, self.sigma, 'spread')
        new_option.payoff=self.payoff(self.S_0)+other.payoff(self.S_0)
        return new_option
    '''
    
class deriv_product:
    ''' Combining variety of vanilla options to make time spreads and other derivative products'''
    
    def __init__(self, long, short, ratio=None):
        ''' long : array of long options
           short : array of short options
           ratio : tuple of ratios between options
        '''
        self.long=long
        self.short=short
        if ratio:
            self.ratio=ratio
        else:
            self.ratio=tuple([1 for i in range(len(self.long)+len(self.short))])
     
    ''' Need to find a better way to implement this that actually works'''
    def summ(self, method):
        tot=0
        for i,j in enumerate(self.long):
            tot+=self.ratio[i]*j.method()
        for i,j in enumerate(self.short):
            k=i+len(self.long)
            tot-=self.ratio[k]*j.method()
    
    def price(self):
        ''' 
        Note negative return corresponds to you would make money buying this product,
        equivalent to taking a short position'''
        
        tot=0
        for i,j in enumerate(self.long):
            tot+=self.ratio[i]*j.Black_Scholes_price()
        for i,j in enumerate(self.short):
            k=i+len(self.long)
            tot-=self.ratio[k]*j.Black_Scholes_price()
        return tot
    
    def stochastic_vol(self, theta, rho2, kappa, V_0):
        ''' 
        Allows you to apply stochastic volatiltity attributes to 
        all options in the derivative product at once
        '''
        for i in zip(self.long,self.short):
            i.theta=theta
            i.rho2=rho2
            i.kappa=kappa
            i.V_0=V_0
            i.sigma=None # We remove notion of constant sigma
   
    def stochastic_MC_price(self):
        tot=0
        for i,j in enumerate(self.long):
            tot+=self.ratio[i]*j.stochastic_MC_price()
        for i,j in enumerate(self.short):
            k=i+len(self.long)
            tot-=self.ratio[k]*j.stochastic_MC_price()
        return tot
    
    def delta(self):
        tot=0
        for i,j in enumerate(self.long):
            tot+=self.ratio[i]*j.delta()
        for i,j in enumerate(self.short):
            tot-=self.ratio[i]*j.delta()
        return tot
    
    def vega(self):
        tot=0
        for i,j in enumerate(self.long):
            tot+=self.ratio[i]*j.vega()
        for i,j in enumerate(self.short):
            tot-=self.ratio[i]*j.vega()
        return tot
    
    def theta(self):
        tot=0
        for i,j in enumerate(self.long):
            tot+=self.ratio[i]*j.theta()
        for i,j in enumerate(self.short):
            tot-=self.ratio[i]*j.theta()
        return tot
    
    def gamma(self):
        tot=0
        for i,j in enumerate(self.long):
            tot+=self.ratio[i]*j.gamma()
        for i,j in enumerate(self.short):
            tot-=self.ratio[i]*j.gamma()
        return tot
    
    def rho(self):
        tot=0
        for i,j in enumerate(self.long):
            tot+=self.ratio[i]*j.rho()
        for i,j in enumerate(self.short):
            tot-=self.ratio[i]*j.rho()
            
class Asian_option:
    ''' 
    This will be used as a base class for other types of single asset exotic products
    '''
    def __init__(self, S_0, K, T, r, sigma, ag, Type):
        self.S_0=S_0 # Initial price
        self.K=K # Strike
        self.T=T # Time to expiration
        self.r=r # risk free interest rate
        self.sigma=sigma # volatility initial volatility if stochastic
        if ag!='a' and ag!='g':
            raise Exception('Input a or g for arithmetic and geometric asian options respectively')
        if Type!='call' and Type!='put':
            raise Exception("Invalid option type, try 'put' or 'call'.")
        self.Type=Type
        self.ag=ag
        
    ''' could make a create paths method'''
    def create_paths(self, sim_no, steps, dt):
        a=(self.r-0.5*(self.sigma**2))*dt
        b=self.sigma*dt**0.5
        paths=np.ones((sim_no, steps))*self.S_0
        for i in range(1, steps):
            interm=np.random.normal(size=(sim_no,))*b+a
            paths[:,i]=paths[:,i-1]*np.exp(interm)
        return paths
    
    def MC_price(self, sim_no=1000, steps=100):
        '''
        Employed vectorised method to simulate all the paths
        for efficiency
        '''
           
        tot_payoff=0
        dt=self.T/steps
        paths=self.create_paths(sim_no, steps, dt)
        if self.ag=='a':
            totals=np.sum(paths, axis=1)
            means=totals/steps
        else:
            totals=np.prod(paths, axis=1)
            means=totals**(1/steps)
        for j in means:
            if self.Type=='call':
                tot_payoff+=max(0, j-self.K)
            else:
                tot_payoff+=max(0, self.K-j)
        return np.exp(-self.r*self.T)*(tot_payoff/sim_no)
    
    def delta(self, ds=1, sim_no=4000):
        self.S_0+=ds
        a=self.MC_price(sim_no)
        self.S_0-=ds
        b=self.MC_price(sim_no)
        return (a-b)/ds
    
    def gamma(self, ds=1, sim_no=4000):
        self.S_0+=ds
        a=self._MC_price(sim_no)
        self.S_0-=ds
        b=self.MC_price(sim_no)
        self.S_0-=ds
        c=self.MC_price(sim_no)
        return (a-2*b+c)/ds**2
    
    def theta(self, dt=1, sim_no=4000):
        self.T+=dt
        a=self.MC_price(sim_no)
        self.T-=dt
        b=self.MC_price(sim_no)
        return (b-a)/dt
        
        
class lookback_option(Asian_option):
    '''Child class of Asian options to inherit relevant greek methods'''
    def __init__(self, S_0, K, T, r, sigma, Type):
        self.S_0=S_0 # Initial price
        self.K=K # Strike
        self.T=T # Time to expiration
        self.r=r # risk free interest rate
        self.sigma=sigma # volatility initial volatility if stochastic
        if Type!='call' and Type!='put':
            raise Exception("Invalid option type, try 'put' or 'call'.")
        self.Type=Type
    
    def MC_price(self, sim_no=5000, steps=500):
        tot_payoff=0
        dt=self.T/steps
        paths=self.create_paths(sim_no, steps, dt)
        maxes=np.amax(paths, axis=1)
        for j in maxes:
            if self.Type=='call':
                tot_payoff+=max(0, j-self.K)
            else:
                tot_payoff+=max(0, self.K-j)
        return np.exp(-self.r*self.T)*(tot_payoff/sim_no)
    
class barrier_option(Asian_option):
    '''Child class of Asian options to inherit relevant greek methods'''
    def __init__(self, S_0, K, T, r, sigma, inout, barrier, Type):
        self.S_0=S_0 # Initial price
        self.K=K # Strike
        self.T=T # Time to expiration
        self.r=r # risk free interest rate
        self.sigma=sigma # volatility initial volatility if stochastic
        if Type!='call' and Type!='put':
            raise Exception("Invalid option type, try 'put' or 'call'.")
        self.Type=Type
        self.inout=inout
        self.barrier=barrier
        
    def MC_price(self, sim_no=500, steps=200):
        tot_payoff=0
        dt=self.T/steps
        a=(self.r-0.5*(self.sigma**2))*dt
        b=self.sigma*dt**0.5
        for i in range(sim_no):
            S_T=self.S_0
            for j in range(steps):
                 S_T=S_T*np.exp(a+(b*np.random.normal()))
                 if self.inout=='in':
                     if S_T>self.barrier:
                         C=option_vanilla(S_T, self.K, self.T-(j*dt), self.r, self.sigma, self.Type)
                         tot_payoff+=C.Black_Scholes_price()
                         break
                 else:
                     if S_T<self.barrier:
                         C=option_vanilla(S_T, self.K, self.T-(j*dt), self.r, self.sigma, self.Type)
                         tot_payoff+=C.Black_Scholes_price()
                         break
                         
        return np.exp(-self.r*self.T)*(tot_payoff/sim_no)



class american_option:
    # Class implementing functionalities of an American option
    # Will include binomial/dynamic programming pricing approach and LSMC
    def __init__(self, S_0, K, T, r, sigma, Type='put'):
        self.S_0 = S_0  # Initial price
        self.K = K      # Strike
        self.T = T      # Time to expiration
        self.r = r      # Risk-free interest rate
        self.sigma = sigma  # Volatility, initial volatility if stochastic
        if Type != 'call' and Type != 'put':
            raise Exception("Invalid option type, try 'put' or 'call'.")
        self.Type = Type

    def payoff(self, x):
        if self.Type == 'call':
            return np.maximum(x - self.K, 0)
        else:
            return np.maximum(self.K - x, 0)

    def price_path_matrix(self, sim_no, steps):
        prices = np.zeros((sim_no, steps))
        prices[:, 0] = self.S_0
        dt = self.T / (steps - 1)
        for i in range(1, steps):
            rands = np.random.normal(scale=dt**0.5, size=sim_no)
            prices[:, i] = prices[:, i-1] * np.exp((self.r - 0.5*self.sigma**2) * dt + self.sigma * rands)
        return prices

    def binomial_price(self, sim_no, dt):
        steps = int(self.T / dt) + 1
        prices = self.price_path_matrix(sim_no, steps)

        discount = np.exp(-self.r * dt)
        
        vals = np.zeros((sim_no, steps))
        vals[:, -1] = self.payoff(prices[:, -1])

        for i in range(steps - 2, -1, -1):
            node_val = discount * 0.5 * (vals[:, i + 1])
            early_val = self.payoff(prices[:, i])

            vals[:, i] = np.maximum(node_val, early_val)
        return np.mean(vals[:, 0])  # Average the values at t=0 for all paths

    def LSMC_price(self, sim_no=500, dt=0.2, poly_degree=2, verbose=False):
        # Verbose is used to also print the decision matrix
        steps = int(self.T / dt) + 1
        prices = self.price_path_matrix(sim_no, steps)
        decision_matrix = np.zeros((sim_no, steps))
        cashflows = np.zeros((sim_no, steps))

        if self.Type == 'call':
            cashflows[:, -1] = np.maximum(prices[:, -1] - self.K, 0)
        else:
            cashflows[:, -1] = np.maximum(self.K - prices[:, -1], 0)
        decision_matrix[:, -1] = (cashflows[:, -1] > 0).astype(int)

        for i in range(steps - 2, -1, -1):
            # Get path_indices where option is in the money to carry out regression
            if self.Type == 'put':
                path_indices = np.where(prices[:, i] < self.K)[0]
            else:
                path_indices = np.where(prices[:, i] > self.K)[0]

            if path_indices.size > 0:
                X = prices[path_indices, i]
                Y = np.zeros(prices[:, i][path_indices].shape)
                t = 0

                for j in range(i + 1, steps):
                    t += 1
                    Y += cashflows[path_indices, j] * np.exp(-self.r * t * dt)
                
                poly_model = PolynomialFeatures(poly_degree)
                X_poly = poly_model.fit_transform(X.reshape(-1, 1))
                reg = LinearRegression()
                reg.fit(X_poly, Y)
                continuation_vals = reg.predict(X_poly)

                exercise_values = self.payoff(X)
                
                early_indices = np.where(exercise_values > continuation_vals)[0]

                if early_indices.size > 0:
                    cashflows[path_indices[early_indices], i] = exercise_values[early_indices]
                    decision_matrix[path_indices[early_indices], i] = 1
                    for j in range(i + 1, steps):
                        cashflows[path_indices[early_indices], j] = 0
                        decision_matrix[path_indices[early_indices], j] = 0

        discounted_cashflows = np.sum(cashflows * np.exp(-self.r * np.arange(steps) * dt), axis=1)
        
        if verbose:
            print('Decision Matrix:')
            print(decision_matrix)
            print('\n')
            print('Cashflow Matrix: ')
            print(cashflows)

        return np.mean(discounted_cashflows)

                


                
                
                
            
            
        
        
        
        
        
        
