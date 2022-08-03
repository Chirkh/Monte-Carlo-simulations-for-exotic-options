"""
In particular this file contains monte carlo simulations of spread call options and baskets of underlyings.
This file also contains an implementation of stochastic heston volatility and the resulting 
implied volatility smile. To calculate implied vol we use the Newton Raphson method. There are also some 
simple implementations of antithetic and control variates for vanilla european options which can be extended
to more exotic contracts. To generate correlated Brownina motions traditional Cholesky decmoposition methods 
have been used as well as numpy's built in multivariable random normal sampler.
"""

import numpy as np
from scipy.linalg import cholesky
from matplotlib import pyplot as plt
from scipy.stats import norm



def control_variate_european_option(S,K,r,R,sigma,sim_no,steps, call=True, beta):
    '''We will use the underlying itself as a control variate. The correlation between the underlying and option
    value reduces the variance. To minimise the variance beta is taken to be Sxy/Sxx where X is the underlying 
    and Y is the option price we are trying to estimate using monte carlo. Often have to estimate beta itself 
    by running extra simulations and calculating Sxx and Sxy.
    '''
    dt=T/steps
    tot_payoff=0
    E_S=S*np.exp(r*T) # Expected value or mean of asset price at time T
    for i in range(sim_no):
        S_t=S_t*np.exp((r-0.5*sigma**2)*dt+(sigma*dt**0.5*phi))
        if call:
            tot_payoff+=max(S_t-K,0)-beta*(S_t-E_S)
        else:
            tot_payoff+=max(K-S_t,0)-beta*(S_t-E_S)
    return np.exp(-r*T)
            

def antithetic_european_option(S,K,r,T,sigma,sim_no,steps, call=True):
    '''
    Simple implementation of antithetic variates to increase speed of convergence although
    here its won't make too much difference to efficiency. Sampling random increments phi and
    -phi means that we replicate the normal distribution and so convergence occurs quicker.
    '''
    tot_payoff=0
    dt=T/steps
    S_t1=S
    S_t2=S
    for i in range(sim_no):
        phi=np.random.normal()
        S_t1= S_t1*np.exp((r-0.5*sigma**2)*dt+(sigma*dt**0.5*phi))
        S_t2= S_t2*np.exp((r-0.5*sigma**2)*dt+(sigma*dt**0.5*phi))
        if call:
            tot_payoff+=(max(S_t1-K,0)+max(S_t2-K,0))/2 # Add average of both paths payoffs which are inversely correlated
        else:
            tot_payoff+=(max(K-S_t1,0)+max(K-S_t2,0))/2
    return np.exp(-r*T)*(tot_payoff/sim_no)
               


def spread_call_option(S1, S2, K, r, T, sigma, sim_no, steps, correlation_mat, call=True):
    '''
    Application of Monte Carlo for a multi asset spread call option.
    Uses Cholesky decomposition to calculate Wiener terms in accordance to correlation matrix.
    '''
    tot_payoff=0
    dt=T/steps
    M=cholesky(correlation_mat, lower=True)
    for i in range(sim_no):
        S1_t=S1
        S2_t=S2
        for j in range(steps):
            epsilon=np.random.normal(0,1,size=(2,))
            phis=np.matmul(M,epsilon)
            S1_t=S1_t*np.exp((r-0.5*sigma**2)*dt+(sigma*dt**0.5*phis[0]))
            S2_t=S2_t*np.exp((r-0.5*sigma**2)*dt+(sigma*dt**0.5*phis[1]))
        if call:
            tot_payoff+=max(S1_t-S2_t-K,0)
        else:
            tot_payoff+=max(K-(S1_t-S2_t),0)
        
    return np.exp(-r*T)*(tot_payoff/sim_no)
    

def option_basket_call(S ,K, r, T, sigma, sim_no, correlation_mat, weights, call=True):
    '''
    This function calculates the value of a call option on a basket of n stocks which 
    are entered as an array. The correlations between the stocks are handled using Cholesky
    decomposition. Predetermined weigths of stocks in the basket is also inputted as an array
    and the payoff is a simple call/put on the basket whose value is determined as the weighted 
    average at expiry minus the strike.
    '''
    tot_payoff=0
    S=np.array(S)
    weights=np.array(weights)
    dt=T/sim_no
    M=cholesky(correlation_mat, lower=True)
    for i in range(sim_no):
        epsilon=np.random.normal(0,1,size=(len(S),))
        phis=np.matmul(M,epsilon)
        S=S*np.exp((r-0.5*sigma**2)*dt+(sigma*dt**0.5*phis))
        if call:
            tot_payoff+=max(np.sum(S*weights)-K,0)
        else:
            tot_payoff+=max(K-np.sum(S*weights),0)
    return np.exp(-r*T)*(tot_payoff/sim_no)


def heston_vol_path(S, r, T, kappa, theta, V_0, rho, steps):
    '''
    Simulates a path for a given asset following heston's stochastic volatility model using 
    monte carlo methods and cholesky decomposition to generate correlated Wiener increments
    between the stochastic volatility and the stochastic asset itself.
    '''
    dt=T/steps
    correlation_mat=np.array([[1,rho],[rho,1]])
    M=cholesky(correlation_mat, lower=True) # Instead of using Cholesky for correlated stochastic variables we can use numpy multivariable normal function
    S_t=S
    V_t=V_0
    prices=[]
    cov_check=0
    for i in range(steps):
        epsilon=np.random.normal(0,1,size=(2,))
        phis=np.matmul(M,epsilon)
        cov_check+=(phis[0]*phis[1])
        sigma=abs(V_t)**0.5
        S_t=S_t*np.exp((r-0.5*sigma**2)*dt+(sigma*dt**0.5*phis[0]))
        V_t=abs(V_t)+kappa*(theta-abs(V_t))*dt+(abs(V_t)**0.5)*phis[1]*dt**0.5
        prices.append(S_t)
    #print(cov_check/steps)
    return prices


def rainbow_call_on_max(S,K,r,T,sigma,sim_no, correlation_mat, steps):
    '''
    This function exploits numpy arrays and the multivariable normal function to generate correlated 
    Wiener processes. This can be extended to other functions as well for some gain in efficiency. The 
    payoff of this rainbow function for N stocks is max(max(S_1,...,S_N)-K) at expiry
    '''
    dt=T/steps
    S=np.array(S)
    S_t=np.reshape(S, (sim_no, len(S)))
    tot_payoff=0
    for i in range(steps):
        phis=np.random.multivariate_normal(np.zeros(len(S)), np.array(correlation_mat), size=sim_no)
        S_t=S_t*np.exp((r-0.5*sigma**2)*dt+(sigma*dt**0.5*phis))
    for path in range(sim_no):
        maximum=np.max(S_t[path])
        tot_payoff+=max(maximum-K,0)
    return np.exp(-r*T)*(tot_payoff/sim_no)
    

def stochastic_vol_call_price(S,K,T,r, sigma, sim_no, kappa, steps, theta, rho):
    '''
    Function calculates the value of a call option using stochastic heston volatility 
    and monte carlo methods
    '''
    tot_payoff=0
    for i in range(sim_no):
        prices=heston_vol_path(S, r, T, kappa, theta, sigma**2, rho, steps)
        tot_payoff+=max(prices[-1]-K,0)
    return np.exp(-r*T)*(tot_payoff/sim_no)


def volatility_smile(S,T,r,sigma,kappa,theta,sim_no,steps,rho):
    '''
    Function plot the volatility smile implied by Heston volatility
    '''
    f=plt.figure()
    option_prices=[]
    IVs=[]
    Ks=[]
    for K in range(75,200,20):
        Ks.append(K)
        option_prices.append(stochastic_vol_call_price(S, K, T, r, sigma, sim_no, kappa, steps, theta, rho))
    for i in range(len(option_prices)):
        IVs.append(implied_volatility_call(option_prices[i], S, Ks[i], T, r))
    plt.scatter(Ks, IVs)
    plt.show()
        

def black_scholes_call(S, K, T, r, sigma):
    d1=(np.log(S/K)+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    C_price = S*norm.cdf(d1)- norm.cdf(d2)*K*np.exp(-r*T)
    return C_price

def vega(S, K, T, r, sigma):
    d1 = (np.log(S/K)+(r+sigma**2/2)*T)/sigma*np.sqrt(T)
    vega = S*norm.pdf(d1)*T**(0.5)
    return vega

def implied_volatility_call(C, S, K, T, r, diff=0.001,
                            max_iter=100):
    '''
    This function calculates the implied volatility from the option price using the Newton Raphson method
    '''
    sigma = ((2*np.pi/T)**0.5)*C/S # This initial estimate gives good initial estimate and fast covergence.
    for i in range(max_iter):
        if abs((black_scholes_call(S, K, T, r, sigma) - C)) < diff:
            break
        # using newton rapshon to update the estimate
        sigma=sigma-(black_scholes_call(S, K, T, r, sigma)-C)/vega(S, K, T, r, sigma)
    return sigma 
    

volatility_smile(100,0.5,0.05,0.2,3,0.04,5000,1200,-0.8)      
    
        
    
    
    
