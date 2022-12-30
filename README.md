# Monte-Carlo-simulations-for-exotic-options

## C++
Respository contains variety of functions for calculating the price of numerous different exotic options such as Asian options, in and out barrier options and lookback options using Monte Carlo methods. Also contains a function for implementing Heston's model for stochastic volatility to generate random paths of underlying stock and contains analytic solutions to Black Scholes equations for vanilla european calls.

There is also an implementation of the Binomial pricing model to price American options with early exercise.

## Python
Repository also contains Monte Carlo pricing for methods for a variety of multi-asset options including some rainbow and spread options implemented in Python. Correlated Brownian motions of underlyings are generated using Cholesky decomposition methods and numpy's multivariable random normal generator. The Python file also contains functionality to plot implied volatility curves implied by Heston stochastic volatility, an example of which is given below for underlying at 100 for t=0. To calculate implied volatility I have used the Newton-Raphson method. There are also implementations of antithetic and control variates to reduce variance and increase convergence speed for vanilla european options.

![volatiltiy smile](https://user-images.githubusercontent.com/91262171/182554043-f5fbd234-742b-4997-ac37-92865017a36e.png)

### Dependencies
NumPy

