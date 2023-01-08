# Monte-Carlo-simulations-for-exotic-options

## MCopt
Python library for pricing variety of options products and other derivative products using Monte Carlo methods and Black-Scholes analytic equations. Library also has functionality for calculating greeks both using finite difference methods and using analytic solutions where possible.

## Core Functionalities
* Pricing vanilla call and put options using:
  1) Black-Scholes equation
  2) Monte Carlo methods
  3) Implementing Heston stochastic volatility
* Creating/Pricing derivative products of arbitrary numbers of vanilla options:
  1) Time spreads
  2) Butterflies
  3) Straddles etc
* Multi-Asset options
  1) Option baskets
  2) Rainbow options
* Greeks using finite difference and analytic solutions where possible
  1) Delta
  2) Gamma
  3) Vega
  4) Rho
  5) Theta

Note: not all relevant functions have yet been vectorised for optimal efficiency in Monte Carlo simulations.
## Dependencies
NumPy and SciPy
  



## Miscellaneous simulations outside of MCLib
### C++
Respository contains variety of functions for calculating the price of numerous different exotic options such as Asian options, in and out barrier options and lookback options using Monte Carlo methods. Also contains a function for implementing Heston's model for stochastic volatility to generate random paths of underlying stock and contains analytic solutions to Black Scholes equations for vanilla european calls.

There is also an implementation of the Binomial pricing model to price American options with early exercise.

### Python
Repository also contains Monte Carlo pricing for methods for a variety of multi-asset options including some rainbow and spread options implemented in Python. Correlated Brownian motions of underlyings are generated using Cholesky decomposition methods and numpy's multivariable random normal generator. The Python file also contains functionality to plot implied volatility curves implied by Heston stochastic volatility, an example of which is given below for underlying at 100 for t=0. To calculate implied volatility I have used the Newton-Raphson method. There are also implementations of antithetic and control variates to reduce variance and increase convergence speed for vanilla european options.

![volatiltiy smile](https://user-images.githubusercontent.com/91262171/182554043-f5fbd234-742b-4997-ac37-92865017a36e.png)

### Dependencies
NumPy, SciPy

