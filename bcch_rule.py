import numpy as np 
from scipy.stats import norm 
from sklearn.linear_model import Lasso 

def bcch_rule(X, Y, c, alpha):

    """
    Selection rule for l1 penalty in LASSO proposed by Belloni, Chen, Chernozhukov, & Hansen (2012)

    param X     : vector of covariates 
    param Y     : outcome variable 
    param c     : see Belloni et al. (2012)
    param alpha : see Belloni et al. (2012)
    return      : choice for l1 penalty 
    """

    Y = np.array(Y).reshape(-1, 1)
    X = np.array(X)
    
    n, p = len(Y), X.shape[1]
    
    max_pilot = np.max(np.mean((X ** 2) * (Y ** 2), axis=0)) ** 0.5
    lambda_pilot = ((2 * c) / np.sqrt(n)) * norm.ppf(1 - (alpha / (2 * p))) * max_pilot
    
    lasso_pilot = Lasso(alpha=lambda_pilot, fit_intercept=False).fit(X, Y)
    
    residuals = Y - lasso_pilot.predict(X).reshape(-1, 1)
    
    max_bcch = np.max(np.mean((X ** 2) * (residuals ** 2), axis=0)) ** 0.5
    lambda_bcch = ((2 * c) / np.sqrt(n)) * norm.ppf(1 - (alpha / (2 * p))) * max_bcch
    
    return lambda_bcch
