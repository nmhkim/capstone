import numpy as np 
import tensorflow as tf 
from nn_sdid_cv import build_model, grid_search
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def SDiD(Y_1, Y_0, D, X, g_ML, g_params):

    """
    Computes SDiD ATT and standard error
    Function works for sklearn and dmlc's XGBoost (uses sklearn wrapper)

    param Y_1       : outcome of individual in post-treatment period 
    param Y_0       : outcome of individual in pre-treatment period 
    param D         : indicator variable for treatment status 
    param X         : vector of individual's covariates
    param g_ML      : supervised classification model for propensity score estimation
    param g_params  : dict of grids for hyperparameter tuning in g_ML  
    return          : ATT, SE  
    """

    p_hat = np.mean(D)
        
    prop_score = GridSearchCV(estimator=g_ML, param_grid=g_params, scoring='neg_log_loss', cv=2)
    prop_score.fit(X, D)
        
    g_hat = prop_score.predict_proba(X)[:, 1]
    
    att = np.mean(((Y_1 - Y_0) / p_hat) * ((D - g_hat) / (1 - g_hat)))
    
    Y_1 = Y_1.reshape(-1, 1) # Reshape arrays to 2D 
    Y_0 = Y_0.reshape(-1, 1) # ""
    D = D.reshape(-1, 1)     # ""
    
    N = Y_1.shape[0] # Number of individuals (observations)
    
    df = np.concatenate((Y_1, Y_0, D, X), axis=1) # Combine everything into one 2D array 
    
    df_mu = df[np.where(df[:, 2] == 0)] # Where D == 0
    Y_1_mu, Y_0_mu, X_mu = df_mu[:, 0], df_mu[:, 1], df_mu[:, 3:]
    
    cond_mean = RandomForestRegressor(random_state=0)
    cond_mean.fit(X_mu, Y_1_mu - Y_0_mu)
        
    mu_hat = cond_mean.predict(X)
    
    psi = (((Y_1 - Y_0) / p_hat) * ((D - g_hat) / (1 - g_hat))) - att
    psi = psi - (((D - g_hat) / (p_hat * (1 - g_hat))) * mu_hat)

    G = (-1 * att) / p_hat
    
    var = np.mean((psi + (G * (D - p_hat))) ** 2)
    
    return att, np.sqrt(var / N)

def SDiD_TF(Y_1, Y_0, D, X, n_inputs, g_params):
    
    """
    Same function as SDiD() but for TensorFlow 
    param n_inputs  : number of input neurons in first layer of network 
    param g_params  : list containing grids for hyperparameter tuning in g_ML 
    eturn           : ATT, SE  
    """

    np.random.seed(123) # Set seed for replicability
    tf.random.set_seed(123)
        
    p_hat = np.mean(D)
        
    g_mspec, g_opt, g_lr, g_reg = grid_search(g_params[0], g_params[1], g_params[2], g_params[3], X, D, n_inputs, clf=True) 
    g_ML = build_model(g_mspec, g_opt, g_lr, g_reg, n_inputs, clf=True)
                                          
    prop_score = g_ML.fit(X, D, verbose=0, batch_size=32, epochs=20)
    g_hat = prop_score.model.predict(X).reshape(-1, )
    
    att = np.mean(((Y_1 - Y_0) / p_hat) * ((D - g_hat) / (1 - g_hat)))
    
    Y_1 = Y_1.reshape(-1, 1) # Reshape arrays to 2D 
    Y_0 = Y_0.reshape(-1, 1) # ""
    D = D.reshape(-1, 1)     # ""
    
    N = Y_1.shape[0] # Number of individuals (observations)
    
    df = np.concatenate((Y_1, Y_0, D, X), axis=1) # Combine everything into one 2D array 
    
    df_mu = df[np.where(df[:, 2] == 0)] # Where D == 0
    Y_1_mu, Y_0_mu, X_mu = df_mu[:, 0], df_mu[:, 1], df_mu[:, 3:]
    
    cond_mean = RandomForestRegressor(random_state=0)
    cond_mean.fit(X_mu, Y_1_mu - Y_0_mu)
        
    mu_hat = cond_mean.predict(X)
    
    psi = (((Y_1 - Y_0) / p_hat) * ((D - g_hat) / (1 - g_hat))) - att
    psi = psi - (((D - g_hat) / (p_hat * (1 - g_hat))) * mu_hat)

    G = (-1 * att) / p_hat
    
    var = np.mean((psi + (G * (D - p_hat))) ** 2)
    
    return att, np.sqrt(var / N)