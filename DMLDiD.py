import numpy as np 
import tensorflow as tf 
from nn_dml_cv import build_model, grid_search
from sklearn.model_selection import GridSearchCV

def DMLDiD(Y_1, Y_0, D, X, g_ML, g_params, mu_ML, mu_params):
    
    """
    Computes DMLDiD ATT and standard error
    Function works for sklearn and dmlc's XGBoost (uses sklearn wrapper)

    param Y_1       : outcome of individual in post-treatment period 
    param Y_0       : outcome of individual in pre-treatment period 
    param D         : indicator variable for treatment status 
    param X         : vector of individual's covariates
    param g_ML      : supervised classification model for propensity score estimation
    param g_params  : dict of grids for hyperparameter tuning in g_ML 
    param mu_ML     : supervised regression model for ell_1k estimation 
    param mu_params : dict of grids for hyperparameter tuning in mu_ML 
    return          : ATT, SE  
    """

    Y_1 = Y_1.reshape(-1, 1) # Reshape arrays to 2D 
    Y_0 = Y_0.reshape(-1, 1) # ""
    D = D.reshape(-1, 1)     # ""
    
    df = np.concatenate((Y_1, Y_0, D, X), axis=1) # Combine everything into one 2D array 
    
    np.random.seed(123) # Set seed for replicability
    np.random.shuffle(df) # Shuffle data 
    
    K = 2 # Number of folds for cross-fitting 
    N = Y_1.shape[0] # Number of individuals (observations)
    n = round(df.shape[0] / K) # Number of approx observations for each fold
    
    df_I_k = [df[:n, :], df[n:, :]] # List of partions/folds
    
    att_I_k, var_I_k = [], [] # Empty lists to hold results for each fold 
    
    for df_k in df_I_k: # For each fold calculate DMLDiD 
        D, X = df_k[:, 2], df_k[:, 3:]
        
        p_hat = np.mean(D)
        
        prop_score = GridSearchCV(estimator=g_ML, param_grid=g_params, scoring='neg_log_loss', cv=2)
        prop_score.fit(X, D)
        
        g_hat = prop_score.predict_proba(X)[:, 1]
        
        df_mu = df_k[np.where(df_k[:, 2] == 0)] # Slicing to estimate mu_hat 
        
        Y_1_mu, Y_0_mu, X_mu = df_mu[:, 0], df_mu[:, 1], df_mu[:, 3:]
      
        cond_mean = GridSearchCV(estimator=mu_ML, param_grid=mu_params, scoring='neg_mean_squared_error', cv=2)
        cond_mean.fit(X_mu, Y_1_mu - Y_0_mu) 
        
        mu_hat = cond_mean.predict(X)
        
        Y_1, Y_0 = df_k[:, 0], df_k[:, 1]
    
        t0 = ((Y_1 - Y_0) / p_hat) * ((D - g_hat) / (1 - g_hat))
        
        c1 = ((D - g_hat) / (p_hat * (1 - g_hat))) * mu_hat
        
        att_k = np.mean(t0 - c1)
        
        att_I_k.append(att_k)
    
    theta = np.mean(att_I_k)
    
    for df_k in df_I_k: # For each fold calculate variance
        D, X = df_k[:, 2], df_k[:, 3:]
        
        p_hat = np.mean(D)
        
        prop_score = GridSearchCV(estimator=g_ML, param_grid=g_params, scoring='neg_log_loss', cv=2)
        prop_score.fit(X, D)
        
        g_hat = prop_score.predict_proba(X)[:, 1] 
        
        df_mu = df_k[np.where(df_k[:, 2] == 0)] # Slicing to estimate mu_hat 
        
        Y_1_mu, Y_0_mu, X_mu = df_mu[:, 0], df_mu[:, 1], df_mu[:, 3:]
        
        cond_mean = GridSearchCV(estimator=mu_ML, param_grid=mu_params, scoring='neg_mean_squared_error', cv=2)
        cond_mean.fit(X_mu, Y_1_mu - Y_0_mu)
        
        mu_hat = cond_mean.predict(X)
        
        Y_1, Y_0 = df_k[:, 0], df_k[:, 1]
    
        psi = (((Y_1 - Y_0) / p_hat) * ((D - g_hat) / (1 - g_hat))) - theta
        psi = psi - (((D - g_hat) / (p_hat * (1 - g_hat))) * mu_hat)

        G = (-1 * theta) / p_hat
    
        var_k = np.mean((psi + (G * (D - p_hat))) ** 2)
        
        var_I_k.append(var_k)
        
    sigma = np.mean(var_I_k)
    
    return theta, np.sqrt(sigma / N)

def DMLDiD_TF(Y_1, Y_0, D, X, n_inputs, g_params, mu_params):
    
    """
    Same function as DMLDiD() but for TensorFlow 
    param n_inputs  : number of input neurons in first layer of network 
    param g_params  : list containing grids for hyperparameter tuning in g_ML 
    param mu_params : list containing grids for hyperparameter tuning in mu_ML 
    return          : ATT, SE  
    """
    
    Y_1 = Y_1.reshape(-1, 1) 
    Y_0 = Y_0.reshape(-1, 1) 
    D = D.reshape(-1, 1)     
    
    df = np.concatenate((Y_1, Y_0, D, X), axis=1) 
    
    np.random.seed(123) 
    tf.random.set_seed(123)
    
    np.random.shuffle(df)  
    
    K = 2 
    N = Y_1.shape[0] 
    n = round(df.shape[0] / K) 
    
    df_I_k = [df[:n, :], df[n:, :]] 
    
    att_I_k, var_I_k = [], [] 
    
    for df_k in df_I_k:  
        D, X = df_k[:, 2], df_k[:, 3:]
        
        p_hat = np.mean(D)
        
        g_mspec, g_opt, g_lr, g_reg = grid_search(g_params[0], g_params[1], g_params[2], g_params[3], X, D, n_inputs, clf=True)  
        g_ML = build_model(g_mspec, g_opt, g_lr, g_reg, n_inputs, clf=True)
                                          
        prop_score = g_ML.fit(X, D, verbose=0, batch_size=32, epochs=20)
        g_hat = prop_score.model.predict(X).reshape(-1, )
        
        df_mu = df_k[np.where(df_k[:, 2] == 0)]
        
        Y_1_mu, Y_0_mu, X_mu = df_mu[:, 0], df_mu[:, 1], df_mu[:, 3:]
        
        mu_mspec, mu_opt, mu_lr, mu_reg = grid_search(mu_params[0], mu_params[1], mu_params[2], mu_params[3], 
                                                      X_mu, Y_1_mu - Y_0_mu, n_inputs, clf=False) 
        mu_ML = build_model(mu_mspec, mu_opt, mu_lr, mu_reg, n_inputs, clf=False)
        
        cond_mean = mu_ML.fit(X_mu, Y_1_mu - Y_0_mu, verbose=0, batch_size=32, epochs=20)
        mu_hat = cond_mean.model.predict(X).reshape(-1, )
        
        Y_1, Y_0 = df_k[:, 0], df_k[:, 1]
    
        t0 = ((Y_1 - Y_0) / p_hat) * ((D - g_hat) / (1 - g_hat))
        
        c1 = ((D - g_hat) / (p_hat * (1 - g_hat))) * mu_hat
        
        att_k = np.mean(t0 - c1)
        
        att_I_k.append(att_k)
    
    theta = np.mean(att_I_k)
    
    for df_k in df_I_k: # For each fold calculate variance
        D, X = df_k[:, 2], df_k[:, 3:]
        
        p_hat = np.mean(D)
        
        g_mspec, g_opt, g_lr, g_reg = grid_search(g_params[0], g_params[1], g_params[2], g_params[3], X, D, n_inputs, clf=True) 
        g_ML = build_model(g_mspec, g_opt, g_lr, g_reg, n_inputs, clf=True)
                                          
        prop_score = g_ML.fit(X, D, verbose=0, batch_size=32, epochs=20)
        g_hat = prop_score.model.predict(X).reshape(-1, )
        
        df_mu = df_k[np.where(df_k[:, 2] == 0)]
        
        Y_1_mu, Y_0_mu, X_mu = df_mu[:, 0], df_mu[:, 1], df_mu[:, 3:]
        
        mu_mspec, mu_opt, mu_lr, mu_reg = grid_search(mu_params[0], mu_params[1], mu_params[2], mu_params[3], 
                                                      X_mu, Y_1_mu - Y_0_mu, n_inputs, clf=False) 
        mu_ML = build_model(mu_mspec, mu_opt, mu_lr, mu_reg, n_inputs, clf=False)
        
        cond_mean = mu_ML.fit(X_mu, Y_1_mu - Y_0_mu, verbose=0, batch_size=32, epochs=20)
        mu_hat = cond_mean.model.predict(X).reshape(-1, )
        
        Y_1, Y_0 = df_k[:, 0], df_k[:, 1]
    
        psi = (((Y_1 - Y_0) / p_hat) * ((D - g_hat) / (1 - g_hat))) - theta
        psi = psi - (((D - g_hat) / (p_hat * (1 - g_hat))) * mu_hat)

        G = (-1 * theta) / p_hat
    
        var_k = np.mean((psi + (G * (D - p_hat))) ** 2)
        
        var_I_k.append(var_k)
        
    sigma = np.mean(var_I_k)
    
    return theta, np.sqrt(sigma / N)