import numpy as np 
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras import Sequential, layers
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from keras.losses import BinaryCrossentropy, MeanSquaredError

def build_model(mspec, optimizer, lr, reg, n_inputs, clf):
    """
    Compiles neural network using TensorFlow/keras 

    param mspec       : specification of numbers of hidden layers and neurons 
    param optimizer   : stochastic gradient descent or Adam 
    param lr          : learning rate for optimizer
    param reg         : l1 penalty for weights 
    param n_inputs    : number of neurons in first layer 
    param clf         : True for classification, False for regression   
    return            : model/network  
    """

    model = keras.Sequential() # Feedforward neural network 
    
    model.add(layers.Dense(units=mspec[0], activation='relu', kernel_regularizer=regularizers.L1(reg), 
                           input_shape=(n_inputs,))) # First hidden layer 
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr) # Use stochastic GD by default 
    
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr) # Use Adam if specified 
    
    for m in mspec[1:]:
        model.add(layers.Dense(units=m, activation='relu', kernel_regularizer=regularizers.L1(reg))) 
        
    if clf:
        model.add(layers.Dense(units=1, activation='sigmoid')) # Sigmoid output for classification 
        model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy'])
    else: 
        model.add(layers.Dense(units=1))
        model.compile(optimizer=opt, loss=MeanSquaredError(), metrics=['mse']) 
        
    return model 

def grid_search(mspecs, optimizers, lrates, regs, X, y, n_inputs, clf): 
    
    """
    Performs grid searching/hyperparameter tuning to optimize neural networks 

    param mspecs     : list of model specifications 
    param optimizers : list of optimizers 
    param lrates     : list of learning rates for optimizers 
    param regs       : list of l1 penalties for weights 
    param X          : features/covariates 
    param y          : outcome variable 
    param n_inputs   : number of neurons in first layer 
    param clf        : True for classification, False for regression 
    return           : parameters of optimal network 
    """

    np.random.seed(123)
    tf.random.set_seed(123)
    
    hidden_layers, opt_names, lrate_values, reg_values, loss_values = [], [], [], [], [] 
    
    kf = KFold(n_splits=2, random_state=0, shuffle=True)
    
    if clf:
        kf = StratifiedKFold(n_splits=2, random_state=0, shuffle=True) 
    
    for o in optimizers:
        for m in mspecs:
            for lr in lrates: 
                for re in regs:

                    model = build_model(m, o, lr, re, n_inputs, clf) 

                    fold_loss = [] 

                    for train_idx, test_idx in kf.split(X, y):
                        model.fit(X[train_idx], y[train_idx], batch_size=32, epochs=20, verbose=0)

                        loss, metric = model.evaluate(X[test_idx], y[test_idx], verbose=0)
                        fold_loss.append(loss)

                    mean_loss = np.mean(fold_loss)

                    hidden_layers.append(m)
                    opt_names.append(o)
                    lrate_values.append(lr)
                    reg_values.append(re)
                    loss_values.append(mean_loss)
        
        
    min_idx = np.argmin(loss_values)
    
    return hidden_layers[min_idx], opt_names[min_idx], lrate_values[min_idx], reg_values[min_idx]