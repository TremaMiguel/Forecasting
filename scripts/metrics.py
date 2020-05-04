import numpy as np
# MASE 
def error_term(train, test, pred):
    # Numerator
    numerator = test - pred
    
    # Denominator
    shift = train.shift(1)
    denominator = train - shift
    denominator = np.abs(denominator.dropna())
    denominator = np.mean(np.sum(denominator))

    return numerator/denominator



def MASE(train, test, pred):
    
    # Call error_term function
    error = error_term(train, test, pred)
    
    return np.mean(np.abs(error))

def MSSE(train, test, pred):
    
    # Call error_term function
    error = error_term(train, test, pred)
    
    return np.mean(error**2)

def MdASE(train, test, pred):

     # Call error_term function
    error = error_term(train, test, pred)

    return np.median(np.abs(error))

def MdSSE(train, test, pred):

    # Call error_term function
    error = error_term(train, test, pred)

    return np.median(error**2)

