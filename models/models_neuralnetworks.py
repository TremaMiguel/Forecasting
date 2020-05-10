import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from numpy import concatenate
from math import sqrt 
from sklearn.metrics import mean_squared_error

class NeuralNetwork():
    
    def __init__(self, dt:'pd.DataFrame'=None, target_variable:str=None, X_train:'pd.DataFrame'=None, X_test:'pd.DataFrame'=None, y_train:'pd.DataFrame'=None, y_test:'pd.DataFrame'=None, train = True, *args): 
        if train:
            self.X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            self.X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            self.y_train = y_train
            self.y_test = y_test
            self.train = train
            self.args = args
        else:
            self.dt = dt 
            self.target = dt[target_variable]
            self.dependent = dt.drop(columns=[target_variable]) 
            self.train = train

    def nn_model(self, model:str, mdl_params:dict, fit_params:dict, plot:bool, layers:int, units:int, steps:int, loss_function:'tf.keras.layers', optimizer:str, epochs:int):

        # TO DO : Add a CNN-NN or bidirectional LSTM and RNN-NN
        if model == 'Stacked-LSTM':
            nn_model = tf.keras.models.Sequential()

            # Add layers
            for l in range(layers): 
                nn_model.add(units/l,tf.keras.layers.LSTM(**mdl_params))

        # Output Layer
        nn_model.add(tf.keras.layers.Dense(steps))
        nn_model.compile(loss=loss_function,optimizer = optimizer)

        # Fit model 
        nn_model.fit(self.X_train, self.y_train, **fit_params)

        # Training and validation Loss visualization
        if plot:
            t_loss, v_loss = nn_model.history['loss'], nn_model.history['val_loss']
            plt.figure(figsize = (12,8))
            plt.plot(epochs, t_loss, 'black', label='Training loss')
            plt.plot(epochs, v_loss, 'red', label='Validation loss')
            plt.title(f"Training and validation loss with {optimizer}")
            plt.legend()
            plt.show()

        # Predictions 
        y_pred = nn_model.predict(X_test)
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))

        # Reshape 
        y_pred = concatenate((y_pred, X_test), axis=1)

        # Inverts scaling of real values
        y_test = y_test.reshape((len(y_test), 1))
        y = concatenate((y_test, X_test), axis=1)

        # Metrics
        rmse, mape = sqrt(mean_squared_error(inv_pred, inv_y)), np.mean(np.abs((inv_y - inv_pred) / inv_y)) * 100
        print(f"Test RMSE: {rmse}", "\n", 'Test MAPE: {mape}')
