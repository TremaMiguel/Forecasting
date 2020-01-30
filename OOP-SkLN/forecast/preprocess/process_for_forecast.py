class preprocess():
    
    def __init__(self, df:'pd.Dataframe', target_variable:str):
      '''
	        Initialization Parameters.
	        :param df: DataFrame with target variable to process.
          :param target_variable: Variable to process for forecasting
    	'''
        self.data = df   
        self.target = df[target_variable]
        self.dependent = df.drop(columns=[target_variable]) 
        self.process = pd.DataFrame()
    
    def window_slide(self, lags:int):
    	'''
	        Implement a Window slide to the input data
	        :param lags: how many prior observations to be considered for each target variable 
    	'''
        lags, max_index = lags +1, len(self.target)
        dt = pd.DataFrame(columns = [f"lag_{i}" for i in range(1,lags+1)])
        while max_index - lags >= 0:
            dt.loc[len(dt)] = list(self.target[max_index - lags:max_index])
            max_index -= 1
        dt = dt.rename(columns={f"lag_{lags}": "target_variable"})          # Rename column target
        dt = dt.reindex(index=dt.index[::-1])                               # Inverse order, the most recent observation should be the last observation of the dataframe
        self.process = dt.apply(pd.to_numeric, errors='ignore')
        return self.process 
    
    def train_test_split(self,  train_size, random_state = 0, shuffle = False):
    	'''
	        Implement the train_test_split function of the sklearn library
	        :param train_size: what percentage of data we would like to use to fit the model
	        :param random_state: seed
	        :param shuffle: wheter to shuffle or not the data
    '''
        Y, X = self.process["target_variable"], self.process.drop(columns = ["target_variable"])
        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size= train_size, random_state= random_state, shuffle = shuffle)
        return X_train, X_test, y_train, y_test
