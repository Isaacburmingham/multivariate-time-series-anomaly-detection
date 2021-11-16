import numpy as np
import pandas as pd
#Below function is pulled from Sada Narayanappa's Github with few adjustments. 
# Purpose is to ensure to features are highly correlated with Time

#See citation file for details 

# Find any sensor highly correlated with time and drop them.
def detectTimeCorrelated(df, timecol="time", val=0.94, **kwargs):
    timecol = df.columns[0]
    
    timeser = pd.Series(df[[timecol]].values.reshape(-1))
    if ( timeser.dtype != np.number ):
        timeser = pd.Series(pd.to_datetime(timeser).values.astype(int))
    
    
    DROP_INDEX = 0; # Debugging
    corcols    = []
    for sensor in df.columns:
        if (sensor == timecol ):
            continue;
        #print(f"#Testing {sensor}...")
        # The following code tries to detect correlation by dropping first 8 or last 8 values
        # sometimes dropping first few will show correlation due to start up times
        sensorSeries = pd.Series(df[sensor].values.reshape(-1))
        for i in range(8):
            c1 = timeser[i:].corr(sensorSeries[i:])
            c2 = timeser[i:].corr(sensorSeries[:-i])
            if np.abs(c1) >= val or np.abs(c2) >= val:
                corcols.append(sensor)
                DROP_INDEX = max(DROP_INDEX, i) #lets drop first few rows
                break;
                
    #print(f"#Time Cor: #{len(timeCorSensors)}, #Shape before:{df.shape}")
    #df.drop(timeCorSensors, axis=1, inplace=True)
    #df = df[DROP_INDEX:]
    #print(f"#After dropping: {DROP_INDEX} =>{df.shape}")
        
    return corcols

def reshape_predictions(label,yhat_train,yhat_test,timesteps):
    yhat_train_p =  np.empty(shape=[label.shape[0],])
    yhat_train_p[:] = np.nan
    yhat_train.shape = yhat_train.shape[0]
    yhat_train_p.shape = yhat_train_p.shape[0]
    yhat_train_p[int(timesteps):len(yhat_train)+int(timesteps)] = yhat_train
    
    yhat_test_p =  np.empty(shape=[label.shape[0],])
    yhat_test_p[:] = np.nan
    yhat_test.shape = yhat_test.shape[0]
    yhat_test_p.shape = yhat_test_p.shape[0]
    yhat_test_p[len(yhat_train)+(int(timesteps)*2):len(label)] = yhat_test
    
    return yhat_train_p, yhat_test_p
