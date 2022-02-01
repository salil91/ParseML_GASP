 #!/usr/bin python
"""
Arguments:
1: garun/relaxations directory (with the pickle file)
2: Target (Energy/hardness)
3: ML method (SVR or KRR)
4: Percentage of samples in the training set
"""

import sys
import os.path
import pickle

import numpy as np
import scipy.stats

from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import RandomizedSearchCV as CV
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    try:
        directoryPath = sys.argv[1]
    except:
        print("Directory path containging the parsed pickle file must be defined")
    
    if len(sys.argv) > 2:
        TargetProperty = sys.argv[2]
    else:
        TargetProperty = "energy"
        print("Default Taget: Energy")
    
    if len(sys.argv) > 3:      
        MLmethod = sys.argv[3]
    else:
        MLmethod = "SVR"
        print("Default ML method: SVR")
        
    if len(sys.argv) > 4:
        fracTrainSamples = float(sys.argv[4])/100
    else:
        fracTrainSamples = 0.7
        print("Default Fraction for Training Set: 0.7")
        
    if TargetProperty.lower() == "energy":
        TargetIndex = 6
    elif TargetProperty.lower() == "hardness":
        TargetIndex = 7
    else:
        print("Unsupported Target")
    
    with open(os.path.join(directoryPath, 'global_data.pkl'), 'rb') as handle:
        global_data = pickle.load(handle)
        
    # Remove nan values in TargetProperty
    for idx, datum in enumerate(global_data):
        if np.isnan(datum[TargetIndex]) or datum[TargetIndex] == 0:
            global_data.pop(idx)
    
    numTrainSamples = int(fracTrainSamples * len(global_data))
    model, Vecs, Targets, r2, rmse, mae = predict(global_data, TargetIndex=TargetIndex, MLmethod=MLmethod, numTrainSamples=numTrainSamples)
    saveData = {"Model": model, "Vecs": Vecs, "Targets": Targets,
                "fracTrainSamples": fracTrainSamples, "numTrainSamples": numTrainSamples,
                "R2": r2, "RMSE": rmse, "MAE": mae}

    with open(os.path.join(directoryPath,
                           TargetProperty.lower()+'_'+MLmethod.upper()+'_'+str(int(fracTrainSamples*100))+'.pkl'),
              'wb') as handle:
        pickle.dump(saveData, handle)
        
    
def predict(data, TargetIndex=6, MLmethod="SVR", numTrainSamples=1000, epsilon=0.01, cScale=5, gScale=0.001, n_iter=50):
    
    """
    Splits the input data array into training and testing sets, trains an SVR/KKR model on the data, and prints out 
    RMSE, MAE, and R^2 values.

    Args:
        data: input data array.

        numTrainSamples: number of samples to be used for training

        epsilon: SVR parameter, check paper for details

        cScale, gScale: C and Gamma are SVR hyperparameters that we select through cross validation.

        n_iter: cross validation parameter 
    """
    
    numTestSamples = len(data) - numTrainSamples
    np.random.shuffle(data) # Randomly shuffle the data
    trainSamples = data[:numTrainSamples] # Split into train-test sets
    testSamples = data[-numTestSamples:]
 
    if MLmethod.upper() == "SVR":
        # Use random search CV (5-fold) to select best hyperparameters
        param_dist = {'C': scipy.stats.expon(scale=cScale), 'gamma': scipy.stats.expon(scale=gScale), 
                      'kernel': ['rbf']} 
        ml_model = CV(SVR(epsilon=epsilon), param_distributions=param_dist, cv=5, scoring='neg_mean_squared_error', 
                      n_iter=n_iter, n_jobs=-1)
    elif MLmethod.upper() == "KRR":
        # Use random search CV (5-fold) to select best hyperparameters
        param_dist = {'alpha': scipy.stats.expon(scale=cScale), 'gamma': scipy.stats.expon(scale=gScale), 
                      'kernel': ['rbf']} 
        ml_model = CV(KernelRidge(), param_distributions=param_dist, cv=5, scoring='neg_mean_squared_error', 
                      n_iter=n_iter, n_jobs=-1)
    else:
        print("Unsupported ML method!")
    
    # Formation energies are the default targets
    trainTargets = list(zip(*trainSamples))[TargetIndex] 
    # RDF and ADF matrices are the inputs
    trainVecs_RDF = list(zip(*trainSamples))[3] 
    trainVecs_RDF = list(map(np.ndarray.flatten, trainVecs_RDF))
    trainVecs_ADF = list(zip(*trainSamples))[4] 
    trainVecs_ADF = list(map(np.ndarray.flatten, trainVecs_ADF))
    trainVecs = np.column_stack((trainVecs_RDF, trainVecs_ADF))
  
    # Do the same with the test set
    testTargets = list(zip(*testSamples))[TargetIndex]
    testVecs_RDF = list(zip(*testSamples))[3] 
    testVecs_RDF = list(map(np.ndarray.flatten, testVecs_RDF))
    testVecs_ADF = list(zip(*testSamples))[4] 
    testVecs_ADF = list(map(np.ndarray.flatten, testVecs_ADF))
    testVecs = np.column_stack((testVecs_RDF, testVecs_ADF))
    
    # Combine train and test sets
    # Targets = trainTargets + testTargets
    # Vecs = [*trainVecs, *testVecs]
    
    # Feature scaling
    scaler = SS().fit(trainVecs) 
    trainVecs = scaler.transform(trainVecs)
    testVecs = scaler.transform(testVecs)

    print("Train set:", len(trainTargets))
    print("Test set:", len(testTargets), "\n")

    # Fit ML model
    ML_model = ml_model.fit(trainVecs, trainTargets) 
    ML_best = ML_model.best_estimator_ # Pick the best set of hyperparameters for predictions

    # Make predictions and report errors
    teE_Pred = ML_best.predict(testVecs) 
    rmse = np.sqrt(mean_squared_error(testTargets, teE_Pred)) 
    mae = mean_absolute_error(testTargets, teE_Pred)  
    r2 = r2_score(testTargets, teE_Pred)
       
    return ML_best, testVecs, testTargets, r2, rmse, mae


if __name__ == "__main__":
    main()
