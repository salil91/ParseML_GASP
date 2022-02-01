 #!/usr/bin python
"""
Arguments:
1: garun/relaxations directory (with the presplit pickle file)
2: Percentage of samples in the training set
3: ML method (SVR or KRR)
4: Target
"""

import sys
import os.path
import pickle

import numpy as np
import scipy.stats

from sklearn.preprocessing import StandardScaler as SS
from sklearn.model_selection import RandomizedSearchCV as CV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def main():
    try:
        directoryPath = sys.argv[1]
    except:
        print("Directory path containging the parsed pickle file must be defined")
        
    if len(sys.argv) > 2:
        fracTrainSamples = float(sys.argv[2])/100
    else:
        fracTrainSamples = 0.7
        print("Default Fraction for Training Set: 0.7")
        
    if len(sys.argv) > 3:      
        MLmethod = sys.argv[3]
    else:
        MLmethod = "SVR"
        print("Default ML method: SVR")
        
    if len(sys.argv) > 4:
        TargetProperty = sys.argv[4]
    else:
        TargetProperty = "energy"
        print("Default Target: ", TargetProperty)
        
    pklName_split = "train_test_split"+'_'+str(int(fracTrainSamples*100))    
    with open(os.path.join(directoryPath, pklName_split+'.pkl'), 'rb') as handle:
        saveData_split = pickle.load(handle)

    # Split data array into training and testing sets
    trainSamples = saveData_split["training samples"]
    testSamples = saveData_split["testing samples"]
    structureIDs = saveData_split["structureIDs"]
    structures = saveData_split["structures"]

    model,  Vecs, Targets, r2, rmse, mae = predict(trainSamples, testSamples, TargetProperty=TargetProperty,
                                                           MLmethod=MLmethod, fracTrainSamples=fracTrainSamples)
    saveData = {"structureIDs": structureIDs, "structures": structures, "Model": model,
                "Vecs": Vecs, "Targets": Targets, "R2": r2, "RMSE": rmse, "MAE": mae}

    pklName = TargetProperty.lower()+'_'+MLmethod.upper()+'_'+str(int(fracTrainSamples*100))
    with open(os.path.join(directoryPath, pklName+'.pkl'), 'wb') as handle:
        pickle.dump(saveData, handle)

    
def predict(trainSamples, testSamples, TargetProperty, MLmethod="SVR", fracTrainSamples=0.7, epsilon=0.01, cScale=5, gScale=0.001, n_iter=50):
    
    """
    Splits the input data array into training and testing sets, trains an SVR/KKR model on the data, and prints out 
    RMSE, MAE, and R^2 values.

    Args:
        trainSamples, testSamples: input data arrays for training and testing

        numTrainSamples: number of samples to be used for training

        epsilon: SVR parameter, check paper for details

        cScale, gScale: C and Gamma are SVR hyperparameters that we select through cross validation.

        n_iter: cross validation parameter 
    """
 
    if MLmethod.upper() == "SVR":
        from sklearn.svm import SVR

        # Use random search CV (5-fold) to select best hyperparameters
        param_dist = {'C': scipy.stats.expon(scale=cScale), 'gamma': scipy.stats.expon(scale=gScale), 
                      'kernel': ['rbf']} 
        ml_model = CV(SVR(epsilon=epsilon), param_distributions=param_dist, cv=5, scoring='neg_mean_squared_error', 
                      n_iter=n_iter, n_jobs=-1)
    elif MLmethod.upper() == "KRR":
        from sklearn.kernel_ridge import KernelRidge

        # Use random search CV (5-fold) to select best hyperparameters
        param_dist = {'alpha': scipy.stats.expon(scale=cScale), 'gamma': scipy.stats.expon(scale=gScale), 
                      'kernel': ['rbf']} 
        ml_model = CV(KernelRidge(), param_distributions=param_dist, cv=5, scoring='neg_mean_squared_error', 
                      n_iter=n_iter, n_jobs=-1)
    else:
        print("Unsupported ML method!")
        
    # Define TargetIndex
    if TargetProperty.lower() == "energy":
        TargetIndex = 6
    elif TargetProperty.lower() == "hardness":
        TargetIndex = 7
    else:
        print("Unsupported Target")
    
    # Targets for the training set
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

    # Fit ML model
    ML_model = ml_model.fit(trainVecs, trainTargets) 
    ML_best = ML_model.best_estimator_ # Pick the best set of hyperparameters for predictions

    # Make predictions and report errors
    teE_Pred = ML_best.predict(testVecs) 
    rmse = np.sqrt(mean_squared_error(testTargets, teE_Pred)) 
    mae = mean_absolute_error(testTargets, teE_Pred)  
    r2 = r2_score(testTargets, teE_Pred)
    
    print("\n")
    rmse_str = "Test RMSE: {:0.2f}"
    mae_str = "Test MAE: {:0.2f}"
    if TargetIndex == 6:
        print(rmse_str.format(rmse*1000)+" meV/atom")
        print(mae_str.format(mae*1000)+" meV/atom")
    elif TargetIndex == 7:
        print(rmse_str.format(rmse)+" GPa")
        print(mae_str.format(mae)+" GPa")
    else:
        print(rmse_str.format(rmse))
        print(mae_str.format(mae))
    print("R2 score: {:0.3f}".format(r2))
    print("\n")
    
    return ML_best, testVecs, testTargets, r2, rmse, mae


if __name__ == "__main__":
    main()
