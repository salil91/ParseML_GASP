 #!/usr/bin python
"""
Arguments:
1: garun/relaxations directory (with the pickle file)
2: Percentage of samples in the training set
"""

import sys
import os.path
import pickle
import numpy as np


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
        
    with open(os.path.join(directoryPath, 'global_data.pkl'), 'rb') as handle:
        global_data = pickle.load(handle)
        
    # Remove nan values in TargetProperty
    for idx, datum in enumerate(global_data):
        if np.isnan(datum[6]) or datum[6] == 0 or np.isnan(datum[7]) or datum[7] == 0:
            global_data.pop(idx)

    # Split data array into training and testing sets
    trainSamples, testSamples = test_train_split(global_data, fracTrainSamples=fracTrainSamples)
    structureIDs = list(zip(*testSamples))[0]
    structures = list(zip(*testSamples))[1]

    saveData_split = {"structureIDs": structureIDs, "structures": structures,
                      "training samples": trainSamples, "testing samples": testSamples}

    pklName_split = "train_test_split"+'_'+str(int(fracTrainSamples*100))
    with open(os.path.join(directoryPath, pklName_split+'.pkl'), 'wb') as handle:
        pickle.dump(saveData_split, handle)


def test_train_split(data, fracTrainSamples=0.7):
    """
    Splits the input data array into training and testing sets.

    Args:
        data: input data array.

        fracTrainSamples: Fraction of samples to be used for training

    """
    
    numTrainSamples = int(fracTrainSamples * len(data))
    numTestSamples = len(data) - numTrainSamples
    np.random.shuffle(data) # Randomly shuffle the data
    trainSamples = data[:numTrainSamples] # Split into train-test sets
    testSamples = data[-numTestSamples:]
    
    print("Train set: "+str(len(trainSamples)))
    print("Test set: "+str(len(testSamples))+"\n")
    
    return trainSamples, testSamples


if __name__ == "__main__":
    main()