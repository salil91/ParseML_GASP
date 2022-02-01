#!/usr/bin python
"""
Arguments:
1: garun/relaxations directory
2: first element
3: second element
"""

import sys
import os.path
import itertools
import pickle

import numpy as np

from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element


def main():
    directoryPath = sys.argv[1]
    elements = sys.argv[2:]
    structure_ext = ".poscar"
    energy_ext = ".energy"
    hardness_ext = ".hardness"
    
    rdf_tup = calc_rdf_tup(elements)
    adf_tup = calc_adf_tup(elements)
    
    data = parseData(directoryPath, elements, rdf_tup, adf_tup, structure_ext, energy_ext, hardness_ext)
    
    with open(os.path.join(directoryPath,'global_data.pkl'), 'wb') as handle:
        pickle.dump(data, handle)
    
    
def calc_rdf_tup(elements):
    return[list(p) for p in itertools.combinations_with_replacement(elements, 2)]

def calc_adf_tup(elements):
    adf_tup = [list(p) for p in itertools.product(elements, repeat=3)]
    del adf_tup[3]
    del adf_tup[3]
    
    return adf_tup

def calc_mol_frac(elements, Structure):
    molarFrac = [] 
    numElements = len(elements)    
    for i in range(numElements):
        elem = Element(elements[i])
        elemPerc = Structure.composition.get_atomic_fraction(elem)
        molarFrac.append((elements[i], elemPerc))
        
    return molarFrac

def calc_ave_epa(directoryPath, structureID, energy_ext):    
    energyFilePath = os.path.join(directoryPath, structureID+energy_ext)
    with open(energyFilePath) as f:
        lines = f.read().splitlines()
    local_energies = [float(line) for line in lines]
    epa = np.sum(local_energies)/len(local_energies) # This dataset has per-atom energies but we won't use them
    
    return epa

def calc_raw_data(structureIDs, directoryPath, elements, RDF_Tup, ADF_Tup, structure_ext, energy_ext):
    rawData = []
    for structureID in structureIDs:

        # Get average energy per atom for the structures
        epa = calc_ave_epa(directoryPath, structureID, energy_ext)

        # Read the POSCAR for the structures
        poscarFilePath = os.path.join(directoryPath, structureID+structure_ext)
        Structure = Poscar.from_file(poscarFilePath, False).structure

        # Get the RDF and ADF matrices
        RDFMatrix = getRDF_Mat(Structure, RDF_Tup) # One RDF per structure 
        ADFMatrix = getADF_Mat(Structure, ADF_Tup) # One ADF per structure

        # Calculate the molar fractions of elem_A and elem_B in the structure
        molarFrac = calc_mol_frac(elements, Structure)

        # Create the rawData array
        # structureData = [RDFMatrix, ADFMatrix, epa, dict(molarFrac), structureID, Structure]
        rawData.append([structureID, Structure, dict(molarFrac), RDFMatrix, ADFMatrix, epa])
        
    return rawData

def get_ref_e(rawData, elements):
    refEnergies = []
    for elem in elements:
        pureElemCrystals = [crystal for crystal in rawData if crystal[2][elem] == 1.0]
        energies = list(zip(*pureElemCrystals))[5]
        minEnergy = min(energies)
        refEnergies.append((elem, minEnergy))
        
    return dict(refEnergies)
  
def calc_fe(rawData, elements, refEnergies):
    formEnergyData = []
    for datum in rawData:
        molarFracs = datum[2]
        epa = datum[5]
        refContributions = [(molarFracs[elem]*refEnergies[elem]) for elem in elements]
        fe = epa - np.sum(refContributions)
        datum.append(fe)
        formEnergyData.append(datum)
        
    return(formEnergyData)

def get_hardness(formEnergyData, directoryPath, hardness_ext):
    hardnessData = []
    for datum in formEnergyData:
        structureID = datum[0]
        with open(os.path.join(directoryPath,structureID+hardness_ext)) as f:
            hardness = float(f.read())
        datum.append(hardness)
        hardnessData.append(datum)
    
    return hardnessData

def getRDF_Mat(cell, RDF_Tup, cutOffRad = 10.1, sigma = 0.2, stepSize = 0.1):
    
    """
        Calculates the RDF for every structure.

        Args:
            cell: input structure.

            RDF_Tup: list of all element pairs for which the partial RDF is calculated.

            cutOffRad: max. distance up to which atom-atom intereactions are considered.

            sigma: width of the Gaussian, used for broadening

            stepSize:  bin width, binning transforms the RDF into a discrete representation. 

    """

    binRad = np.arange(0.1, cutOffRad, stepSize) # Make bins based on stepSize and cutOffRad
    numBins = len(binRad)
    numPairs = len(RDF_Tup)
    vec = np.zeros((numPairs, numBins)) # Create a vector of zeros (dimension: numPairs*numBins)
    
    # Get all neighboring atoms within cutOffRad for alphaSpec and betaSpec
    # alphaSpec and betaSpec are the two elements from RDF_Tup
    for index,pair in enumerate(RDF_Tup):
        alphaSpec = Element(pair[0])  
        betaSpec = Element(pair[1])
        hist = np.zeros(numBins)  
        neighbors = cell.get_all_neighbors(cutOffRad) 
    
        sites = cell.sites # All sites in the structue
        indicesA = [j[0] for j in enumerate(sites) if j[1].specie == alphaSpec] # Get all alphaSpec sites in the structure
        numAlphaSites = len(indicesA)
        indicesB = [j[0] for j in enumerate(sites) if j[1].specie == betaSpec]  # Get all betaSpec sites in the structure
        numBetaSites = len(indicesB)
    
        # If no alphaSpec or betaSpec atoms, RDF vector is zero
        if numAlphaSites == 0 or numBetaSites == 0:
            vec[index] = hist 
            continue    

        alphaNeighbors = [neighbors[i] for i in indicesA] # Get all neighbors of alphaSpec 
    
        alphaNeighborDistList = []
        for aN in alphaNeighbors:
            tempNeighborList = [neighbor for neighbor in aN if neighbor[0].specie==betaSpec] # Neighbors of alphaSpec that are betaSpec
            alphaNeighborDist = []
            for j in enumerate(tempNeighborList):
                alphaNeighborDist.append(j[1][1])
            alphaNeighborDistList.append(alphaNeighborDist) # Add the neighbor distances of all such neighbors to a list

        # Apply gaussian broadening to the neigbor distances, 
        # so the effect of having a neighbor at distance x is spread out over few bins around x
        for aND in alphaNeighborDistList:
            for dist in aND: 
                inds = dist/stepSize
                inds = int(inds)
                lowerInd = inds-5
                if lowerInd < 0:
                    while lowerInd < 0:
                        lowerInd = lowerInd + 1
                upperInd = inds+5
                if upperInd >= numBins:
                    while upperInd >= numBins:
                        upperInd = upperInd - 1
                ind = range(lowerInd, upperInd)
                evalRad = binRad[ind] 
                exp_Arg = .5 *( ( np.subtract( evalRad, dist)/ (sigma) )**2) # Calculate RDF value for each bin
                rad2 = np.multiply(evalRad, evalRad) # Add a 1/r^2 normalization term, check paper for descripton
                hist[ind] += np.divide(np.exp(-exp_Arg), rad2)
    
        tempHist = hist/numAlphaSites # Divide by number of AlphaSpec atoms in the unit cell to give the final partial RDF
        vec[index] = tempHist
    
    vec = np.row_stack((vec[0], vec[1], vec[2]))  # Combine all vectors to get RDFMatrix
    return vec

def getADF_Mat(cell, ADF_Tup, cutOffRad = 5, sigma = 0.2, stepSize = 0.1):
    
    """
        Calculates the ADF for every structure.

        Args:
            cell: input structure.

            ADF_Tup: list of all element triplets for which the ADF is calculated.

            cutOffRad: max. distance up to which atom-atom intereactions are considered.

            sigma: width of the Gaussian, used for broadening

            stepSize: bin width, binning transforms the ADF into a discrete representation.

    """
    
    binRad = np.arange(-1, 1, stepSize) # Make bins based on stepSize
    numBins = len(binRad)
    numTriplets = len(ADF_Tup)
    vec = np.zeros((numTriplets, numBins)) # Create a vector of zeros (dimension: numTriplets*numBins)

    # Get all neighboring atoms within cutOffRad for alphaSpec, betaSpec, and gammaSpec
    # alphaSpec, betaSpec, and gammSpec are the three elements from ADF_Tup
    for index,triplet in enumerate(ADF_Tup):
        alphaSpec = Element(triplet[0]) 
        betaSpec = Element(triplet[1])
        gammaSpec = Element(triplet[2])
        hist = np.zeros(numBins)
        neighbors = cell.get_all_neighbors(cutOffRad) 

        sites = cell.sites # All sites in the structue
        indicesA = [j[0] for j in enumerate(sites) if j[1].specie == alphaSpec] # Get all alphaSpec sites in the structure
        numAlphaSites = len(indicesA)
        indicesB = [j[0] for j in enumerate(sites) if j[1].specie == betaSpec]  # Get all betaSpec sites in the structure
        numBetaSites = len(indicesB)
        indicesC = [j[0] for j in enumerate(sites) if j[1].specie == gammaSpec] # Get all gammaSpec sites in the structure
        numGammaSites = len(indicesC)
        
        # If no alphaSpec or betaSpec or gammsSpec atoms, RDF vector is zero
        if numAlphaSites == 0 or numBetaSites == 0 or numGammaSites == 0:
            vec[index] = hist 
            continue

        betaNeighbors = [neighbors[i] for i in indicesB] # Neighbors of betaSpec only

        alphaNeighborList = []
        for bN in betaNeighbors:
            tempalphaNeighborList = [neighbor for neighbor in bN if neighbor[0].specie==alphaSpec] # Neighbors of betaSpec that are alphaSpec
            alphaNeighborList.append(tempalphaNeighborList) # Add all such neighbors to a list

        gammaNeighborList = []
        for bN in betaNeighbors:
            tempgammaNeighborList = [neighbor for neighbor in bN if neighbor[0].specie==gammaSpec] # Neighbors of betaSpec that are gammaSpec
            gammaNeighborList.append(tempgammaNeighborList) # Add all such neighbors to a list

        # Calculate cosines for every angle ABC using side lengths AB, BC, AC
        cosines=[]
        f_AB=[]
        f_BC=[]
        for B_i,aN in enumerate(alphaNeighborList):
            for i in range(len(aN)):
                for j in range(len(gammaNeighborList[B_i])):
                    AB = aN[i][1]
                    BC = gammaNeighborList[B_i][j][1]
                    AC = np.linalg.norm(aN[i][0].coords-gammaNeighborList[B_i][j][0].coords)
                    if AC!=0:
                        cos_angle = np.divide(((BC*BC)+(AB*AB)-(AC*AC)),2*BC*AB)
                    else:
                        continue
                    # Use a logistic cutoff that decays sharply, check paper for details [d_k=3, k=2.5]
                    AB_transform = 2.5*(3-AB)
                    f_AB.append(np.exp(AB_transform)/(np.exp(AB_transform)+1))
                    BC_transform = 2.5*(3-BC)
                    f_BC.append(np.exp(BC_transform)/(np.exp(BC_transform)+1))
                    cosines.append(cos_angle)

        
        # Apply gaussian broadening to the neigbor distances, 
        # so the effect of having a neighbor at distance x is spread out over few bins around x
        for r, ang in enumerate(cosines): 
            inds = ang/stepSize
            inds = int(inds)
            lowerInd = inds-2+10
            if lowerInd < 0:
                while lowerInd < 0:
                    lowerInd = lowerInd + 1
            upperInd = inds+2+10
            if upperInd > numBins:
                while upperInd > numBins:
                    upperInd = upperInd - 1
            ind = range(lowerInd, upperInd)
            evalRad = binRad[ind]
            exp_Arg = .5 *( ( np.subtract( evalRad, ang)/ (sigma) )**2) # Calculate ADF value for each bin
            hist[ind] += np.exp(-exp_Arg)
            hist[ind] += np.exp(-exp_Arg)*f_AB[r]*f_BC[r]

        vec[index] = hist

    vec = np.row_stack((vec[0], vec[1], vec[2], vec[3], vec[4], vec[5]))  # Combine all vectors to get ADFMatrix
    return(vec)

def parseData(directoryPath, elements, RDF_Tup, ADF_Tup, structure_ext, energy_ext, hardness_ext):
    """
    Creates an input data array for the ML algorithm. For each structure - its RDFMatrix, ADFMatrix, 
    formation energy, structure, and molar fraction are stored.

    Args:
        directoryPath: path to Archive data folder.

        elements: list of elements in the binary system, in this case elem_A and elem_B.

        RDF_Tup: tuple of element pairs for which the partial RDF is computed.
        
        ADF_Tup: tuple of element triplets for which the partial ADF is computed.

    """
    
    # List and sort files by structureIDs
    structureIDs = [os.path.splitext(f)[0]
                    for f in os.listdir(directoryPath)
                    if structure_ext in f]
    structureIDs = sorted(structureIDs)
    
    # Create the rawData array
    rawData = calc_raw_data(structureIDs, directoryPath, elements, RDF_Tup, ADF_Tup, structure_ext, energy_ext)
    
    # Get the reference energies for elem_A and elem_B
    # i.e. the lowest avg_energy of pure element structures in the dataset
    refEnergies = get_ref_e(rawData, elements)
    
    # Calculate the formation energies from the total energies and reference energies
    # Replace total energy entries in rawData with formation energies (per atom)
    formEnergyData = calc_fe(rawData, elements, refEnergies)
        
    # Get the hardness data from the .hardness files
    hardnessData = get_hardness(formEnergyData, directoryPath, hardness_ext)
        
    return(hardnessData)


if __name__ == "__main__":
    main()
