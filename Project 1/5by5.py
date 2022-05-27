import numpy as np
import pcn_logic_eg as pcnLearnAlg

"""Training Set"""

"""I training set"""
iVector1 = [1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            1, 1, 1, 1, 1]

iVector2 = [1, 1, 1, 1, 1,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            1, 1, 1, 1, 1]

iVector3 = [1, 1, 1, 1, 0,
            0, 1, 1, 0, 0,
            0, 1, 1, 0, 0,
            0, 1, 1, 0, 0,
            1, 1, 1, 1, 0]

iVector4 = [0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0]


""""L training set"""
lVector1 = [1, 1, 0, 0, 0,
            1, 1, 0, 0, 0,
            1, 1, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1]

lVector2 = [1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 1, 1, 1, 0]

lVector3 = [1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0,
            1, 0, 0, 0, 0]

lVector4 = [1, 1, 0, 0, 0,
            1, 1, 0, 0, 0,
            1, 1, 0, 0, 0,
            1, 1, 0, 0, 0,
            1, 1, 0, 0, 0]


trainingSet = np.array([iVector1, iVector2, iVector3, iVector4, lVector1, lVector2, lVector3, lVector4])
targets = np.array([[0],[0],[0],[0],[1],[1],[1],[1]])

pRef = pcnLearnAlg.pcn(trainingSet, targets)
pRef.pcntrain(trainingSet, targets, .1, 150)

"""Test Classifier"""
print("\nConfusion Matrix Test")
"""Test set for I Shapes"""
#Correct I cases
iTestVector1 = [1, 1, 1, 1, 1,
                0, 1, 1, 1, 0,
                0, 1, 1, 1, 0,
                0, 1, 1, 1, 0,
                1, 1, 1, 1, 1]

iTestVector2 = [1, 1, 1, 1, 1,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                1, 1, 1, 1, 1]

#Incorrect I cases
iTestVector3 = [0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 1, 0, 0,
                1, 1, 0, 1, 0]

iTestVector4 = [0, 0, 0, 0, 0,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1]

"""Test L Shapes"""
#Correct L cases
lTestVector1 = [1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1]

lTestVector2 = [1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0,
                1, 1, 0, 0, 0]

#Incorrect L cases
lTestVector3 = [0, 0, 0, 0, 0,
                1, 0, 1, 0, 0,
                1, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                1, 1, 1, 0, 0]

lTestVector4 = [0, 1, 0, 0, 0,
                0, 0, 0, 1, 0,
                1, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                1, 1, 1, 0, 0]                
testSet = np.array([iTestVector1, iTestVector2, iTestVector3, iTestVector4, lTestVector1, lTestVector2, lTestVector3, lTestVector4])
pRef.confmat(testSet, targets)