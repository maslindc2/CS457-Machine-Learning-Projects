import numpy as np
import pcn_logic_eg as pcnLearnAlg

"""i Training Set"""
iVector1 = [1, 1, 1,
            0, 1, 0,
            1, 1, 1]

iVector2 = [0, 0, 0,
            0, 0, 1,
            0, 0, 1]

iVector3 = [0, 0, 0,
            0, 1, 0,
            0, 1, 0,]

iVector4 = [0, 0, 0,
            1, 0, 0,
            1, 0, 0,]

"""L Training"""
lVector1 = [1, 0, 0,
            1, 0, 0,
            1, 1, 1]

#shorter L
lVector2 = [1, 0, 0,
            1, 0, 0,
            1, 1, 0,]

lVector3 = [0, 1, 0,
            0, 1, 0,
            0, 1, 1,]

#Lowercase l
lVector4 = [1, 0, 0,
            1, 0, 0,
            1, 0, 0,]

trainingSet = np.array([iVector1, iVector2, iVector3, iVector4, lVector1, lVector2, lVector3, lVector4])
targets = np.array([[0],[0],[0],[0],[1],[1],[1],[1]])

pRef = pcnLearnAlg.pcn(trainingSet, targets)
pRef.pcntrain(trainingSet, targets, .1, 100)

"""Test Classifier"""
print("\nConfusion Matrix Test")

"""Test Set for I shapes"""
#Correct i Cases
iTestVector1 = [1, 1, 1,
                0, 1, 0,
                1, 1, 1]
#Correct i
iTestVector2 = [0, 0, 0,
                0, 0, 1,
                0, 0, 1]
#Incorrect I shapes
iTestVector3 = [1, 1, 1,
                0, 0, 0,
                1, 0, 1]

iTestVector4 = [0, 1, 0,
                1, 1, 1,
                1, 1, 1]

"""Test Set for L shapes"""
#Correct L shapes
lTestVector1 = [1, 0, 0,
                1, 0, 0,
                1, 1, 1]

lTestVector2 = [1, 0, 0,
                1, 0, 0,
                1, 0, 0]

#Incorrect L shapes
lTestVector3 = [1, 0, 1,
                0, 1, 0,
                1, 0, 1]

lTestVector4 = [1, 0, 1,
                1, 1, 1,
                1, 0, 1]

testSet = np.array([iTestVector1, iTestVector2, iTestVector3, iTestVector4, lTestVector1, lTestVector2, lTestVector3, lTestVector4])
pRef.confmat(testSet, targets)