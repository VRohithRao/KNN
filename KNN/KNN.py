__author__ = 'Rohith'

import numpy as np
import math
import arff
import csv

def computeDistances(testFileName,trainFileName):
    testData,testMean,testSD = loadData(testFileName, 'test')
    trainData,trainMean,trainSD = loadData(trainFileName, 'train')

    result = []
    finalResult = []
    appendResults = []
    k = [1,3,5,7,9]

    for testRow in testData:
        finalResult = []

        testMatrix = np.asarray(normalizationKNN(testRow,testMean,testSD), dtype="float")
        for trainRow in trainData:
            className = trainRow[len(trainRow)-1:][0]
            trainRow = trainRow[:len(trainRow)-1]

            trainMatrix = np.asarray(normalizationKNN(trainRow,trainMean,trainSD), dtype="float")
            diffMatrix = testMatrix - trainMatrix
            diffMatrixTranspose = np.transpose(diffMatrix)

            resultMatrix = diffMatrix.dot(diffMatrixTranspose)
            euclideanDistance = math.sqrt(resultMatrix)
            result.append([euclideanDistance,className])

        result.sort()
        predictKNN(result, 0, k, finalResult)
        result = []
        finalResult.insert(0,testRow[0])
        finalResult.insert(1,testRow[1])
        finalResult.insert(2,testRow[2])
        finalResult.insert(3,testRow[3])

        appendResults.append(finalResult)

    writeOutput(appendResults)

def writeOutput(finalResult):
    outputFileName = 'Output-KNN.csv'
    lengthEin = len(finalResult)

    with open(outputFileName, 'wb') as f:
        writer = csv.writer(f, delimiter = ',')
        for val in finalResult:
            writer.writerow([val[0],val[1],val[2],val[3],val[4],val[5],
                            val[6],val[7],val[8]])



def normalizationKNN(rowData,mean,standardDeviation):
    zScore = []
    for i in range(0,len(rowData)):
        zScore.append((rowData[i] - mean[i])/standardDeviation[i])

    return zScore

def predictKNN(result, value, k, finalResult):
    i = 0
    className = {}
    tempList = []
    classKey = []
    manipulate = {}

    if value < len(k):
        kValue = k[value]
        while i < kValue:
            tempList.append(result[i])

            if result[i][1] in className:
                count = className.get(result[i][1])
                className[result[i][1]] = count + 1
            else:
                className[result[i][1]] = 1
            i += 1

        highest = max(className.values())

        for key, val in className.items():
            if val == highest:
                classKey.append(key)
                # print(classKey)

        if len(classKey) > 1:
            for listValue in tempList:
                if listValue[1] in classKey:
                    countDist = listValue[0]
                    if listValue[1] in manipulate:
                        manipulate[listValue[1]] = manipulate.get(listValue[1]) + countDist
                    else:
                        manipulate[listValue[1]] = countDist
        else:
            finalResult.append(classKey[0])

        if len(manipulate) > 0:
            # print(min(manipulate))
            finalResult.append(min(manipulate))

        value += 1
        predictKNN(result, value, k, finalResult)

def loadData(fileName,flag):
    data = []
    tempData = []
    mean = []
    standardDeviation = []
    x1 = x2 = x3 = x4 = count = 0.0

    if flag == 'test':
        for row in arff.load(fileName):
            data.append([row.sepal_length, row.sepal_width, row.petal_length, row.petal_width])
            x1 = x1 + row.sepal_length
            x2 = x2 + row.sepal_width
            x3 = x3 + row.petal_length
            x4 = x4 + row.petal_width
            count = count + 1
    else:
        for row in arff.load(fileName):
            data.append([row.sepal_length, row.sepal_width, row.petal_length, row.petal_width, row[4]])
            tempData.append([row.sepal_length, row.sepal_width, row.petal_length, row.petal_width])
            x1 = x1 + row.sepal_length
            x2 = x2 + row.sepal_width
            x3 = x3 + row.petal_length
            x4 = x4 + row.petal_width
            count = count + 1

    mean = [float(x1)/float(count), float(x2)/float(count), float(x3)/float(count), float(x4)/float(count)]

    if flag =='test':
        intermediateArr = np.asarray(data, dtype="float")
    else:
        intermediateArr = np.asarray(tempData, dtype="float")

    standardDeviation = [np.std(intermediateArr[:,0]),np.std(intermediateArr[:,1]),np.std(intermediateArr[:,2]),
                            np.std(intermediateArr[:,3])]

    return data, mean, standardDeviation

computeDistances('test.arff','train.arff')
