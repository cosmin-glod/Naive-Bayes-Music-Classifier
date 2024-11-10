import numpy as np
from scipy.stats import multivariate_normal

def gaussianNaiveBayes(trainData, trainLabels, testData):
    bluesTrain = trainData[np.where(trainLabels == "blues")]
    classicalTrain = trainData[np.where(trainLabels == "classical")]
    countryTrain = trainData[np.where(trainLabels == "country")]
    discoTrain = trainData[np.where(trainLabels == "disco")]
    hiphopTrain = trainData[np.where(trainLabels == "hiphop")]
    jazzTrain = trainData[np.where(trainLabels == "jazz")]
    metalTrain = trainData[np.where(trainLabels == "metal")]
    popTrain = trainData[np.where(trainLabels == "pop")]
    reggaeTrain = trainData[np.where(trainLabels == "reggae")]
    rockTrain = trainData[np.where(trainLabels == "rock")]

    meanBlues = np.mean(bluesTrain, axis = 0)
    covBlues = np.cov(bluesTrain, rowvar = False)

    meanClassical = np.mean(classicalTrain, axis = 0)
    covClassical = np.cov(classicalTrain, rowvar = False)

    meanCountry = np.mean(countryTrain, axis = 0)
    covCountry = np.cov(countryTrain, rowvar = False)

    meanDisco = np.mean(discoTrain, axis = 0)
    covDisco = np.cov(discoTrain, rowvar = False)

    meanHiphop = np.mean(hiphopTrain, axis = 0)
    covHiphop = np.cov(hiphopTrain, rowvar = False)

    meanJazz = np.mean(jazzTrain, axis = 0)
    covJazz = np.cov(jazzTrain, rowvar = False)

    meanMetal = np.mean(metalTrain, axis = 0)
    covMetal = np.cov(metalTrain, rowvar = False)

    meanPop = np.mean(popTrain, axis = 0)
    covPop = np.cov(popTrain, rowvar = False)

    meanReggae = np.mean(reggaeTrain, axis = 0)
    covReggae = np.cov(reggaeTrain, rowvar = False)

    meanRock = np.mean(rockTrain, axis = 0)
    covRock = np.cov(rockTrain, rowvar = False)

    probBlues = multivariate_normal.pdf(testData, meanBlues, covBlues) * 1 / 10
    probClassical = multivariate_normal.pdf(testData, meanClassical, covClassical) * 1 / 10
    probCountry = multivariate_normal.pdf(testData, meanCountry, covCountry) * 1 / 10
    probDisco = multivariate_normal.pdf(testData, meanDisco, covDisco) * 1 / 10
    probHiphop = multivariate_normal.pdf(testData, meanHiphop, covHiphop) * 1 / 10
    probJazz = multivariate_normal.pdf(testData, meanJazz, covJazz) * 1 / 10
    probMetal = multivariate_normal.pdf(testData, meanMetal, covMetal) * 1 / 10
    probPop = multivariate_normal.pdf(testData, meanPop, covPop) * 1 / 10
    probReggae = multivariate_normal.pdf(testData, meanReggae, covReggae) * 1 / 10
    probRock = multivariate_normal.pdf(testData, meanRock, covRock) * 1 / 10

    np.seterr(divide = 'ignore')
    logProb = np.log(np.vstack((probBlues, probClassical, probCountry, probDisco, probHiphop, probJazz, probMetal, probPop, probReggae, probRock)))

    return logProb


