import librosa
import numpy as np

def chroma(data):
    computedChroma = librosa.feature.chroma_stft(y = data)
    mean = np.mean(computedChroma)
    var = np.var(computedChroma)
    return mean, var

def rootMeanSquareEnergy(data):
    rms = librosa.feature.rms(y = data)
    mean = np.mean(rms)
    var = np.var(rms)
    return mean, var

def spectralCentroidAndBandwidth(data):
    # spectral centroid
    centroid = librosa.feature.spectral_centroid(data)
    centroidMean = np.mean(centroid)
    centroidVar = np.var(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(data)
    bandwidthMean = np.mean(bandwidth)
    bandwidthVar = np.var(bandwidth)

    return centroidMean, centroidVar, bandwidthMean, bandwidthVar




