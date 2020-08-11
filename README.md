# Anomaly detection based on Deep Autoencoding Gaussian Mixture Model (DAGMM)

# Description
According to [Anomaly Detection: A Survey](http://cucis.ece.northwestern.edu/projects/DMS/publications/AnomalyDetection.pdf), unsupervised anomaly methods can be classified into following categories:
- Classification based techniques
- Clustering based techniques
- Nearest neighbor based techniques
- Statistical techniques

Recently, reconstruction based techniques have emerged as a new kind of anomaly detection techniques.

DAGMM can be viewed as spanning both reconstruction based and clustering based techniques, which makes it more interesting. According to [this paper](https://openreview.net/forum?id=BJJLHbb0-), DAGMM comprises of two networks:
- Compression network: It consists of an encoder and a decoder. The encoder can reduce the input into a low-dimension representation and the decoder can reconstruct the input such that reconstruction errors can be computed. What's innovative is that the paper proposed to construct a combined representation consisting of both the low-dimension representation and two error features.
- Estimation network: Used to estimate the parameters for Gaussian mixtures based on the combined low-dimension representation. A sample energy for each low-dimension data point can be computed as an indicator for anomalies.

# Dependencies
The code in this repo was implemented and tested with PyTorch 1.5.1 and Python 3.8.3