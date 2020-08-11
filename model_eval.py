import os
import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import dataset as ds
import config as cf
from dagmm import hyper_params, CompressNet, EstimateNet

has_threshold = True

def load_model():
    compressor = CompressNet(
        hyper_params.input_dim,
        hyper_params.cn_hidden1_dim,
        hyper_params.cn_hidden2_dim,
        hyper_params.cn_hidden3_dim,
        hyper_params.zc_dim
    )

    estimator = EstimateNet(
        hyper_params.zc_dim + 2,
        hyper_params.en_hidden_dim,
        hyper_params.dropout_p,
        hyper_params.mixture_dim,
        hyper_params.lam1,
        hyper_params.lam2
    )

    try:
        checkpoint = torch.load(os.path.join(cf.model_dir, cf.model_filename))
        compressor.load_state_dict(checkpoint[cf.compressor_state])
        estimator.load_state_dict(checkpoint[cf.estimator_state])
    except Exception as e:
        print('Failed to load model: %s' % (e))
        exit(1)
    
    compressor.eval()
    estimator.eval()
    return compressor, estimator

def compute_threshold(compressor, estimator, intrusion_ds):
    energies = np.zeros(shape=(len(intrusion_ds)))
    step = 0
    energy_interval = 50

    train_loader = data.DataLoader(intrusion_ds, batch_size=10)
    with torch.no_grad():
        for x, y in train_loader:
            z, x_hat = compressor(x)
            gamma = estimator(z)
            
            m_prob = estimator.mixture_prob(gamma)
            m_mean = estimator.mixture_mean(gamma, z)
            m_cov = estimator.mixture_covar(gamma, z, m_mean)

            for i in range(z.shape[0]):
                zi = z[i].unsqueeze(1)
                sample_energy = estimator.sample_energy(m_prob, m_mean, m_cov, zi)
                #print('sample energy: ', sample_energy)

                energies[step] = sample_energy.detach().item()
                step += 1

            if step % energy_interval == 0:
                print('Iteration: %d    sample energy: %.4f' % (step, sample_energy))
    
    threshold = np.percentile(energies, 80)
    print('threshold: %.4f' %(threshold))

def main():
    intrusion_ds = ds.Intrusion_Dataset('train')
    compressor, estimator = load_model()

    if has_threshold == False:
        threshold = compute_threshold(compressor, estimator, intrusion_ds)
    else:
        #threshold = -6.2835
        threshold = -4.2050
    
    print('threshold: ', threshold)

    intrusion_ds.set_mode('test')
    test_loader = data.DataLoader(intrusion_ds, batch_size=10)

    scores = np.zeros(shape=(len(intrusion_ds), 2))
    step = 0
    with torch.no_grad():
        for x, y in test_loader:
            z, x_hat = compressor(x)
            gamma = estimator(z)

            m_prob = estimator.mixture_prob(gamma)
            m_mean = estimator.mixture_mean(gamma, z)
            m_cov = estimator.mixture_covar(gamma, z, m_mean)

            for i in range(z.shape[0]):
                zi = z[i].unsqueeze(1)
                sample_energy = estimator.sample_energy(m_prob, m_mean, m_cov, zi)
                se = sample_energy.detach().item()

                scores[step] = [int(y[i]), int(se > threshold)]
                step += 1
    
    accuracy = accuracy_score(scores[:, 0], scores[:, 1])
    precision, recall, fscore, support = precision_recall_fscore_support(scores[:, 0], scores[:, 1], average='binary')
    print('Accuracy: %.4f  Precision: %.4f  Recall: %.4f  F-score: %.4f' % (accuracy, precision, recall, fscore))

if __name__ == "__main__":
    main()
