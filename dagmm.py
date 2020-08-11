import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import config as cf

class HyperParams:
    def __init__(self):
        self.epoch_num = 5
        self.batch_size = 100
        self.input_dim = 118
        self.cn_hidden1_dim = 60
        self.cn_hidden2_dim = 30
        self.cn_hidden3_dim = 10
        self.zc_dim = 1
        self.en_hidden_dim = 10
        self.mixture_dim = 2
        self.dropout_p = 0.5
        self.lam1 = 0.1
        self.lam2 = 0.005

hyper_params = HyperParams()

def relative_euclidean_distance(x, x_hat):
    x_ = x.unsqueeze(1)
    x_hat_ = x_hat.unsqueeze(1)
    # d1 shape: [batch_size, 1]
    d1 = torch.cdist(x_, x_hat_).squeeze(1)
    # d2 shape: [batch_size, 1]
    d2 = torch.cdist(x_, torch.zeros(x_.shape)).squeeze(1)
    return d1/d2


class CompressNet(nn.Module):
    def __init__(self, x_dim, hidden1_dim, hidden2_dim, hidden3_dim, zc_dim):
        super(CompressNet, self).__init__()

        self.encoder_layer1 = nn.Linear(x_dim, hidden1_dim)
        self.encoder_layer2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.encoder_layer3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.encoder_layer4 = nn.Linear(hidden3_dim, zc_dim)

        self.decoder_layer1 = nn.Linear(zc_dim, hidden3_dim)
        self.decoder_layer2 = nn.Linear(hidden3_dim, hidden2_dim)
        self.decoder_layer3 = nn.Linear(hidden2_dim, hidden1_dim)
        self.decoder_layer4 = nn.Linear(hidden1_dim, x_dim)

    def forward(self, x):
        h = self.encoder_layer1(x)
        h = self.encoder_layer2(h)
        h = self.encoder_layer3(h)
        zc = self.encoder_layer4(h)

        h = self.decoder_layer1(zc)
        h = self.decoder_layer2(h)
        h = self.decoder_layer3(h)
        x_hat = self.decoder_layer4(h)

        # ed shape: [batch_size, 1]
        ed = relative_euclidean_distance(x, x_hat)
        cos = nn.CosineSimilarity(dim=1)
        # cosim shape: [batch_size, 1]
        cosim = cos(x, x_hat).unsqueeze(1)
        # z shape: [batch_size, zc_dim+2]
        z = torch.cat((zc, ed, cosim), dim=1)
        assert zc.shape[0] == z.shape[0]
        assert zc.shape[1] == z.shape[1] - 2

        return z, x_hat
    
    def reconstruct_error(self, x, x_hat):
        e = torch.tensor(0.0)
        for i in range(x.shape[0]):
            e += torch.dist(x[i], x_hat[i])
        return e / x.shape[0]

class EstimateNet(nn.Module):
    def __init__(self, z_dim, hidden_dim, dropout_p, mixture_dim, lam1, lam2):
        super(EstimateNet, self).__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.mixture_dim = mixture_dim
        self.lam1 = lam1
        self.lam2 = lam2

        self.layer1 = nn.Linear(z_dim, hidden_dim)
        self.drop = nn.Dropout(dropout_p)
        self.layer2 = nn.Linear(hidden_dim, mixture_dim)

    def forward(self, z):
        h = self.layer1(z)
        h = torch.tanh(h)
        h = self.drop(h)
        h = self.layer2(h)
        # gamma shape: [batch_size, mixture_dim]
        gamma = F.softmax(h, dim=1)
        return gamma

    # return shape: [mixture_dim]
    def mixture_prob(self, gamma):
        n = gamma.shape[0]
        return torch.sum(gamma, dim=0) / n

    # return shape: [mixture_dim, z_dim]
    def mixture_mean(self, gamma, z):
        gamma_t = torch.t(gamma)
        miu = torch.mm(gamma_t, z)
        miu = miu / torch.sum(gamma_t, dim=1).unsqueeze(1)
        return miu

    # return shape: [mixture_dim, z_dim, z_dim]
    def mixture_covar(self, gamma, z, miu):
        cov = torch.zeros((self.mixture_dim, self.z_dim, self.z_dim))
        # z_t shape: [z_dim, batch_size]
        z_t = torch.t(z)
        for k in range(self.mixture_dim):
            miu_k = miu[k].unsqueeze(1)
            # dm shape: [z_dim, batch_size]
            dm = z_t - miu_k
            # gamma_k shape: [batch_size, batch_size]
            gamma_k = torch.diag(gamma[:, k])
            # cov_k shape: [z_dim, z_dim]
            cov_k = torch.chain_matmul(dm, gamma_k, torch.t(dm))
            cov_k = cov_k / torch.sum(gamma[:, k])
            cov[k] = cov_k
        return cov
    
    # m_prob shape: [mixture_dim]
    # m_mean shape: [mixture_dim, z_dim]
    # m_cov shape: [mixture_dim, z_dim, z_dim]
    # zi shape: [z_dim, 1]
    def sample_energy(self, m_prob, m_mean, m_cov, zi):
        e = torch.tensor(0.0)
        cov_eps = torch.eye(m_mean.shape[1]) * (1e-12)
        for k in range(self.mixture_dim):
            # miu_k shape: [z_dim, 1]
            miu_k = m_mean[k].unsqueeze(1)
            d_k = zi - miu_k

            # solve the singular covariance
            inv_cov = torch.inverse(m_cov[k] + cov_eps)
            e_k = torch.exp(-0.5 * torch.chain_matmul(torch.t(d_k), inv_cov, d_k))
            e_k = e_k / torch.sqrt(torch.abs(torch.det(2 * math.pi * m_cov[k])))
            e_k = e_k * m_prob[k]
            e += e_k.squeeze()
        return - torch.log(e)

    def energy(self, gamma, z):
        m_prob = self.mixture_prob(gamma)
        m_mean = self.mixture_mean(gamma, z)
        m_cov = self.mixture_covar(gamma, z, m_mean)

        e = torch.tensor(0.0)
        for i in range(z.shape[0]):
            zi = z[i].unsqueeze(1)
            ei = self.sample_energy(m_prob, m_mean, m_cov, zi)
            e += ei
        
        p = torch.tensor(0.0)
        for k in range(self.mixture_dim):
            cov_k = m_cov[k]
            p_k = torch.sum(1 / torch.diagonal(cov_k, 0))
            p += p_k
        
        return (self.lam1 / z.shape[0]) * e + self.lam2 * p
