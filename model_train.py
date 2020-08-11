import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.utils.data as data

import dataset as ds
import config as cf
import dagmm as dg
from dagmm import hyper_params, CompressNet, EstimateNet

def plot_loss_moment(losses):
    _, ax = plt.subplots(figsize=(16, 9), dpi=80)
    ax.plot(losses, 'blue', label='train', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(cf.image_dir, 'loss_dagmm.png'))

def main():
    cf.setup_dirs()

    train_ds = ds.Intrusion_Dataset('train')
    train_loader = data.DataLoader(train_ds, batch_size=hyper_params.batch_size, shuffle=True, drop_last=True)

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

    losses = []
    loss_plot_interval = 20
    steps = 0
    loss_total = 0

    compressor_opt = torch.optim.Adam(compressor.parameters(), lr=1e-4, amsgrad=True)
    estimator_opt = torch.optim.Adam(estimator.parameters(), lr=1e-4, amsgrad=True)

    for i in range(hyper_params.epoch_num):
        for x, _ in train_loader:
            steps += 1

            z, x_hat = compressor(x)
            loss = compressor.reconstruct_error(x, x_hat)

            gamma = estimator(z)
            loss += estimator.energy(gamma, z)
            loss_total += loss.detach().item()

            compressor_opt.zero_grad()
            estimator_opt.zero_grad()

            loss.backward()
            compressor_opt.step()
            estimator_opt.step()

            if steps % loss_plot_interval == 0:
                print('Epoch: %d  step: %d  loss: %.4f' % (i, steps, loss_total/loss_plot_interval))
                losses.append(loss_total / loss_plot_interval)
                loss_total = 0

    plot_loss_moment(losses)
    torch.save({
        cf.compressor_state: compressor.state_dict(),
        cf.estimator_state: estimator.state_dict(),
        cf.compressor_opt_state: compressor_opt.state_dict(),
        cf.estimator_opt_state: estimator_opt.state_dict()
    }, os.path.join(cf.model_dir, cf.model_filename))

if __name__ == "__main__":
    main()