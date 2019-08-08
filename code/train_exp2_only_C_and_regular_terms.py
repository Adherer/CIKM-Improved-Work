import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from collections import defaultdict
import itertools
import logging
import random
import sys
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchnet.meter import ConfusionMeter
from data_prep.msda_preprocessed_amazon_dataset import get_msda_amazon_datasets
from man_models import *
import utils
from sklearn.manifold import TSNE
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from center_loss import CenterLoss

random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)

# save model and logging
if not os.path.exists(opt.exp2_target_model_save_file):
    os.makedirs(opt.exp2_target_model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.exp2_target_model_save_file, '2019.07.08_exp2_log.txt'))
log.addHandler(fh)

# output options
log.info(opt)
# ---------------------- some settings ---------------------- #


# ---------------------- training ---------------------- #
def train(train_sets, test_sets):

    # ---------------------- dataloader ---------------------- #
    # dataset loaders
    train_loaders, train_iters, test_loaders, test_iters = {}, {}, {}, {}

    # 加载有label的训练数据
    for domain in opt.domains:
        train_loaders[domain] = DataLoader(train_sets[domain],
                                           opt.batch_size, shuffle=True)
        train_iters[domain] = iter(train_loaders[domain])

        test_loaders[domain] = DataLoader(test_sets[domain],
                                          opt.batch_size, shuffle=False)
        test_iters[domain] = iter(test_loaders[domain])

    # ---------------------- model initialization ---------------------- #
    F_d = {}
    C = None
    if opt.model.lower() == 'mlp':
        for domain in opt.domains:
            F_d[domain] = MlpFeatureExtractor(opt.feature_num, opt.F_hidden_sizes,
                opt.domain_hidden_size, opt.dropout, opt.F_bn)

    C = SentimentClassifier(opt.C_layers,opt.domain_hidden_size, opt.domain_hidden_size, opt.num_labels,
                            opt.dropout, opt.C_bn)
    # 转移到gpu上
    C = C.to(opt.device)
    for f_d in F_d.values():
        f_d = f_d.to(opt.device)

    criterion_cent = CenterLoss(num_classes=2, feat_dim=opt.domain_hidden_size, use_gpu=True)
    optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=0.5)
    optimizer = optim.Adam(itertools.chain(*map(list, [C.parameters()] + [f.parameters() for f in F_d.values()])),
                           lr=opt.learning_rate)

    # training
    correct, total = defaultdict(int), defaultdict(int)
    # D accuracy
    d_correct, d_total = 0, 0

    best_acc = 0.0
    best_acc_dict = {}
    margin = 3
    margin_lambda = 0.1
    # center_loss_weight_cent = 0.1
    for epoch in range(opt.max_epoch):
        C.train()
        for f in F_d.values():
            f.train()

        # conceptually view 1 epoch as 1 epoch of the first domain
        num_iter = len(train_loaders[opt.domains[0]])
        # First_stage
        for _ in tqdm(range(num_iter)):
            for f_d in F_d.values():
                f_d.zero_grad()
            C.zero_grad()
            optimizer_centloss.zero_grad()

            for domain in opt.domains:
                inputs, targets = utils.endless_get_next_batch(
                    train_loaders, train_iters, domain)
                targets = targets.to(opt.device)
                domain_feat = F_d[domain](inputs)
                visual_feature, c_outputs = C(domain_feat)
                # loss_cent = criterion_cent(visual_feature, targets)
                # loss_cent *= center_loss_weight_cent
                loss_cent = 0.0
                loss_part_1 = functional.nll_loss(c_outputs, targets)
                targets = targets.unsqueeze(1)
                targets_onehot = torch.FloatTensor(opt.batch_size, 2)
                targets_onehot.zero_()
                targets_onehot.scatter_(1, targets.cpu(), 1)
                targets_onehot = targets_onehot.to(opt.device)
                loss_part_2 = margin_lambda * margin_regularization(inputs, targets_onehot, F_d[domain], C, margin)
                # loss_part_2 = 0.0

                print("loss_part_1: " + str(loss_part_1))
                print("loss_part_2: " + str(loss_part_2))
                print("loss_cent: " + str(loss_cent))
                l_c = loss_part_1 + loss_part_2 + loss_cent
                l_c.backward()
                _, pred = torch.max(c_outputs, 1)
                total[domain] += targets.size(0)
                correct[domain] += (pred == targets).sum().item()

            optimizer.step()
            # for param in criterion_cent.parameters():
            #     param.grad.data *= (1. / center_loss_weight_cent)
            optimizer_centloss.step()

        # end of epoch
        log.info('Ending epoch {}'.format(epoch+1))
        if d_total > 0:
            log.info('D Training Accuracy: {}%'.format(100.0*d_correct/d_total))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.domains))
        log.info('\t'.join([str(100.0*correct[d]/total[d]) for d in opt.domains]))

    log.info('Evaluating test sets:')
    test_acc = {}
    for domain in opt.domains:
        test_acc[domain] = evaluate(domain, test_loaders[domain], F_d[domain], C)
    avg_test_acc = sum([test_acc[d] for d in opt.domains]) / len(opt.domains)
    log.info(f'Average test accuracy: {avg_test_acc}')

    if avg_test_acc > best_acc:
        log.info(f'New best Average test accuracy: {avg_test_acc}')
        best_acc = avg_test_acc
        best_acc_dict = test_acc
        for d in opt.domains:
            if d in F_d:
                torch.save(F_d[d].state_dict(),
                           '{}/net_F_d_{}.pth'.format(opt.exp2_model_save_file, d))
        torch.save(C.state_dict(),
                   '{}/netC.pth'.format(opt.exp2_model_save_file))

    log.info(f'Loading model for feature visualization from {opt.exp2_model_save_file}...')
    for domain in opt.domains:
        F_d[domain].load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                                            f'net_F_d_{domain}.pth')))
    num_iter = len(train_loaders[opt.domains[0]])
    visual_features, senti_labels = get_visual_features(num_iter, test_loaders, test_iters, F_d, C)
    # visual_features, senti_labels = get_visual_features(num_iter, train_loaders, train_iters, F_d)
    return best_acc, best_acc_dict, visual_features, senti_labels


# 这个正则明天再来检查一下正确性(与公式对比)，然后再画出T-sne的图来
def margin_regularization(inputs, targets, F_d, C, margin):
    private_features = F_d(inputs)
    visual_feature, _ = C(private_features)
    graph_source = torch.sum(targets[:, None, :] * targets[None, :, :], 2)
    distance_source = torch.mean((visual_feature[:, None, :] - visual_feature[None, :, :]) ** 2, 2)
    margin_loss = torch.mean(graph_source * distance_source + (1-graph_source)*F.relu(margin - distance_source))
    return margin_loss


def get_visual_features(num_iter, test_loaders, test_iters, F_d, C):
    visual_features, senti_labels = None, None
    for _ in tqdm(range(num_iter)):
        for domain in opt.domains:
            d_inputs, targets = utils.endless_get_next_batch(
                test_loaders, test_iters, domain)
            private_features = F_d[domain](d_inputs)
            # private_features, _ = C(private_features)
            if visual_features is None:
                visual_features = private_features
                senti_labels = targets
            else:
                visual_features = torch.cat([visual_features, private_features], 0)
                senti_labels = torch.cat([senti_labels, targets], 0)

    return visual_features, senti_labels


def evaluate(name, loader, F_d, C):
    if F_d:
        F_d.eval()
    C.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(opt.num_labels)
    for inputs, targets in tqdm(it):
        targets = targets.to(opt.device)
        if not F_d:
            # unlabeled domain
            d_features = torch.zeros(len(targets), opt.domain_hidden_size).to(opt.device)
        else:
            d_features = F_d(inputs)
        _, outputs = C(d_features)
        _, pred = torch.max(outputs, 1)
        confusion.add(pred.data, targets.data)
        total += targets.size(0)
        correct += (pred == targets).sum().item()
    acc = correct / total
    log.info('{}: Accuracy on {} samples: {}%'.format(name, total, 100.0*acc))
    log.debug(confusion.conf)
    return acc


def scatter(data, label, dir, file_name, mus=None, mark_size=2):
    if not os.path.exists(dir):
        os.makedirs(dir)

    if label.ndim == 2:
        label = np.argmax(label, axis=1)

    df = pd.DataFrame(data={'x': data[:, 0], 'y': data[:, 1], 'class': label})
    sns_plot = sns.lmplot('x', 'y', data=df, hue='class', fit_reg=False, scatter_kws={'s': mark_size})
    sns_plot.savefig(os.path.join(dir, file_name))
    if mus is not None:
        df_mus = pd.DataFrame(
            data={'x': mus[:, 0], 'y': mus[:, 1], 'class': np.asarray(range(mus.shape[0])).astype(np.int32)})
        sns_plot_mus = sns.lmplot('x', 'y', data=df_mus, hue='class', fit_reg=False, scatter_kws={'s': mark_size * 20})
        sns_plot_mus.savefig(os.path.join(dir, 'mus_' + file_name))


def t_sne(domain, visual_features, senti_labels):
    compressed_visual_features = TSNE(random_state=2019).fit_transform(visual_features)
    scatter(data=compressed_visual_features, label=senti_labels,
            dir="./result",
            file_name=domain + "_center_loss.png")
    plt.show()


def main():
    unlabeled_domains = ['books', 'dvd', 'electronics', 'kitchen']
    test_acc_dict = {}
    i = 1
    opt.shared_lambd = 0.025
    opt.private_lambd = 0.025

    ave_acc = 0.0
    for u_domain in unlabeled_domains:
        opt.domains = ['books', 'dvd', 'electronics', 'kitchen']
        opt.num_labels = 2
        opt.unlabeled_domains = u_domain.split()
        opt.dev_domains = u_domain.split()
        opt.domains.remove(u_domain)
        opt.exp2_model_save_file = './save/man_exp2/exp' + str(i)
        if not os.path.exists(opt.exp2_model_save_file):
            os.makedirs(opt.exp2_model_save_file)

        datasets = {}
        raw_unlabeled_sets = {}
        log.info(f'Loading {opt.dataset} Datasets...')
        for domain in opt.all_domains:
            datasets[domain], raw_unlabeled_sets[domain] = get_msda_amazon_datasets(
                opt.prep_amazon_file, domain, 1, opt.feature_num)
        opt.num_labels = 2
        log.info(f'Done Loading {opt.dataset} Datasets.')
        log.info(f'Domains: {opt.domains}')

        train_sets, dev_sets, test_sets, unlabeled_sets = {}, {}, {}, {}
        for domain in opt.domains:
            train_sets[domain] = datasets[domain]
            test_sets[domain] = raw_unlabeled_sets[domain]

        best_acc, best_acc_dict, visual_features, senti_labels = train(train_sets, test_sets)
        print(visual_features.shape)
        print(senti_labels.shape)
        log.info(f'Training done...')
        for key in best_acc_dict:
            log.info(str(key) + ": " + str(best_acc_dict[key]))
        test_acc_dict[u_domain] = best_acc
        i += 1

        # ---------------------- 可视化 ---------------------- #
        log.info(f'feature visualization')

        print("Computing t-SNE 2D embedding")
        t0 = time()
        t_sne(u_domain, visual_features.detach().cpu().numpy(), senti_labels.detach().cpu().numpy())
        print("t-SNE 2D embedding of the digits (time %.2fs)" % (time() - t0))

    log.info(f'Training done...')
    log.info(f'test_acc\'s result is: ')
    for key in test_acc_dict:
        log.info(str(key) + ": " + str(test_acc_dict[key]))
        ave_acc += test_acc_dict[key]

    log.info(f'ave_test_acc\'s result is: ')
    log.info(ave_acc / 4)


if __name__ == '__main__':
    main()

