import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from collections import defaultdict
import itertools
import logging
import pickle
import random
import sys
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader
from torchnet.meter import ConfusionMeter

from options import opt
random.seed(opt.random_seed)
np.random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed_all(opt.random_seed)

from data_prep.fdu_mtl_dataset import get_fdu_mtl_datasets, FduMtlDataset
from man_models import *
from man_vocab import Vocab
import utils
from time import time
import torch.nn.functional as F
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# save model and logging
if not os.path.exists(opt.exp3_model_save_file):
    os.makedirs(opt.exp3_model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.exp3_model_save_file, 'log.txt'))
log.addHandler(fh)
# output options
log.info(opt)


def train(vocab, train_sets, dev_sets, test_sets, unlabeled_sets):
    """
    train_sets, dev_sets, test_sets: dict[domain] -> AmazonDataset
    For unlabeled domains, no train_sets are available
    """
    # dataset loaders
    train_loaders, unlabeled_loaders = {}, {}
    train_iters, unlabeled_iters = {}, {}
    dev_loaders, test_loaders = {}, {}
    my_collate = utils.sorted_collate if opt.model=='lstm' else utils.unsorted_collate
    for domain in opt.domains:
        train_loaders[domain] = DataLoader(train_sets[domain],
                opt.batch_size, shuffle=True, collate_fn=my_collate)
        train_iters[domain] = iter(train_loaders[domain])
    for domain in opt.dev_domains:
        dev_loaders[domain] = DataLoader(dev_sets[domain],
                opt.batch_size, shuffle=False, collate_fn=my_collate)
        test_loaders[domain] = DataLoader(test_sets[domain],
                opt.batch_size, shuffle=False, collate_fn=my_collate)
    for domain in opt.all_domains:
        if domain in opt.unlabeled_domains:
            uset = unlabeled_sets[domain]
        else:
            # for labeled domains, consider which data to use as unlabeled set
            if opt.unlabeled_data == 'both':
                uset = ConcatDataset([train_sets[domain], unlabeled_sets[domain]])
            elif opt.unlabeled_data == 'unlabeled':
                uset = unlabeled_sets[domain]
            elif opt.unlabeled_data == 'train':
                uset = train_sets[domain]
            else:
                raise Exception(f'Unknown options for the unlabeled data usage: {opt.unlabeled_data}')
        unlabeled_loaders[domain] = DataLoader(uset,
                opt.batch_size, shuffle=True, collate_fn = my_collate)
        unlabeled_iters[domain] = iter(unlabeled_loaders[domain])

    # model
    F_s = None
    F_d = {}
    C, D = None, None
    if opt.model.lower() == 'dan':
        F_s = DanFeatureExtractor(vocab, opt.F_layers, opt.shared_hidden_size,
                               opt.sum_pooling, opt.dropout, opt.F_bn)
        for domain in opt.domains:
            F_d[domain] = DanFeatureExtractor(vocab, opt.F_layers, opt.domain_hidden_size,
                                           opt.sum_pooling, opt.dropout, opt.F_bn)
    elif opt.model.lower() == 'lstm':
        F_s = LSTMFeatureExtractor(vocab, opt.F_layers, opt.shared_hidden_size,
                                   opt.dropout, opt.bdrnn, opt.attn)
        for domain in opt.domains:
            F_d[domain] = LSTMFeatureExtractor(vocab, opt.F_layers, opt.domain_hidden_size,
                                               opt.dropout, opt.bdrnn, opt.attn)
    elif opt.model.lower() == 'cnn':
        F_s = CNNFeatureExtractor(vocab, opt.F_layers, opt.shared_hidden_size,
                                  opt.kernel_num, opt.kernel_sizes, opt.dropout)
        for domain in opt.domains:
            F_d[domain] = CNNFeatureExtractor(vocab, opt.F_layers, opt.domain_hidden_size,
                                              opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise Exception(f'Unknown model architecture {opt.model}')

    C = SentimentClassifier(opt.C_layers, opt.shared_hidden_size + opt.domain_hidden_size,
            opt.shared_hidden_size + opt.domain_hidden_size, opt.num_labels,
            opt.dropout, opt.C_bn)
    D = DomainClassifier(opt.D_layers, opt.shared_hidden_size, opt.shared_hidden_size,
                         len(opt.all_domains), opt.loss, opt.dropout, opt.D_bn)

    F_s, C, D = F_s.to(opt.device), C.to(opt.device), D.to(opt.device)
    for f_d in F_d.values():
        f_d = f_d.to(opt.device)
    # optimizers
    optimizer = optim.Adam(itertools.chain(*map(list, [F_s.parameters() if F_s else [], C.parameters()] + [f.parameters() for f in F_d.values()])), lr=opt.learning_rate)
    optimizerD = optim.Adam(D.parameters(), lr=opt.D_learning_rate)

    # testing
    if opt.test_only:
        log.info(f'Loading model from {opt.exp3_model_save_file}...')
        if F_s:
            F_s.load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                                           f'netF_s.pth')))
        for domain in opt.all_domains:
            if domain in F_d:
                F_d[domain].load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                        f'net_F_d_{domain}.pth')))
        C.load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                                                  f'netC.pth')))
        D.load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                                                  f'netD.pth')))

        log.info('Evaluating validation sets:')
        acc = {}
        for domain in opt.all_domains:
            acc[domain] = evaluate(domain, dev_loaders[domain],
                                   F_s, F_d[domain] if domain in F_d else None, C)
        avg_acc = sum([acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average validation accuracy: {avg_acc}')
        log.info('Evaluating test sets:')
        test_acc = {}
        for domain in opt.all_domains:
            test_acc[domain] = evaluate(domain, test_loaders[domain],
                    F_s, F_d[domain] if domain in F_d else None, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average test accuracy: {avg_test_acc}')
        return {'valid': acc, 'test': test_acc}

    # training
    best_acc, best_avg_acc = defaultdict(float), 0.0
    for epoch in range(opt.max_epoch):
        F_s.train()
        C.train()
        D.train()
        LAMBDA = 3
        lambda1 = 0.1
        lambda2 = 0.1

        for f in F_d.values():
            f.train()

        # training accuracy
        correct, total = defaultdict(int), defaultdict(int)
        # D accuracy
        shared_d_correct, private_d_correct, d_total = 0, 0, 0
        # conceptually view 1 epoch as 1 epoch of the first domain
        num_iter = len(train_loaders[opt.domains[0]])
        for i in tqdm(range(num_iter)):
            # D iterations
            utils.freeze_net(F_s)
            map(utils.freeze_net, F_d.values())
            utils.freeze_net(C)
            utils.unfreeze_net(D)
            # WGAN n_critic trick since D trains slower
            # ********************** D iterations on all domains ********************** #
            # ---------------------- update D with D gradients on all domains ---------------------- #
            n_critic = opt.n_critic
            if opt.wgan_trick:
                if opt.n_critic>0 and ((epoch==0 and i<25) or i%500==0):
                    n_critic = 100

            for _ in range(n_critic):
                D.zero_grad()
                loss_d = {}
                # train on both labeled and unlabeled domains
                for domain in opt.all_domains:
                    # targets not usedndless_get_next_batch(
                    #                             unlabeled_loaders, unlabeled_iters, domain)
                    d_inputs, _ = utils.endless_get_next_batch(
                        unlabeled_loaders, unlabeled_iters, domain)
                    d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs[1]))
                    shared_feat = F_s(d_inputs)
                    shared_d_outputs = D(shared_feat)
                    _, shared_pred = torch.max(shared_d_outputs, 1)
                    if domain != opt.dev_domains[0]:
                        private_feat = F_d[domain](d_inputs)
                        private_d_outputs = D(private_feat)
                        _, private_pred = torch.max(private_d_outputs, 1)

                    d_total += len(d_inputs[1])
                    if opt.loss.lower() == 'l2':
                        _, tgt_indices = torch.max(d_targets, 1)
                        shared_d_correct += (shared_pred == tgt_indices).sum().item()
                        shared_l_d = functional.mse_loss(shared_d_outputs, d_targets)
                        private_l_d = 0.0
                        if domain != opt.dev_domains[0]:
                            private_d_correct += (private_pred == tgt_indices).sum().item()
                            private_l_d = functional.mse_loss(private_d_outputs, d_targets) / len(opt.domains)

                        l_d_sum = shared_l_d + private_l_d
                        l_d_sum.backward()
                    else:
                        shared_d_correct += (shared_pred == d_targets).sum().item()
                        shared_l_d = functional.nll_loss(shared_d_outputs, d_targets)
                        private_l_d = 0.0
                        if domain != opt.dev_domains[0]:
                            private_d_correct += (private_pred == d_targets).sum().item()
                            private_l_d = functional.nll_loss(private_d_outputs, d_targets) / len(opt.domains)
                        l_d_sum = shared_l_d + private_l_d
                        l_d_sum.backward()

                    loss_d[domain] = l_d_sum.item()
                optimizerD.step()

            # ---------------------- update D with C gradients on all domains ---------------------- #
            # ********************** D iterations on all domains ********************** #

            # ********************** F&C iteration ********************** #
            # ---------------------- update F_s & F_ds with C gradients on all labeled domains ---------------------- #
            # F&C iteration
            utils.unfreeze_net(F_s)
            map(utils.unfreeze_net, F_d.values())
            utils.unfreeze_net(C)
            utils.freeze_net(D)
            if opt.fix_emb:
                utils.freeze_net(F_s.word_emb)
                for f_d in F_d.values():
                    utils.freeze_net(f_d.word_emb)
            F_s.zero_grad()
            for f_d in F_d.values():
                f_d.zero_grad()
            C.zero_grad()
            for domain in opt.domains:
                inputs, targets = utils.endless_get_next_batch(
                    train_loaders, train_iters, domain)
                targets = targets.to(opt.device)
                shared_feat = F_s(inputs)
                domain_feat = F_d[domain](inputs)
                features = torch.cat((shared_feat, domain_feat), dim=1)
                c_outputs = C(features)
                loss_part_1 = functional.nll_loss(c_outputs, targets)

                targets = targets.unsqueeze(1)
                targets_onehot = torch.FloatTensor(opt.batch_size, 2)
                targets_onehot.zero_()
                targets_onehot.scatter_(1, targets.cpu(), 1)
                targets_onehot = targets_onehot.to(opt.device)
                loss_part_2 = lambda1 * margin_regularization(inputs, targets_onehot, F_d[domain], LAMBDA)

                loss_part_3 = -lambda2 * center_point_constraint(domain_feat, targets)
                print("lambda1: " + str(lambda1))
                print("lambda2: " + str(lambda2))
                print("loss_part_1: " + str(loss_part_1))
                print("loss_part_2: " + str(loss_part_2))
                print("loss_part_3: " + str(loss_part_3))
                l_c = loss_part_1 + loss_part_2 + loss_part_3
                l_c.backward(retain_graph=True)
                _, pred = torch.max(c_outputs, 1)
                total[domain] += targets.size(0)
                correct[domain] += (pred == targets).sum().item()
            # ---------------------- update F_s & F_ds with C gradients on all labeled domains ---------------------- #

            # ---------------------- update F_s with D gradients on all domains ---------------------- #
            # update F with D gradients on all domains
            for domain in opt.all_domains:
                d_inputs, _ = utils.endless_get_next_batch(
                        unlabeled_loaders, unlabeled_iters, domain)
                shared_feat = F_s(d_inputs)
                shared_d_outputs = D(shared_feat)
                if domain != opt.dev_domains[0]:
                    private_feat = F_d[domain](d_inputs)
                    private_d_outputs = D(private_feat)

                l_d_sum = None
                if opt.loss.lower() == 'gr':
                    d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs[1]))
                    shared_l_d = functional.nll_loss(shared_d_outputs, d_targets)
                    private_l_d, l_d_sum = 0.0, 0.0
                    if domain != opt.dev_domains[0]:
                        # 注意这边的loss function
                        private_l_d = functional.nll_loss(private_d_outputs, d_targets) * -1. / len(opt.domains)
                    if opt.shared_lambd > 0:
                        l_d_sum = shared_l_d * opt.shared_lambd * -1.
                    else:
                        l_d_sum = shared_l_d * -1.
                    if opt.private_lambd > 0:
                        l_d_sum += private_l_d * opt.private_lambd * -1.
                    else:
                        l_d_sum += private_l_d * -1.
                elif opt.loss.lower() == 'bs':
                    d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs[1]))
                    shared_l_d = functional.kl_div(shared_d_outputs, d_targets, size_average=False)
                    private_l_d, l_d_sum = 0.0, 0.0
                    if domain != opt.dev_domains[0]:
                        private_l_d = functional.kl_div(private_d_outputs, d_targets, size_average=False) \
                                      * -1. / len(opt.domains)
                    if opt.shared_lambd > 0:
                        l_d_sum = shared_l_d * opt.shared_lambd
                    else:
                        l_d_sum = shared_l_d
                    if opt.private_lambd > 0:
                        l_d_sum += private_l_d * opt.private_lambd
                    else:
                        l_d_sum += private_l_d
                elif opt.loss.lower() == 'l2':
                    d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs[1]))
                    shared_l_d = functional.mse_loss(shared_d_outputs, d_targets)
                    private_l_d, l_d_sum = 0.0, 0.0
                    if domain != opt.dev_domains[0]:
                        private_l_d = functional.mse_loss(private_d_outputs, d_targets) * -1. / len(opt.domains)
                    if opt.shared_lambd > 0:
                        l_d_sum = shared_l_d * opt.shared_lambd
                    else:
                        l_d_sum = shared_l_d
                    if opt.private_lambd > 0:
                        l_d_sum += private_l_d * opt.private_lambd
                    else:
                        l_d_sum += private_l_d
                l_d_sum.backward()

            optimizer.step()

        # end of epoch
        log.info('Ending epoch {}'.format(epoch + 1))
        if d_total > 0:
            log.info('shared D Training Accuracy: {}%'.format(100.0 * shared_d_correct / d_total))
            log.info('private D Training Accuracy(average): {}%'.format(100.0 * private_d_correct /
                                                                        len(opt.domains) / d_total))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.domains))
        log.info('\t'.join([str(100.0 * correct[d] / total[d]) for d in opt.domains]))

        # 验证集上验证实验
        log.info('Evaluating validation sets:')
        acc = {}
        for domain in opt.dev_domains:
            acc[domain] = evaluate(domain, dev_loaders[domain],
                                   F_s, F_d[domain] if domain in F_d else None, C)
        avg_acc = sum([acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average validation accuracy: {avg_acc}')

        # 测试集上验证实验
        log.info('Evaluating test sets:')
        test_acc = {}
        for domain in opt.dev_domains:
            test_acc[domain] = evaluate(domain, test_loaders[domain],
                                        F_s, F_d[domain] if domain in F_d else None, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average test accuracy: {avg_test_acc}')

        # 保存模型
        if avg_acc > best_avg_acc:
            log.info(f'New best average validation accuracy: {avg_acc}')
            best_acc['valid'] = acc
            best_acc['test'] = test_acc
            best_avg_acc = avg_acc
            with open(os.path.join(opt.exp3_model_save_file, 'options.pkl'), 'wb') as ouf:
                pickle.dump(opt, ouf)
            torch.save(F_s.state_dict(),
                       '{}/netF_s.pth'.format(opt.exp3_model_save_file))
            for d in opt.domains:
                if d in F_d:
                    torch.save(F_d[d].state_dict(),
                               '{}/net_F_d_{}.pth'.format(opt.exp3_model_save_file, d))
            torch.save(C.state_dict(),
                       '{}/netC.pth'.format(opt.exp3_model_save_file))
            torch.save(D.state_dict(),
                    '{}/netD.pth'.format(opt.exp3_model_save_file))

    # end of training
    log.info(f'Best average validation accuracy: {best_avg_acc}')

    log.info(f'Loading model for feature visualization from {opt.exp3_model_save_file}...')

    for domain in opt.domains:
        F_d[domain].load_state_dict(torch.load(os.path.join(opt.exp3_model_save_file,
                                                            f'net_F_d_{domain}.pth')))
    num_iter = len(train_loaders[opt.domains[0]])
    # visual_features暂时不加上shared feature
    # visual_features, senti_labels = get_visual_features(num_iter, unlabeled_loaders, unlabeled_iters, F_s, F_d)
    visual_features, senti_labels = get_visual_features(num_iter, unlabeled_loaders, unlabeled_iters, F_d)
    return best_acc, visual_features, senti_labels


def get_visual_features(num_iter, test_loaders, test_iters, F_d):
    visual_features, senti_labels = None, None
    with torch.no_grad():
        for _ in tqdm(range(num_iter)):
            for domain in opt.domains:
                d_inputs, targets = utils.endless_get_next_batch(
                    test_loaders, test_iters, domain)
                private_features = F_d[domain](d_inputs)
                if visual_features is None:
                    visual_features = private_features
                    senti_labels = targets
                else:
                    visual_features = torch.cat([visual_features, private_features], 0)
                    senti_labels = torch.cat([senti_labels, targets], 0)

    return visual_features, senti_labels


def margin_regularization(inputs, targets, F_d, LAMBDA):
    features = F_d(inputs)
    graph_source = torch.sum(targets[:, None, :] * targets[None, :, :], 2)
    distance_source = torch.mean((features[:, None, :] - features[None, :, :]) ** 2, 2)
    margin_loss = torch.mean(graph_source * distance_source + (1-graph_source)*F.relu(LAMBDA - distance_source))
    return margin_loss


# 中心点约束，同域中不同class的数据尽量拉开
# 采用hinge loss，同域之间拉开但是要看得出来这是同一个域的数据
# 注意每次都是对一个batch中的数据进行处理(一开始时这么做)
def center_point_constraint(F_d_features, targets):

    negative_features = None
    positive_features = None
    for i in range(targets.shape[0]):
        F_d_features_i = torch.unsqueeze(F_d_features[i], 0)
        if targets[i].item() == 0:
            if negative_features is None:
                negative_features = F_d_features_i
            else:
                negative_features = torch.cat([negative_features, F_d_features_i], 0)
        else:
            if positive_features is None:
                positive_features = F_d_features_i
            else:
                positive_features = torch.cat([positive_features, F_d_features_i], 0)

    print(F_d_features.shape)
    print(negative_features.shape)
    print(positive_features.shape)

    negative_features_center = torch.sum(negative_features, 0) / negative_features.shape[0]
    positive_features_center = torch.sum(positive_features, 0) / positive_features.shape[0]

    # 求和后取平均版本
    center_loss = torch.mean(torch.sum(torch.pow(negative_features_center - positive_features_center, 2)))
    # 求和后不取平均版本
    # center_loss = torch.sum(torch.pow(negative_features_center - positive_features_center, 2))
    return center_loss


def evaluate(name, loader, F_s, F_d, C):
    F_s.eval()
    if F_d:
        F_d.eval()
    C.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(opt.num_labels)
    for inputs, targets in tqdm(it):
        # print(type(inputs))
        targets = targets.to(opt.device)
        if not F_d:
            # unlabeled domain
            d_features = torch.zeros(len(targets), opt.domain_hidden_size).to(opt.device)
        else:
            d_features = F_d(inputs)
        features = torch.cat((F_s(inputs), d_features), dim=1)
        outputs = C(features)
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


def t_sne(visual_features, senti_labels):
    compressed_visual_features = TSNE(random_state=2019).fit_transform(visual_features)
    scatter(data=compressed_visual_features, label=senti_labels,
            dir="./result",
            file_name="exp3_tsne-generated_image.png")
    plt.show()


def main():
    if not os.path.exists(opt.exp3_model_save_file):
        os.makedirs(opt.exp3_model_save_file)
    vocab = Vocab(opt.emb_filename)
    log.info(f'Loading {opt.dataset} Datasets...')
    log.info(f'Domains: {opt.domains}')

    train_sets, dev_sets, test_sets, unlabeled_sets = {}, {}, {}, {}
    for domain in opt.domains:
        train_sets[domain], dev_sets[domain], test_sets[domain], unlabeled_sets[domain] = \
            get_fdu_mtl_datasets(vocab, opt.fdu_mtl_dir, domain, opt.max_seq_len)
    opt.num_labels = FduMtlDataset.num_labels
    log.info(f'Done Loading {opt.dataset} Datasets.')

    cv, visual_features, senti_labels = train(vocab, train_sets, dev_sets, test_sets, unlabeled_sets)
    log.info(f'Training done...')
    acc = sum(cv['valid'].values()) / len(cv['valid'])
    log.info(f'Validation Set Domain Average\t{acc}')
    test_acc = sum(cv['test'].values()) / len(cv['test'])
    log.info(f'Test Set Domain Average\t{test_acc}')

    log.info(f'feature visualization')

    print("Computing t-SNE 2D embedding")
    t0 = time()
    t_sne(visual_features.detach().cpu().numpy(), senti_labels.detach().cpu().numpy())
    print("t-SNE 2D embedding of the digits (time %.2fs)" % (time() - t0))


if __name__ == '__main__':
    main()
