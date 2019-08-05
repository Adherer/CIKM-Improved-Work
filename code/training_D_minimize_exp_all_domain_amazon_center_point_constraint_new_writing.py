import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from collections import defaultdict
import itertools
import logging
import pickle
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
from mpl_toolkits.mplot3d import Axes3D
from time import time
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import torch.nn.functional as F
import math
import pandas as pd
import tensorflow as tf

# ---------------------- 2019.07.26更新 ---------------------- #
# 1.加上网格搜索，注意看哪些参数是需要网格搜索完成的，另外注意控制变量法来搜索参数
# 2.加上tensorboardX可视化loss变化情况，主要看private feature和shared feature经过判别器之后的loss变化情况
#   以及Classifier的loss
# ---------------------- 2019.07.26更新 ---------------------- #


# ---------------------- 最小化实验的简单思路 ---------------------- #
# 由于师兄设计的模型比较复杂，可能需要再进一步思考收敛策略，因此将实验最小化，争取先调出D
# 调出D的意思就是，争取让各个F_d都分开，并且F_s与各个F_d之间均有重叠，可以利用t-SNE进行可视化操作
# 注意D网络架构的设计，可以继续探索，加梯度惩罚或者信息瓶颈等都试试看
# 以下是训练的一些细节：
# 首先要加载所有的数据，初步选定的训练方式是：
# 1.每个domain对应于一个dataloader，每个E_p喂入batch_size的数据，并将(len(domains) * batch_size)送入E_s
# 2.需要注意的是，训练方式跟MAN中的一致，只不过在两个domain loss处，要加上F_d的部分


# ---------------------- some settings ---------------------- #
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
def train(train_sets, dev_sets, test_sets, unlabeled_sets):
    """
    train_sets, dev_sets, test_sets: dict[domain] -> AmazonDataset
    For unlabeled domains, no train_sets are available
    """

    # ---------------------- dataloader ---------------------- #
    # dataset loaders
    train_loaders, unlabeled_loaders = {}, {}
    train_iters, unlabeled_iters = {}, {}
    dev_loaders, test_loaders = {}, {}
    # 加载有label的训练数据
    for domain in opt.domains:
        train_loaders[domain] = DataLoader(train_sets[domain],
                                           opt.batch_size, shuffle=True)
        train_iters[domain] = iter(train_loaders[domain])

    for domain in opt.dev_domains:
        dev_loaders[domain] = DataLoader(dev_sets[domain],
                                         opt.batch_size, shuffle=False)
        test_loaders[domain] = DataLoader(test_sets[domain],
                                          opt.batch_size, shuffle=False)

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
                opt.batch_size, shuffle=True)
        unlabeled_iters[domain] = iter(unlabeled_loaders[domain])

    # ---------------------- model initialization ---------------------- #
    F_s = None
    F_d = {}
    C = None
    if opt.model.lower() == 'mlp':
        F_s = MlpFeatureExtractor(opt.feature_num, opt.F_hidden_sizes,
                opt.shared_hidden_size, opt.dropout, opt.F_bn)
        for domain in opt.domains:
            F_d[domain] = MlpFeatureExtractor(opt.feature_num, opt.F_hidden_sizes,
                opt.domain_hidden_size, opt.dropout, opt.F_bn)
    else:
        raise Exception(f'Unknown model architecture {opt.model}')

    C = SentimentClassifier(opt.C_layers, opt.shared_hidden_size + opt.domain_hidden_size,
                            opt.shared_hidden_size + opt.domain_hidden_size, opt.num_labels,
                            opt.dropout, opt.C_bn)
    D = DomainClassifier(opt.D_layers, opt.shared_hidden_size, opt.shared_hidden_size,
                         len(opt.all_domains), opt.loss, opt.dropout, opt.D_bn)

    # 转移到gpu上
    F_s, C, D = F_s.to(opt.device), C.to(opt.device), D.to(opt.device)
    for f_d in F_d.values():
        f_d = f_d.to(opt.device)

    optimizer = optim.Adam(itertools.chain(*map(list, [F_s.parameters() if F_s else [], C.parameters()] + [f.parameters() for f in F_d.values()])),
                           lr=opt.learning_rate)
    optimizerD = optim.Adam(D.parameters(), lr=opt.D_learning_rate)

    # training
    best_acc, best_avg_acc = defaultdict(float), 0.0
    # writer = SummaryWriter()
    for epoch in range(opt.max_epoch):
        F_s.train()
        C.train()
        D.train()
        for f in F_d.values():
            f.train()

        # training accuracy
        correct, total = defaultdict(int), defaultdict(int)
        # D accuracy
        shared_d_correct, private_d_correct, d_total = 0, 0, 0
        # conceptually view 1 epoch as 1 epoch of the first domain
        num_iter = len(train_loaders[opt.domains[0]])
        for i in tqdm(range(num_iter)):
            LAMBDA = 3
            lambda1 = 0.05
            lambda2 = 0.05
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
                if opt.n_critic > 0 and ((epoch == 0 and i < 25) or i % 500 == 0):
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
                    d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs))
                    shared_feat = F_s(d_inputs)
                    shared_d_outputs = D(shared_feat)
                    _, shared_pred = torch.max(shared_d_outputs, 1)
                    if domain != opt.dev_domains[0]:
                        private_feat = F_d[domain](d_inputs)
                        private_d_outputs = D(private_feat)
                        _, private_pred = torch.max(private_d_outputs, 1)

                    d_total += len(d_inputs)
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
            utils.unfreeze_net(F_s)
            map(utils.unfreeze_net, F_d.values())
            utils.unfreeze_net(C)
            utils.freeze_net(D)
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
                    d_targets = utils.get_domain_label(opt.loss, domain, len(d_inputs))
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
                    d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs))
                    shared_l_d = functional.kl_div(shared_d_outputs, d_targets, size_average=False)
                    private_l_d, l_d_sum = 0.0, 0.0
                    if domain != opt.dev_domains[0]:
                        private_l_d = functional.kl_div(private_d_outputs, d_targets, size_average=False) * -1. / len(
                            opt.domains)
                    if opt.shared_lambd > 0:
                        l_d_sum = shared_l_d * opt.shared_lambd
                    else:
                        l_d_sum = shared_l_d
                    if opt.private_lambd > 0:
                        l_d_sum += private_l_d * opt.private_lambd
                    else:
                        l_d_sum += private_l_d
                elif opt.loss.lower() == 'l2':
                    d_targets = utils.get_random_domain_label(opt.loss, len(d_inputs))
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
            # ---------------------- update F_s with D gradients on all domains ---------------------- #

            optimizer.step()
            # ********************** F&C iteration ********************** #
        # end of epoch
        # writer.add_scalar('train/classifier-loss', l_c, epoch)
        # writer.add_scalars('train/shared-private-loss', {'shared': shared_l_d, 'private': private_l_d}, epoch)
        log.info('Ending epoch {}'.format(epoch + 1))
        if d_total > 0:
            log.info('shared D Training Accuracy: {}%'.format(100.0 * shared_d_correct / d_total))
            log.info('private D Training Accuracy(average): {}%'.format(100.0 * private_d_correct /
                                                                        len(opt.domains) / d_total))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.domains))
        log.info('\t'.join([str(100.0 * correct[d] / total[d]) for d in opt.domains]))

        # 训练过程中的验证集
        log.info('Evaluating validation sets:')
        acc = {}
        for domain in opt.dev_domains:
            acc[domain] = evaluate(domain, dev_loaders[domain],
                                   F_s, F_d[domain] if domain in F_d else None, C)
        avg_acc = sum([acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average validation accuracy: {avg_acc}')

        # 训练过程中的测试集
        log.info('Evaluating test sets:')
        test_acc = {}
        for domain in opt.dev_domains:
            test_acc[domain] = evaluate(domain, test_loaders[domain],
                                        F_s, F_d[domain] if domain in F_d else None, C)
        avg_test_acc = sum([test_acc[d] for d in opt.dev_domains]) / len(opt.dev_domains)
        log.info(f'Average test accuracy: {avg_test_acc}')

        if avg_acc > best_avg_acc:
            log.info(f'New best average validation accuracy: {avg_acc}')
            best_acc['valid'] = acc
            best_acc['test'] = test_acc
            best_avg_acc = avg_acc
            with open(os.path.join(opt.exp2_model_save_file, 'options.pkl'), 'wb') as ouf:
                pickle.dump(opt, ouf)
            torch.save(F_s.state_dict(),
                       '{}/netF_s.pth'.format(opt.exp2_model_save_file))
            for d in opt.domains:
                if d in F_d:
                    torch.save(F_d[d].state_dict(),
                               '{}/net_F_d_{}.pth'.format(opt.exp2_model_save_file, d))
            torch.save(C.state_dict(),
                       '{}/netC.pth'.format(opt.exp2_model_save_file))
            torch.save(D.state_dict(),
                       '{}/netD.pth'.format(opt.exp2_model_save_file))

    # end of training
    log.info(f'Best average validation accuracy: {best_avg_acc}')

    log.info(f'Loading model for feature visualization from {opt.exp2_model_save_file}...')
    F_s.load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                                f'netF_s.pth')))
    for domain in opt.domains:
        F_d[domain].load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                                            f'net_F_d_{domain}.pth')))
    num_iter = len(train_loaders[opt.domains[0]])
    # visual_features暂时不加上shared feature
    # visual_features, senti_labels = get_visual_features(num_iter, unlabeled_loaders, unlabeled_iters, F_s, F_d)
    visual_features, senti_labels = get_visual_features(num_iter, unlabeled_loaders, unlabeled_iters, F_d)
    return best_acc, visual_features, senti_labels


def get_visual_features(num_iter, unlabeled_loaders, unlabeled_iters, F_d):
    negative_visual_features = None
    positive_visual_features = None
    negative_senti_labels = None
    positive_senti_labels = None
    for _ in tqdm(range(num_iter)):
        for domain in opt.domains:
            d_inputs, targets = utils.endless_get_next_batch(
                unlabeled_loaders, unlabeled_iters, domain)
            private_features = F_d[domain](d_inputs)
            for i in range(targets.shape[0]):
                targets_i = torch.unsqueeze(targets[i], 0)
                private_features_i = torch.unsqueeze(private_features[i], 0)
                if targets[i].item() == 0:
                    if negative_visual_features is None:
                        negative_visual_features = private_features_i
                        negative_senti_labels = targets_i
                    else:
                        negative_visual_features = torch.cat([negative_visual_features, private_features_i], 0)
                        negative_senti_labels = torch.cat([negative_senti_labels, targets_i], 0)
                else:
                    if positive_visual_features is None:
                        positive_visual_features = private_features_i
                        positive_senti_labels = targets_i
                    else:
                        positive_visual_features = torch.cat([positive_visual_features, private_features_i], 0)
                        positive_senti_labels = torch.cat([positive_senti_labels, targets_i], 0)

    visual_features = torch.cat([negative_visual_features, positive_visual_features], 0)
    senti_labels = torch.cat([negative_senti_labels, positive_senti_labels], 0)
    return visual_features, senti_labels


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

# 这个正则明天再来检查一下正确性(与公式对比)，然后再画出T-sne的图来
def margin_regularization(inputs, targets, F_d, LAMBDA):
    features = F_d(inputs)
    graph_source = torch.sum(targets[:, None, :] * targets[None, :, :], 2)
    distance_source = torch.mean((features[:, None, :] - features[None, :, :]) ** 2, 2)
    margin_loss = torch.mean(graph_source * distance_source + (1-graph_source)*F.relu(LAMBDA - distance_source))
    return margin_loss


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        new_segment_ids = torch.from_numpy(np.repeat(segment_ids.numpy(), s))
        new_segment_ids = new_segment_ids.view(segment_ids.shape[0], *data.shape[1:])
        # segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == new_segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add(0, new_segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor


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

    negative_samples_label = torch.zeros(negative_features.shape[0]).type(torch.LongTensor)
    positive_samples_label = torch.ones(positive_features.shape[0]).type(torch.LongTensor)

    current_negative_samples_ones = torch.ones(negative_features.shape[0])
    current_positive_samples_ones = torch.ones(positive_features.shape[0])

    current_negative_samples_count = unsorted_segment_sum(current_negative_samples_ones, negative_samples_label, 2)
    current_positive_samples_count = unsorted_segment_sum(current_positive_samples_ones, positive_samples_label, 2)

    current_positive_negative_samples_count = torch.max(current_negative_samples_count,
                                                        torch.ones_like(current_negative_samples_count))
    current_positive_positive_samples_count = torch.max(current_positive_samples_count,
                                                        torch.ones_like(current_positive_samples_count))

    current_negative_samples_centroid = torch.div(
        unsorted_segment_sum(data=negative_features.cpu(), segment_ids=negative_samples_label, num_segments=2),
        current_positive_negative_samples_count[:, None]).to(opt.device)
    current_positive_samples_centroid = torch.div(
        unsorted_segment_sum(data=positive_features.cpu(), segment_ids=positive_samples_label, num_segments=2),
        current_positive_positive_samples_count[:, None]).to(opt.device)

    print("*****************************")
    print(current_negative_samples_centroid)
    print(current_positive_samples_centroid)

    center_loss = torch.mean(torch.sqrt(torch.pow(current_negative_samples_centroid - current_positive_samples_centroid, 2)))
    return center_loss

# 这个约束项先不加，它的目的是尽量缩小源域和目标域中，同一类的sample的差异
# 但是没有target domain的特征提取器，因此先缩小源域
# 而源域有多个，若双重for循环一一暴力匹配，则为平方复杂度
# 因此这个约束项暂时不加
def cluster_alignment_regularization():
    pass


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
            file_name=domain + "digits_tsne-generated_new_writing_2.png")
    plt.show()


def adaptation_factor(x):
    den = 1.0 + math.exp(-10 * x)
    lamb = 2.0 / den - 1.0
    return min(lamb, 1.0)


def main():
    unlabeled_domains = ['books', 'dvd', 'electronics', 'kitchen']
    test_acc_dict = {}
    i = 1
    opt.shared_lambd = 0.025
    opt.private_lambd = 0.025

    ave_acc = 0.0
    for domain in unlabeled_domains:
        opt.domains = ['books', 'dvd', 'electronics', 'kitchen']
        opt.num_labels = 2
        opt.unlabeled_domains = domain.split()
        opt.dev_domains = domain.split()
        opt.domains.remove(domain)
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
            unlabeled_sets[domain] = raw_unlabeled_sets[domain]

        # in this setting, dev_domains should only contain unlabeled domains
        for domain in opt.dev_domains:
            dev_sets[domain] = datasets[domain]
            test_sets[domain] = raw_unlabeled_sets[domain]
            unlabeled_sets[domain] = datasets[domain]

        cv, visual_features, senti_labels = train(train_sets, dev_sets, test_sets, unlabeled_sets)
        print(visual_features.shape)
        print(senti_labels.shape)
        log.info(f'Training done...')
        acc = sum(cv['valid'].values()) / len(cv['valid'])
        log.info(f'Validation Set Domain Average\t{acc}')
        test_acc = sum(cv['test'].values()) / len(cv['test'])
        log.info(f'Test Set Domain Average\t{test_acc}')
        test_acc_dict[domain] = test_acc
        i += 1

        # ---------------------- 可视化 ---------------------- #
        log.info(f'feature visualization')

        print("Computing t-SNE 2D embedding")
        t0 = time()
        t_sne(domain, visual_features.detach().cpu().numpy(), senti_labels.detach().cpu().numpy())
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

