import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
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
            lamda = 0.05
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
                loss_part_2 = lamda * margin_regularization(inputs, targets, F_d[domain])
                print("loss_part_1: " + str(loss_part_1))
                print("loss_part_2: " + str(loss_part_2))
                l_c = loss_part_1 + loss_part_2
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
    visual_features, senti_labels = get_visual_features(num_iter, unlabeled_loaders, unlabeled_iters, F_s, F_d)
    return best_acc, visual_features, senti_labels


def get_visual_features(num_iter, unlabeled_loaders, unlabeled_iters, F_s, F_d):
    negative_visual_features = None
    positive_visual_features = None
    negative_senti_labels = None
    positive_senti_labels = None
    for _ in tqdm(range(num_iter)):
        for domain in opt.domains:
            d_inputs, targets = utils.endless_get_next_batch(
                unlabeled_loaders, unlabeled_iters, domain)
            if domain != opt.dev_domains[0]:
                private_features = F_d[domain](d_inputs)
            shared_features = F_s(d_inputs)
            for i in range(targets.shape[0]):
                shared_features_i = torch.unsqueeze(shared_features[i], 0)
                targets_i = torch.unsqueeze(targets[i], 0)
                if targets[i].item() == 0:
                    if negative_visual_features is None:
                        if domain != opt.dev_domains[0]:
                            private_features_i = torch.unsqueeze(private_features[i], 0)
                            negative_visual_features = torch.cat([private_features_i, shared_features_i], 0)
                            negative_senti_labels = torch.cat([targets_i, targets_i], 0)
                        else:
                            negative_visual_features = shared_features_i
                            negative_senti_labels = targets_i
                    else:
                        if domain != opt.dev_domains[0]:
                            private_features_i = torch.unsqueeze(private_features[i], 0)
                            negative_visual_features = torch.cat([negative_visual_features, private_features_i, shared_features_i], 0)
                            negative_senti_labels = torch.cat([negative_senti_labels, targets_i, targets_i], 0)
                        else:
                            negative_visual_features = torch.cat([negative_visual_features, shared_features_i], 0)
                            negative_senti_labels = torch.cat([negative_senti_labels, targets_i], 0)
                else:
                    if positive_visual_features is None:
                        if domain != opt.dev_domains[0]:
                            private_features_i = torch.unsqueeze(private_features[i], 0)
                            positive_visual_features = torch.cat([private_features_i, shared_features_i], 0)
                            positive_senti_labels = torch.cat([targets_i, targets_i], 0)
                        else:
                            positive_visual_features = shared_features_i
                            positive_senti_labels = targets_i
                    else:
                        if domain != opt.dev_domains[0]:
                            private_features_i = torch.unsqueeze(private_features[i], 0)
                            positive_visual_features = torch.cat([positive_visual_features, private_features_i, shared_features_i], 0)
                            positive_senti_labels = torch.cat([positive_senti_labels, targets_i, targets_i], 0)
                        else:
                            positive_visual_features = torch.cat([positive_visual_features, shared_features_i], 0)
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
def margin_regularization(inputs, targets, F_d):
    # print(inputs[0].shape)      # torch.Size([8, 42])
    # print(targets.shape)        # torch.Size([8, 42])
    margin = 30
    samples_i = []
    samples_i_label = []
    features = F_d(inputs)
    # print(features.shape)       # torch.Size([8, 64])
    targets = targets.cpu()
    for i in range(opt.batch_size):
        samples_i.append(torch.stack([features[i] for _ in range(opt.batch_size)], 0))
        samples_i_label.append(torch.stack([targets[i] for _ in range(opt.batch_size)], 0))

    samples_i = torch.stack(samples_i, 0)
    samples_i_label = torch.stack(samples_i_label, 0)
    # print(samples_i.shape)      # torch.Size([8, 8, 42])
    # print(samples_i_label.shape)    # torch.Size([8, 8])

    samples_j = torch.stack([features for _ in range(opt.batch_size)], 0)  # 大x_j
    samples_j_label = torch.stack([targets for _ in range(opt.batch_size)], 0) # 大x_j_label
    # print(samples_j.shape)
    # print(samples_j_label.shape)
    mask_matrix = (samples_i_label == samples_j_label).numpy()
    sub_matrix = (samples_i - samples_j).detach().cpu().numpy()
    norm2_matrix = np.linalg.norm(sub_matrix, axis=2, ord=2)
    # print(norm2_matrix)
    result_matrix = mask_matrix * norm2_matrix + (1 - mask_matrix) * np.maximum(0, margin - norm2_matrix)
    return np.sum(result_matrix) / (opt.batch_size ** 2)

# 这个约束项先不加，它的目的是尽量缩小源域和目标域中，同一类的sample的差异
# 但是没有target domain的特征提取器，因此先缩小源域
# 而源域有多个，若双重for循环一一暴力匹配，则为平方复杂度
# 因此这个约束项暂时不加
def cluster_alignment_regularization():
    pass


# def plot_embedding(data):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)
#     return data
#
#
# def t_sne(domain, n_components, visual_features, domain_labels):
#     if n_components == 2:
#         tsne_features = TSNE(n_components=n_components, random_state=35).fit_transform(visual_features)
#         aim_data = plot_embedding(tsne_features)
#         print(aim_data.shape)
#         plt.figure()
#         plt.subplot(111)
#         plt.scatter(aim_data[:, 0], aim_data[:, 1], c=domain_labels)
#         plt.title("T-SNE Digits")
#         plt.savefig(domain + "_" + "T-SNE_Digits.png")
#     elif n_components == 3:
#         tsne_features = TSNE(n_components=n_components, random_state=35).fit_transform(visual_features)
#         aim_data = plot_embedding(tsne_features)
#         fig = plt.figure()
#         ax = Axes3D(fig)
#         ax.scatter(aim_data[:, 0], aim_data[:, 1], aim_data[:, 2], c=domain_labels)
#         plt.title("T-SNE Digits")
#         plt.savefig(domain + "_" + "T-SNE_Digits_3d.png")
#     else:
#         print("The value of n_components can only be 2 or 3")
#
#     plt.show()

def scatter(x, colors, species_number):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", species_number))

    # We create a scatter plot.
    f = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    return f, ax, sc


def t_sne(domain, visual_features, senti_labels, species_number):
    compressed_visual_features = TSNE(random_state=2019).fit_transform(visual_features)
    scatter(compressed_visual_features, senti_labels, species_number)
    plt.savefig(domain + "digits_tsne-generated.png", dpi=120)
    plt.show()


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
        t_sne(domain, visual_features.detach().cpu().numpy(), senti_labels.detach().cpu().numpy(), len(opt.domains) + 1)
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

