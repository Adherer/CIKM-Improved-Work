import logging
import random
import os
import sys
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter
from man_models import *
from data_prep.msda_preprocessed_amazon_dataset import get_msda_amazon_datasets
import utils
from utils import *
from amazon_subset import Subset

# ---------------------- some settings ---------------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
def train(target_domain_idx, train_sets, test_sets):
    """
    train_sets, test_sets: unlabeled domain(dev domain) -> AmazonDataset
    """
    # test dataset
    print(train_sets[0])
    print(test_sets[0])
    print(train_sets[0][0].shape)
    print(test_sets[0][1].shape)

    # ---------------------- dataloader ---------------------- #
    train_loaders = DataLoader(train_sets, opt.batch_size, shuffle=False)
    test_loaders = DataLoader(test_sets, opt.batch_size, shuffle=False)

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

    # ---------------------- load pre-training model ---------------------- #
    log.info(f'Loading model from {opt.exp2_model_save_file}...')
    F_s.load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                                f'netF_s.pth')))
    for domain in opt.all_domains:
        if domain in F_d:
            F_d[domain].load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                                                f'net_F_d_{domain}.pth')))
    C.load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                              f'netC.pth')))
    D.load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                              f'netD.pth')))

    # ---------------------- get fake label(hard label) ---------------------- #
    log.info('Get fake label:')
    pseudo_labels, _ = genarate_labels(target_domain_idx, False, train_loaders, F_s, F_d, C, D)

    # ********************** Test the accuracy of the label prediction ********************** #
    test_pseudo_labels, targets_total = genarate_labels(target_domain_idx, True, test_loaders, F_s, F_d, C, D)
    label_correct_acc = calc_label_prediction(test_pseudo_labels, targets_total)
    log.info(f'the correct rate of label prediction: {label_correct_acc}')
    # ********************** Test the accuracy of the label prediction ********************** #

    # ------------------------------- 构造target数据集 ------------------------------- #
    target_dataset = Subset(train_sets, pseudo_labels)
    target_dataloader_labelled = DataLoader(target_dataset, opt.batch_size, shuffle=True)

    target_train_iters = iter(target_dataloader_labelled)

    # ------------------------------- F_d_target模型以及一些必要的参数定义 ------------------------------- #
    F_d_target = MlpFeatureExtractor(opt.feature_num, opt.F_hidden_sizes,
                                     opt.domain_hidden_size, opt.dropout, opt.F_bn)

    F_d_target = F_d_target.to(opt.device)

    # ******************************* 加载预训练模型的权重，不用再从头训练 ******************************* #
    f_d_choice = random.randint(0, len(opt.domains))
    F_d_target.load_state_dict(torch.load(os.path.join(opt.exp2_model_save_file,
                                                       f'net_F_d_{opt.domains[f_d_choice]}.pth')))
    optimizer_F_d_target = optim.Adam(F_d_target.parameters(), lr=opt.learning_rate)

    # ------------------------------- 测试一下只利用shared feature的准确率 ------------------------------- #
    log.info('Evaluating test sets only on shared feature:')
    test_acc = evaluate(opt.dev_domains, test_loaders, F_s, None, C)
    log.info(f'test accuracy: {test_acc}')

    for epoch in range(opt.max_epoch):
        F_d_target.train()
        C.train()
        # training accuracy
        correct, total = 0, 0
        num_iter = len(target_dataloader_labelled)
        print(num_iter)
        for _ in tqdm(range(num_iter)):
            utils.freeze_net(F_s)
            C.zero_grad()
            F_d_target.zero_grad()
            inputs, targets = utils.endless_get_next_batch(target_dataloader_labelled, target_train_iters)
            targets = targets.to(opt.device)
            shared_feat = F_s(inputs)
            domain_feat = F_d_target(inputs)
            features = torch.cat((shared_feat, domain_feat), dim=1)
            c_outputs = C(features)
            l_c = functional.nll_loss(c_outputs, targets)
            l_c.backward()
            _, pred_idx = torch.max(c_outputs, 1)
            total += targets.size(0)
            correct += (pred_idx == targets).sum().item()

            optimizer_F_d_target.step()

        # 暂时留下一个问题，这里没有写验证集，因为如果这里要设置验证集的话，需要将训练集切割
        # 因为unlabeled domain里面只有2000个label sample以及raw_unlabeled_sets(四千多张)
        # 后者训练时用，前者测试的时候用
        # end of epoch
        print("correct is %d" % correct)
        print("total is %d" % total)
        log.info('Ending epoch {}'.format(epoch + 1))
        log.info('Training accuracy:')
        log.info(opt.unlabeled_domains)
        log.info(str(100.0 * correct / total))

        # 训练过程中的验证集
        print("correct is %d" % correct)
        print("total is %d" % total)
        log.info('Ending epoch {}'.format(epoch + 1))
        log.info('Training accuracy:')
        log.info('\t'.join(opt.unlabeled_domains))
        log.info(str(100.0 * correct / total))

    # 保存模型
    torch.save(F_d_target.state_dict(),
               '{}/net_F_d_target.pth'.format(opt.exp2_target_model_save_file))
    torch.save(C.state_dict(),
               '{}/netC_target.pth'.format(opt.exp2_target_model_save_file))

    # 在测试集上测试，选择2000张有label的标签数据
    # 方便对比与单独训练有label时的情况
    log.info('Evaluating test sets:')
    test_acc = evaluate(opt.dev_domains, test_loaders, F_s, F_d_target, C)
    log.info(f'test accuracy: {test_acc}')
    return test_acc


# D的损失函数还是熵的形式
def train_D(target_domain_idx, optimizerD, F_d_features, shared_feature, D):
    # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
    # 这是一个底层的问题，值得好好深究一下！
    target_one_hot_label = torch.LongTensor([target_domain_idx]).unsqueeze(1)
    target_one_hot_label = torch.zeros(1, len(opt.all_domains)).scatter_(1, target_one_hot_label, 1)
    # print(target_one_hot_label)
    batch_target_one_hot_label = np.concatenate(([target_one_hot_label.numpy() for _ in range(opt.batch_size)]), 0)
    batch_target_one_hot_label = torch.from_numpy(batch_target_one_hot_label).to(opt.device)
    # print(batch_target_one_hot_label.shape)
    D.zero_grad()
    private_feature_d_outputs = []
    shared_feature_d_outputs = D(shared_feature)
    padding_tensor = torch.zeros((shared_feature.shape[0], shared_feature.shape[1] - F_d_features[0].shape[1]),
                                 requires_grad=True).to(opt.device)
    loss_F_s = torch.sum(batch_target_one_hot_label * shared_feature_d_outputs)
    loss_F_s.requires_grad = True
    loss_F_ds = 0.0
    # print(shared_feature.shape)
    # print(padding_tensor.shape)
    # print(F_d_features[0].shape)
    for f_d in F_d_features:
        f_d = torch.cat([f_d, padding_tensor], 1)
        f_d.requires_grad = True
        private_feature_d_outputs.append(D(f_d))

    # 这里还能优化成矩阵形式
    source_domain_idx = []
    for i in range(4):
        if i is not target_domain_idx:
            source_domain_idx.append(i)

    for f_d_outputs, source_idx in zip(private_feature_d_outputs, source_domain_idx):
        source_one_hot_label = torch.LongTensor([source_idx]).unsqueeze(1)
        source_one_hot_label = torch.zeros(1, len(opt.all_domains)).scatter_(1, source_one_hot_label, 1)
        # print(source_one_hot_label)
        batch_source_one_hot_label = np.concatenate(([source_one_hot_label.numpy() for _ in range(opt.batch_size)]), 0)
        batch_source_one_hot_label = torch.from_numpy(batch_source_one_hot_label).to(opt.device)
        # print(batch_source_one_hot_label.shape)
        loss_F_ds += torch.sum(batch_source_one_hot_label * f_d_outputs)

    loss_sum = (loss_F_s + loss_F_ds) * opt.lambd
    loss_sum.requires_grad = True
    # print(loss_sum)
    loss_sum.backward()
    optimizerD.step()


def genarate_labels(target_domain_idx, test_for_label_acc, dataloader, F_s, F_d, C, D):
    """Genrate pseudo labels for unlabeled domain dataset."""
    # ---------------------- Switch to eval() mode ---------------------- #
    optimizerD = optim.Adam(D.parameters(), lr=opt.D_learning_rate)
    F_s.eval()
    C.eval()
    for f_d in F_d.values():
        f_d.eval()
    D.eval()

    it = iter(dataloader)               # batch_size为128
    print(len(it))

    F_d_features = []
    F_d_outputs = []
    targets_total, pseudo_labels = None, None

    with torch.no_grad():
        for inputs, targets in tqdm(it):

            # 少于一个batch的数据，丢弃
            if inputs.shape[0] < opt.batch_size:
                continue
            # ---------------------- 得到F_d_features和shared_feature ---------------------- #
            for f_d in F_d.values():
                F_d_features.append(f_d(inputs))
            shared_feature = F_s(inputs)

            # ---------------------- 送入D中训练 ---------------------- #
            if test_for_label_acc is False:
                utils.freeze_net(F_s)
                map(utils.freeze_net, F_d.values())
                utils.freeze_net(C)
                utils.unfreeze_net(D)
                train_D(target_domain_idx, optimizerD, F_d_features, shared_feature, D)

            # ---------------------- 处理F_s经过D之后的"分数" ---------------------- #
            # 此时的shared_feature_d_outputs是source domain中domain数量 + 1(target domain)
            shared_feature_d_outputs = D(shared_feature)
            # print(shared_feature_d_outputs.shape)

            # 去掉target domain这一维度
            indices = []
            for i in range(len(opt.all_domains)):
                if opt.all_domains[i] == opt.unlabeled_domains:
                    continue
                else:
                    indices.append(i)

            indices = torch.tensor(indices).to(opt.device)
            # print(indices)
            selected_shared_feature_d_outputs = torch.index_select(shared_feature_d_outputs, 1, indices)
            # print(selected_shared_feature_d_outputs.shape)

            # 将selected_shared_feature_d_outputs经过softmax，进行归一化
            norm_selected_shared_feature_d_outputs = F.softmax(selected_shared_feature_d_outputs, dim=1)

            # ---------------------- 所有F_d经过C之后的所有classifier分数 ---------------------- #
            for f_d_feature in F_d_features:
                padding_shared_feature = torch.zeros((shared_feature.shape[0], shared_feature.shape[1]),
                                                     requires_grad=True).to(opt.device)
                f_d_feature = torch.cat([f_d_feature, padding_shared_feature], 1)
                F_d_outputs.append(C(f_d_feature))

            # ---------------------- 得到c * w后的分数 ---------------------- #
            norm_selected_shared_feature_d_outputs = np.repeat(norm_selected_shared_feature_d_outputs.cpu().numpy(),
                                                               repeats=opt.num_labels, axis=1)
            norm_selected_shared_feature_d_outputs = torch.from_numpy(norm_selected_shared_feature_d_outputs)
            F_d_outputs_tensor = torch.stack(F_d_outputs, 0).reshape(opt.batch_size, opt.num_labels * (len(opt.all_domains) - 1))
            c_mul_w = norm_selected_shared_feature_d_outputs * F_d_outputs_tensor.cpu()

            # ---------------------- 对c_mul_w处理得到hard label ---------------------- #
            even_indices = torch.LongTensor(np.arange(0, 2 * (len(opt.all_domains) - 1), 2))
            odd_indices = torch.LongTensor(np.arange(1, 2 * (len(opt.all_domains) - 1), 2))
            even_index_scores = torch.index_select(c_mul_w, 1, even_indices)
            odd_index_scores = torch.index_select(c_mul_w, 1, odd_indices)
            even_index_scores_sum = torch.sum(even_index_scores, 1).unsqueeze(1)
            odd_index_scores_sum = torch.sum(odd_index_scores, 1).unsqueeze(1)
            pred_scores = torch.cat([even_index_scores_sum, odd_index_scores_sum], 1)
            _, pred_idx = torch.max(pred_scores, 1)

            # ---------------------- 保存pseudo_labels ---------------------- #
            if pseudo_labels is None:
                pseudo_labels = pred_idx
                targets_total = targets
            else:
                pseudo_labels = torch.cat(
                    [pseudo_labels, pred_idx], 0)
                targets_total = torch.cat(
                    [targets_total, targets], 0)

            F_d_features.clear()
            F_d_outputs.clear()

    print(pseudo_labels.shape)
    print(targets_total.shape)
    print(">>> Generate pseudo labels {}, target samples {}".format(
        len(pseudo_labels), targets_total.shape[0]))

    return pseudo_labels, targets_total


def evaluate(name, loader, F_s, F_d_target, C):
    F_s.eval()
    if F_d_target:
        F_d_target.eval()
    C.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(opt.num_labels)
    for inputs, targets in tqdm(it):
        targets = targets.to(opt.device)
        if F_d_target:
            d_features = F_d_target(inputs)
        else:
            d_features = torch.zeros(len(targets), opt.domain_hidden_size).to(opt.device)
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


def calc_label_prediction(test_pseudo_labels, targets_total):
    equal_idx = torch.nonzero(torch.eq(test_pseudo_labels, targets_total.cpu())).squeeze()
    print(equal_idx.shape[0])
    print(test_pseudo_labels.shape[0])
    return equal_idx.shape[0] / test_pseudo_labels.shape[0]


def main():
    if not os.path.exists(opt.exp2_model_save_file):
        os.makedirs(opt.exp2_model_save_file)

    unlabeled_domains = ['books', 'dvd', 'electronics', 'kitchen']
    test_acc_dict = {}
    i = 1
    ave_acc = 0.0
    # unlabeled samples as train_sets, labeled samples as test_sets
    for domain in unlabeled_domains:
        opt.domains = ['books', 'dvd', 'electronics', 'kitchen']
        # unlabeled samples as train_sets, labeled samples as test_sets
        test_sets, train_sets = get_msda_amazon_datasets(
            opt.prep_amazon_file, domain, 1, opt.feature_num)
        opt.num_labels = 2
        opt.unlabeled_domains = domain
        opt.dev_domains = domain
        opt.domains.remove(domain)
        opt.exp2_model_save_file = './save/man_exp2/exp' + str(i)
        test_acc = train(i - 1, train_sets, test_sets)          # i表示target domain的index
        test_acc_dict[domain] = test_acc
        i += 1

    log.info(f'Training done...')
    log.info(f'test_acc\'s result is: ')
    for key in test_acc_dict:
        log.info(str(key) + ": " + str(test_acc_dict[key]))
        ave_acc += test_acc_dict[key]

    log.info(f'ave_test_acc\'s result is: ')
    log.info(ave_acc / 4)


if __name__ == '__main__':
    main()

