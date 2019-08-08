import torch
import torch.nn.functional as F
import numpy as np
import random

# 这里有待验证一下
def margin_regularization_1(targets, features, LAMBDA):
    graph_source = torch.sum(targets[:, None, :] * targets[None, :, :], 2)
    print(targets[:, None, :])
    print(targets[None, :, :])
    print(targets[:, None, :] * targets[None, :, :])
    print(graph_source)
    distance_source = torch.mean((features[:, None, :] - features[None, :, :]) ** 2, 2)
    print(distance_source)
    print(graph_source * distance_source)
    margin_loss = torch.mean(graph_source * distance_source + (1-graph_source)*F.relu(LAMBDA - distance_source))
    return margin_loss


def margin_regularization_2(targets, features):
    # print(inputs[0].shape)      # torch.Size([8, 42])
    # print(targets.shape)        # torch.Size([8, 42])
    margin = 3
    samples_i = []
    samples_i_label = []
    # print(features.shape)       # torch.Size([8, 64])
    targets = targets.cpu()
    for i in range(4):
        samples_i.append(torch.stack([features[i] for _ in range(4)], 0))
        samples_i_label.append(torch.stack([targets[i] for _ in range(4)], 0))

    samples_i = torch.stack(samples_i, 0)
    samples_i_label = torch.stack(samples_i_label, 0)
    # print(samples_i.shape)      # torch.Size([8, 8, 42])
    # print(samples_i_label.shape)    # torch.Size([8, 8])

    samples_j = torch.stack([features for _ in range(4)], 0)  # 大x_j
    samples_j_label = torch.stack([targets for _ in range(4)], 0) # 大x_j_label
    # print(samples_j.shape)
    # print(samples_j_label.shape)
    mask_matrix = (samples_i_label == samples_j_label).numpy()
    sub_matrix = (samples_i - samples_j).detach().cpu().numpy()
    norm2_matrix = np.mean(sub_matrix ** 2, 2)
    # norm2_matrix = np.linalg.norm(sub_matrix, axis=2, ord=2)    # 注意这里往往都不开方
    # print(norm2_matrix)
    result_matrix = mask_matrix * norm2_matrix + (1 - mask_matrix) * np.maximum(0, margin - norm2_matrix)
    return np.sum(result_matrix) / (4 ** 2)


if __name__ == '__main__':
    features = torch.Tensor([[0., 0., 1., 1.],
                             [0., 0., 1., 0.],
                             [1., 0., 1., 1.],
                             [0., 1., 1., 1.]])

    targets = torch.Tensor([0, 1, 0, 1]).unsqueeze(-1).type(torch.LongTensor)
    targets_onehot = torch.FloatTensor(4, 2)
    targets_onehot.zero_()
    targets_onehot.scatter_(1, targets, 1)
    print(margin_regularization_1(targets_onehot, features, 3))
    targets = targets.squeeze(-1)
    print(margin_regularization_2(targets, features))
