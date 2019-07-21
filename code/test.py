# import torch
# x = torch.randn(3, 10)
# print(x)
# print(x.shape)
# num_index = []
# for i in range(10):
#     if i is not 8:
#         num_index.append(i)
#
# num_index = torch.tensor(num_index)
# print(num_index)
# x = torch.index_select(x, 1, num_index)
# print(x)

# import torch.nn.functional as F
# import torch
# import numpy as np
# x = torch.from_numpy(np.arange(1, 31).reshape(3, 10)).type(torch.FloatTensor)
# print(x.shape)
# x = F.softmax(x, 1)
# print(x)
# print(x.shape)
# import torch
# import numpy as np
# x_array = np.arange(1, 31).reshape(3, 10)
# print(x_array)
# print(x_array.shape)
# x_array = np.repeat(x_array, repeats=2, axis=1)
# x = torch.from_numpy(x_array)
# print(x)
# print(x.shape)

import torch

# a = torch.Tensor([[1, 2],
#                   [3, 4],
#                   [5, 6]])
# b = torch.Tensor([[1, 2],
#                   [3, 4],
#                   [5, 6]])
# c = torch.Tensor([[10000, 20000],
#                   [30000, 40000],
#                   [50000, 60000]])
# test = torch.Tensor([3, 7, 11])
# a_sum = torch.sum(a, 1)
# b_sum = torch.sum(b, 1)
# print(a_sum)
# print(a_sum.shape)
# print(b_sum)
# print(b_sum.shape)
# print(a_sum == test)
# print(torch.cat([a, b], 1))
# print(a.shape)
# my_list = [a, b, c]
# x = torch.stack(my_list, 0).reshape(3, 6)
# print(x)
# print(x.shape)

label = torch.LongTensor([0]).unsqueeze(1)
print(label)
one_hot = torch.zeros(1, 4).scatter_(1, label, 1)
print(one_hot)