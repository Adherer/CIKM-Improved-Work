# result_bs_8 = {(0.05, 0.01): 0.8184775633651158, (0.05, 0.005): 1.6346634219262315,
#                (0.025, 0.025): 2.458299784505124, (0.025, 0.01): 3.2733858303163177,
#                (0.025, 0.005): 4.093573688146581, (0.01,0.025): 4.915979978460787,
#                (0.01, 0.01): 5.738169201505061, (0.01, 0.005): 6.561996425392508}
#
# result_bs_64 = {(0.05, 0.025): 0.8255104070268675, (0.05, 0.01): 1.6543130102819155,
#                 (0.05, 0.005): 2.4816632003252947, (0.025, 0.025): 3.3038936615881536,
#                 (0.025, 0.01): 4.128672446067421, (0.025, 0.005): 4.957671977748744,
#                 (0.01, 0.025): 5.788353753796929, (0.01, 0.01): 6.614425393197451, (0.01, 0.005): 7.437537401525915}
#
# result_bs_128 = {(0.05, 0.01): 0.8295201856889731, (0.05, 0.005): 1.6601419342240191,
#                  (0.025, 0.025): 2.492719813744914, (0.025, 0.01): 3.323947091438084,
#                  (0.025, 0.005): 4.143782538195367, (0.01, 0.025): 4.9739182231101635,
#                  (0.01, 0.01): 5.801799154399122, (0.01, 0.005): 6.6307871820410025}
#
# test1, test2 = 0, 0
# for key in result_bs_8.keys():
#     test2 = result_bs_8[key]
#     result_bs_8[key] = test2 - test1
#     test1 = test2
#
# test1, test2 = 0, 0
# for key in result_bs_64.keys():
#     test2 = result_bs_64[key]
#     result_bs_64[key] = test2 - test1
#     test1 = test2
#
# test1, test2 = 0, 0
# for key in result_bs_128.keys():
#     test2 = result_bs_128[key]
#     result_bs_128[key] = test2 - test1
#     test1 = test2
#
# print(result_bs_8)
# print(result_bs_64)
# print(result_bs_128)
#
# print(max(result_bs_8, key=result_bs_8.get))
# print(max(result_bs_64, key=result_bs_64.get))
# print(max(result_bs_128, key=result_bs_128.get))

# import torch
# import numpy as np
# a = torch.from_numpy(np.arange(0, 20).reshape(10, 2))
# print(a[1].shape)
# b = torch.cat([torch.unsqueeze(a[0], 0), torch.unsqueeze(a[1], 0)], 0)
# print(b)
# print(b.shape)
# test = torch.unsqueeze(a[0], 0)
# print(test)
# print(test[0].item())
# print(test.shape)
# print(a[0].shape)
# print(a[0][8].item())
# b = torch.cat([a, torch.unsqueeze(a[8], 0)], 0)
# print(a)
# print(b)
# a = torch.from_numpy(np.arange(0, 20).reshape(10, 2))
# print(a[:, None, :])
# b = torch.sum(a[:, None, :] * a[None, :, :], 2)
# print(a)
# print(b)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
import numpy as np
import torch

y = np.array([1, 3, 4, 8 ,9])
test = tf.square(y)
y = torch.from_numpy(y)
test2 = torch.pow(y, 2)
with tf.Session() as sess:
    test = sess.run(test)
    print(test)
# target_result = tf.argmax(target_pred, 1)

print(test2)