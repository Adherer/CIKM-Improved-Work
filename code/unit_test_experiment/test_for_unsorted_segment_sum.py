import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
import torch
import numpy as np

c = tf.constant([[1, 2, 3, 4],
                 [-1, -2, -3, -4],
                 [5, 6, 7, 8]])
result = tf.segment_sum(c, tf.constant([0, 0, 1]))  # 第二个参数长度必须为3
result_ = tf.segment_sum(c, tf.constant([0, 1, 2]))
result1 = tf.unsorted_segment_sum(c, tf.constant([2, 1, 1]), 3)  # 第二个参数长度必须为3
result1_ = tf.unsorted_segment_sum(c, tf.constant([1, 0, 1]), 2)
sess = tf.Session()
print("result")
print(sess.run(result))
print("result_")
print(sess.run(result_))
print("result1")
print(sess.run(result1))
print("result1_")
print(sess.run(result1_))
"""
result
[[0 0 0 0]
 [5 6 7 8]]
result_
[[ 1  2  3  4]
 [-1 -2 -3 -4]
 [ 5  6  7  8]]
result1
[[0 0 0 0]
 [4 4 4 4]
 [1 2 3 4]]
result1_
[[-1 -2 -3 -4]
 [ 6  8 10 12]]
"""

# def unsorted_segment_sum(data, segment_ids, num_segments):
#     """
#     Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.
#
#     :param data: A tensor whose segments are to be summed.
#     :param segment_ids: The segment indices tensor.
#     :param num_segments: The number of segments.
#     :return: A tensor of same data type as the data argument.
#     """
#     assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"
#
#     # segment_ids is a 1-D tensor repeat it to have the same shape as data
#     if len(segment_ids.shape) == 1:
#         s = torch.prod(torch.tensor(data.shape[1:])).long()
#         new_segment_ids = torch.from_numpy(np.repeat(segment_ids.numpy(), s))
#         new_segment_ids = new_segment_ids.view(segment_ids.shape[0], *data.shape[1:])
#         # segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])
#
#     assert data.shape == new_segment_ids.shape, "data.shape and segment_ids.shape should be equal"
#
#     shape = [num_segments] + list(data.shape[1:])
#     tensor = torch.zeros(*shape).scatter_add(0, new_segment_ids, data.float())
#     tensor = tensor.type(data.dtype)
#     return tensor
#
#
# def torch_compute(negative_features, positive_features):
#     negative_samples_label = torch.zeros(negative_features.shape[0]).type(torch.LongTensor)
#     print(negative_samples_label)
#     # positive_samples_label = torch.zeros(positive_features.shape[0]).type(torch.LongTensor)
#     positive_samples_label = torch.ones(positive_features.shape[0]).type(torch.LongTensor)
#     print(positive_samples_label)
#
#     current_negative_result = torch.ones(negative_features.shape[0])
#     current_positive_result = torch.ones(positive_features.shape[0])
#
#     current_negative_count = unsorted_segment_sum(current_negative_result, negative_samples_label, 2)
#     current_positive_count = unsorted_segment_sum(current_positive_result, positive_samples_label, 2)
#     print("==============================")
#     print(current_negative_count)
#     print(current_positive_count)
#
#     current_positive_negative_count = torch.max(current_negative_count, torch.ones_like(current_negative_count))
#     current_positive_positive_count = torch.max(current_positive_count, torch.ones_like(current_positive_count))
#
#     print("------------------------------")
#     print(current_positive_negative_count)
#     print(current_positive_positive_count)
#     current_negative_centroid = torch.div(
#         unsorted_segment_sum(data=negative_features, segment_ids=negative_samples_label, num_segments=2),
#         current_positive_negative_count[:, None])
#     current_positive_centroid = torch.div(
#         unsorted_segment_sum(data=positive_features, segment_ids=positive_samples_label, num_segments=2),
#         current_positive_positive_count[:, None])
#
#     print("*****************************")
#     print(current_negative_centroid)
#     print(current_positive_centroid)
#
#     fm_mask = torch.gt(current_negative_count * current_positive_count, 0).type(torch.FloatTensor)
#     print("old mask is ++++++++++++++++++++++++++++")
#     print(fm_mask)
#     fm_mask /= torch.mean(fm_mask + 1e-8)
#
#     print("new mask is ++++++++++++++++++++++++++++")
#     print(fm_mask)
#     center_loss = torch.mean(torch.mean(torch.pow(current_negative_centroid - current_positive_centroid, 2), 1) * fm_mask)
#     return center_loss


def tensorflow_compute(negative_features, positive_features):
    source_result = tf.zeros(negative_features.shape[0], dtype=tf.int64)
    target_result = tf.ones(positive_features.shape[0], dtype=tf.int64)
    # target_result = tf.zeros(positive_features.shape[0], dtype=tf.int64)

    # 这个函数在pytorch中应该找什么替代？
    current_source_count = tf.unsorted_segment_sum(tf.ones_like(source_result, dtype=tf.float32), source_result, 2)
    current_target_count = tf.unsorted_segment_sum(tf.ones_like(target_result, dtype=tf.float32), target_result, 2)

    current_positive_source_count = tf.maximum(current_source_count, tf.ones_like(current_source_count))
    current_positive_target_count = tf.maximum(current_target_count, tf.ones_like(current_target_count))

    current_source_centroid = tf.divide(
        tf.unsorted_segment_sum(data=negative_features, segment_ids=source_result, num_segments=2),
        current_positive_source_count[:, None])
    current_target_centroid = tf.divide(
        tf.unsorted_segment_sum(data=positive_features, segment_ids=target_result, num_segments=2),
        current_positive_target_count[:, None])

    fm_mask = tf.to_float(tf.greater(current_source_count * current_target_count, 0))
    fm_mask /= tf.reduce_mean(fm_mask + 1e-8)

    center_loss = tf.reduce_mean(tf.reduce_mean(tf.square(current_source_centroid - current_target_centroid), 1) * fm_mask)
    return center_loss, current_source_count, current_target_count, current_positive_source_count, current_positive_target_count


# if __name__ == '__main__':
#     batch_size = 64
#     negative_features = torch.rand(batch_size, 120)
#     positive_features = torch.rand(batch_size * 2, 120)
#     negative_features = tf.convert_to_tensor(negative_features.numpy())
#     positive_features = tf.convert_to_tensor(positive_features.numpy())
#     with tf.Session() as sess:
#         tensorflow_center_loss, current_source_count, current_target_count, current_positive_source_count, current_positive_target_count = sess.run(tensorflow_compute(negative_features, positive_features))
#
#     print("tensorflow_center_loss_" + str(tensorflow_center_loss))
#     print(current_source_count)
#     print(current_target_count)
#     print(current_positive_source_count)
#     print(current_positive_target_count)
#     print(current_positive_source_count[0:, None])
#     print(current_positive_target_count[0:, None])
