import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import tensorflow as tf
import torch
import numpy as np


def tensorflow_compute(negative_features, positive_features):
    source_result = tf.zeros(negative_features.shape[0], dtype=tf.int64)
    target_result = tf.ones(positive_features.shape[0], dtype=tf.int64)

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

    center_loss = tf.reduce_mean(tf.reduce_mean(tf.square(current_source_centroid - current_target_centroid), 1))
    return center_loss


def center_point_constraint(negative_features, positive_features):
    negative_features_center = torch.sum(negative_features, 0) / negative_features.shape[0]
    positive_features_center = torch.sum(positive_features, 0) / positive_features.shape[0]

    # 先用平方loss试试看
    norm_loss = torch.pow(torch.norm(negative_features_center - positive_features_center, p=2), 2)
    return norm_loss


if __name__ == '__main__':
    batch_size = 64
    negative_features = torch.rand(batch_size, 120)
    positive_features = torch.rand(batch_size * 2, 120)
    ordinary_loss = center_point_constraint(negative_features, positive_features)
    negative_features = tf.convert_to_tensor(negative_features.numpy())
    positive_features = tf.convert_to_tensor(positive_features.numpy())
    with tf.Session() as sess:
        tensorflow_center_loss = sess.run(tensorflow_compute(negative_features, positive_features))

    print("tensorflow_center_loss_" + str(tensorflow_center_loss))
    print("ordinary_loss_" + str(ordinary_loss))

