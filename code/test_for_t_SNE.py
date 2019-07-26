import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
from time import time

digits = load_digits()

# 将坐标缩放到[0,1]区间
def plot_embedding(data):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    return data


def t_sne(n_components):
    if n_components == 2:
        tsne_digits = TSNE(n_components=n_components, random_state=35).fit_transform(digits.data)
        aim_data = plot_embedding(tsne_digits)
        print(aim_data.shape)
        plt.figure()
        plt.subplot(111)
        plt.scatter(aim_data[:, 0], aim_data[:, 1], c=digits.target)
        plt.title("T-SNE Digits")
        plt.savefig("T-SNE_Digits.png")
    elif n_components == 3:
        tsne_digits = TSNE(n_components=n_components, random_state=35).fit_transform(digits.data)
        aim_data = plot_embedding(tsne_digits)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(aim_data[:, 0], aim_data[:, 1], aim_data[:, 2], c=digits.target)
        plt.title("T-SNE Digits")
        plt.savefig("T-SNE_Digits_3d.png")
    else:
        print("The value of n_components can only be 2 or 3")

    plt.show()


def main():
    print("Computing t-SNE embedding")
    t0 = time()
    t_sne(2)
    print("t-SNE embedding of the digits (time %.2fs)" % (time() - t0))


if __name__ == '__main__':
    main()