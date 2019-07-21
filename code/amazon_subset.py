import torch.utils.data as data


class Subset(data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        pseudo_labels (sequence): The label corresponding to the entire dataset
    """
    def __init__(self, dataset, pseudo_labels):
        self.dataset = dataset
        self.pseudo_labels = pseudo_labels

    def __getitem__(self, idx):
        # print(idx)
        sample, _ = self.dataset[idx]
        return sample, self.pseudo_labels[idx]

    def __len__(self):
        return len(self.pseudo_labels)
