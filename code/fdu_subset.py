import torch.utils.data as data


class Subset(data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, pseudo_labels, pred_value_list):
        self.dataset = dataset
        self.indices = indices
        self.pseudo_labels = pseudo_labels
        self.pred_value_list = pred_value_list

    def __getitem__(self, idx):
        # print(idx)
        sample, _ = self.dataset[self.indices[idx]]
        return sample, self.pseudo_labels[idx], self.pred_value_list[idx]

    def __len__(self):
        return len(self.indices)
