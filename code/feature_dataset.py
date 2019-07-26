import torch.utils.data as data


class DomainSubset(data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        domain_labels (sequence): The domain label corresponding to the entire dataset
    """
    def __init__(self, dataset, domain_labels):
        self.dataset = dataset
        self.domain_labels = domain_labels

    def __getitem__(self, idx):
        return self.dataset[idx], self.domain_labels[idx]

    def __len__(self):
        return len(self.domain_labels)

