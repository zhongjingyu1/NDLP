import torch
from torch.utils.data import Dataset
from imagelist import *

class CustomDataset(Dataset):
    """Dataset.
    """

    def __init__(self, dataset, split):
        assert dataset in ["coco-lt", "voc-lt", "voc", "nus-wide"]
        if dataset == 'coco-lt':
            self.data_source = ImageList(root='../data/coco/',
                                        list_file='../appendix/coco/coco_lt_%s.txt' % split,
                                        label_file='../appendix/coco/coco_labels.txt',
                                        nb_classes=80,
                                        split=split)
        elif dataset == 'voc-lt':
            self.data_source = ImageList(root='../data/voc/',
                                        list_file='../appendix/VOCdevkit/voc_lt_%s.txt' % split,
                                        label_file='../appendix/VOCdevkit/voc_labels.txt',
                                        nb_classes=20,
                                        split=split)

        self.targets = self.data_source.labels # one-hot label
        self.categories = self.data_source.categories
        self.fns = self.data_source.fns
        self.ind1 = self.data_source.ind1
    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        return img, target


class CustomDataset_partial(Dataset):
    """Dataset.
    """

    def __init__(self, dataset, split, given_label_matrix):
        assert dataset in ["coco-lt", "voc-lt", "voc", "nus-wide"]
        if dataset == 'coco-lt':
            self.data_source = ImageList(root='../data/coco/',
                                         list_file='../appendix/coco/coco_lt_%s.txt' % split,
                                         label_file='../appendix/coco/coco_labels.txt',
                                         nb_classes=80,
                                         split=split)
        elif dataset == 'voc-lt':
            self.data_source = ImageList(root='../data/voc/',
                                         list_file='../appendix/VOCdevkit/voc_lt_%s.txt' % split,
                                         label_file='../appendix/VOCdevkit/voc_labels.txt',
                                         nb_classes=20,
                                         split=split)

        self.targets = self.data_source.labels  # one-hot label
        self.categories = self.data_source.categories
        self.fns = self.data_source.fns
        self.given_label_matrix = given_label_matrix

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, target = self.data_source.get_sample(idx)
        partial_label = self.given_label_matrix[idx]
        return img, target, partial_label, idx

def build_dataset1(dataset, split):
    assert split in ['train', 'test', 'val']

    assert dataset in ["coco-lt", "voc-lt"]
    if split == 'train':
        dataset = CustomDataset(
        dataset=dataset,
        split=split
        )
    elif split == 'test':
        dataset = CustomDataset(
        dataset=dataset,
        split=split
        )

    return dataset

def build_dataset_partial(dataset, split, given_label_matrix=None):
    assert split in ['train', 'test', 'val']

    assert dataset in ["coco-lt", "voc-lt"]
    if split == 'train':
        dataset = CustomDataset_partial(
        dataset=dataset,
        split=split,
        given_label_matrix=given_label_matrix
        )
    elif split == 'test':
        dataset = CustomDataset_partial(
        dataset=dataset,
        split=split,
        given_label_matrix=given_label_matrix
        )

    return dataset
