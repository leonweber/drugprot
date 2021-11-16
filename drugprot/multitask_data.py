import random

import numpy as np
from torch.utils.data import Dataset, BatchSampler


class MultiTaskDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        task_id_2_data_set_dic = {}
        for dataset in datasets:
            task_id = dataset.meta.name
            assert task_id not in task_id_2_data_set_dic, (
                "Duplicate task_id %s" % task_id
            )
            task_id_2_data_set_dic[task_id] = dataset

        self._task_id_2_data_set_dic = task_id_2_data_set_dic

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def __getitem__(self, idx):
        task_id, sample_id = idx
        return self._task_id_2_data_set_dic[task_id][sample_id]


class MultiTaskBatchSampler(BatchSampler):
    def __init__(self, datasets, dataset_to_batch_size, mix_opt=0, extra_task_ratio=0, drop_last=False):
        self._datasets = datasets
        self.datset_to_batch_size = dataset_to_batch_size
        self.drop_last = drop_last
        self._mix_opt = mix_opt
        self._extra_task_ratio = extra_task_ratio
        train_data_list = []
        for dataset in datasets:
            train_data_list.append(
                self._get_shuffled_index_batches(len(dataset),
                                                 dataset_to_batch_size[dataset.meta.name])
            )
        self._train_data_list = train_data_list

    @staticmethod
    def _get_shuffled_index_batches(dataset_len, batch_size):
        index_batches = [
            list(range(i, min(i + batch_size, dataset_len)))
            for i in range(0, dataset_len, batch_size)
        ]
        random.shuffle(index_batches)
        return index_batches

    def __len__(self):
        return sum(len(train_data) for train_data in self._train_data_list)

    def __iter__(self):
        all_iters = [iter(item) for item in self._train_data_list]
        all_indices = self._gen_task_indices(
            self._train_data_list, self._mix_opt, self._extra_task_ratio
        )
        for local_task_idx in all_indices:
            task_id = self._datasets[local_task_idx].meta.name
            batch = next(all_iters[local_task_idx])
            yield [(task_id, sample_id) for sample_id in batch]

    @staticmethod
    def _gen_task_indices(train_data_list, mix_opt, extra_task_ratio):
        all_indices = []
        if len(train_data_list) > 1 and extra_task_ratio > 0:
            main_indices = [0] * len(train_data_list[0])
            extra_indices = []
            for i in range(1, len(train_data_list)):
                extra_indices += [i] * len(train_data_list[i])
            random_picks = int(
                min(len(train_data_list[0]) * extra_task_ratio, len(extra_indices))
            )
            extra_indices = np.random.choice(extra_indices, random_picks, replace=False)
            if mix_opt > 0:
                extra_indices = extra_indices.tolist()
                random.shuffle(extra_indices)
                all_indices = extra_indices + main_indices
            else:
                all_indices = main_indices + extra_indices.tolist()

        else:
            for i in range(1, len(train_data_list)):
                all_indices += [i] * len(train_data_list[i])
            if mix_opt > 0:
                random.shuffle(all_indices)
            all_indices += [0] * len(train_data_list[0])
        if mix_opt < 1:
            random.shuffle(all_indices)
        return all_indices