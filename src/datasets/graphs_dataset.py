import os
import pathlib
from typing import List
import numpy as np
import torch
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

def sample_data_with_given_indices(data, indices):
    return [data[i] for i in indices]

class DiverseGraphsDataset(InMemoryDataset):
    def __init__(self, 
                 dataset_name,
                 graphs_file,
                 split, #[train/test/val]
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 ):
        
        
        self.graphs_file = graphs_file
        self.dataset_name = dataset_name
        self.split = split
        
        
        super().__init__(root=root, transform=transform, pre_filter=pre_filter,
                         pre_transform=pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
            return [self.split + '.pt']
    
    def download(self):
        """
        Replaced download interface so that we process oour graphs locally
        """
        
        adjacencies: List[np.ndarray] = np.load(self.graphs_file, 
                                                allow_pickle=True,
                                                ) # type: ignore
        
        adjacencies: List[torch.FloatTensor] = [
            torch.FloatTensor(adj) for adj in adjacencies
        ]
        
        num_of_graphs = len(adjacencies)
        
        generator_cpu = torch.Generator()
        generator_cpu.manual_seed(0)
        
        test_len = int(round(num_of_graphs * 0.1))
        train_len = int(round((num_of_graphs - test_len) * 0.9))
        val_len = num_of_graphs - train_len - test_len
        
        indices = torch.randperm(n=num_of_graphs, generator=generator_cpu)
        print(f"Dataset length are: {train_len=} {val_len=} {test_len=}")
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len+val_len]
        test_indices = indices[train_len + val_len:]
        
        train_data = sample_data_with_given_indices(data=adjacencies, indices=train_indices)
        val_data = sample_data_with_given_indices(data=adjacencies, indices=val_indices)
        test_data = sample_data_with_given_indices(data=adjacencies, indices=test_indices)
        
        # for i, adj in enumerate(adjacencies):
            
        #     if i in train_indices:
        #         train_data.append(adj)
        #     elif i in val_indices:
        #         val_data.append(adj)
        #     elif i in test_indices:
        #         test_data.append(adj)
        #     else:
        #         raise ValueError(f"Index {i} is not in any split")
        
        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])
        
        print(f"Completed preprocessing of graphs")
    
    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float() # NOTE what is y?
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            data_list.append(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class DiverseGraphsDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=100):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        datasets = {'train': DiverseGraphsDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path, graphs_file=self.cfg.dataset.graphs_file,
                                                 
                                                 ),
                    'val': DiverseGraphsDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path, graphs_file=self.cfg.dataset.graphs_file
                                        ),
                    'test': DiverseGraphsDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path, graphs_file=self.cfg.dataset.graphs_file
                                        )}
        
        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]
    

class DiverseDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'diverse_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])               # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
