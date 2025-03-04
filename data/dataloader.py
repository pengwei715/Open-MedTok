import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch as GraphBatch


def collate_fn(batch):
    """
    Custom collate function for batching samples with text and graph data.
    
    Args:
        batch: List of samples from the dataset
    
    Returns:
        Batched data with text and graph components
    """
    # Extract components
    codes = [item["code"] for item in batch]
    texts = [item["text"] for item in batch]
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    
    # Create a list of graph Data objects for batching
    graph_list = [
        GraphData(
            x=item["graph_features"],
            edge_index=item["graph_edge_index"]
        )
        for item in batch
    ]
    
    # Batch the graphs
    batched_graphs = GraphBatch.from_data_list(graph_list)
    
    return {
        "codes": codes,
        "texts": texts,
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "graph_features": batched_graphs.x,
        "graph_edge_index": batched_graphs.edge_index,
        "graph_batch": batched_graphs.batch
    }


def collate_fn_pairs(batch):
    """
    Custom collate function for batching pairs of samples.
    
    Args:
        batch: List of sample pairs from the dataset
    
    Returns:
        Batched data with text and graph components for both items in each pair
    """
    # Extract components for the first item in each pair
    codes1 = [item["code1"] for item in batch]
    texts1 = [item["text1"] for item in batch]
    input_ids1 = torch.stack([item["input_ids1"] for item in batch])
    attention_masks1 = torch.stack([item["attention_mask1"] for item in batch])
    
    # Extract components for the second item in each pair
    codes2 = [item["code2"] for item in batch]
    texts2 = [item["text2"] for item in batch]
    input_ids2 = torch.stack([item["input_ids2"] for item in batch])
    attention_masks2 = torch.stack([item["attention_mask2"] for item in batch])
    
    # Extract relation types
    relation_types = [item["relation_type"] for item in batch]
    
    # Create a list of graph Data objects for the first item in each pair
    graph_list1 = [
        GraphData(
            x=item["graph_features1"],
            edge_index=item["graph_edge_index1"]
        )
        for item in batch
    ]
    
    # Create a list of graph Data objects for the second item in each pair
    graph_list2 = [
        GraphData(
            x=item["graph_features2"],
            edge_index=item["graph_edge_index2"]
        )
        for item in batch
    ]
    
    # Batch the graphs
    batched_graphs1 = GraphBatch.from_data_list(graph_list1)
    batched_graphs2 = GraphBatch.from_data_list(graph_list2)
    
    return {
        "codes1": codes1,
        "texts1": texts1,
        "input_ids1": input_ids1,
        "attention_mask1": attention_masks1,
        "graph_features1": batched_graphs1.x,
        "graph_edge_index1": batched_graphs1.edge_index,
        "graph_batch1": batched_graphs1.batch,
        
        "codes2": codes2,
        "texts2": texts2,
        "input_ids2": input_ids2,
        "attention_mask2": attention_masks2,
        "graph_features2": batched_graphs2.x,
        "graph_edge_index2": batched_graphs2.edge_index,
        "graph_batch2": batched_graphs2.batch,
        
        "relation_types": relation_types
    }


# Helper class to create graph data objects
class GraphData(torch.utils.data.Dataset):
    """
    Helper class for creating PyTorch Geometric Data objects.
    """
    def __init__(self, x, edge_index):
        """
        Initialize a graph data object.
        
        Args:
            x: Node features
            edge_index: Graph connectivity in COO format
        """
        self.x = x
        self.edge_index = edge_index
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        from torch_geometric.data import Data
        return Data(x=self.x, edge_index=self.edge_index)


def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4, is_pair=False):
    """
    Create a dataloader for the dataset.
    
    Args:
        dataset: The dataset to load
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for loading
        is_pair: Whether the dataset contains pairs
    
    Returns:
        DataLoader object
    """
    collate = collate_fn_pairs if is_pair else collate_fn
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )
