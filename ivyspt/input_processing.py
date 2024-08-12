import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import numpy as np


def implied_volatility_surfaces(
    options_market_data,
    toy_sample=False,
    max_points=30,
    max_surfaces=10,
    random_state=0,
):
    # Group the data by Datetime and Symbol
    grouped_data = options_market_data.groupby(level=['Datetime', 'Symbol'])
    rng = np.random.default_rng(random_state)

    surfaces = []
    for (date, symbol), surface in grouped_data:

        if toy_sample:
            entire_size = len(surface['Implied Volatility'])
            sample = rng.choice(range(entire_size), size=min(max_points, entire_size), replace=False)

        surface_dict = {
            'Datetime': date,
            'Symbol': symbol,
            'Market Features': {
                'Market Return': surface['Market Return'].values[0],
                'Market Volatility': surface['Market Volatility'].values[0],
                'Treasury Rate': surface['Treasury Rate'].values[0],
            },
            'Surface': {
                'Log Moneyness': surface['Log Moneyness'].values if not toy_sample else surface['Log Moneyness'].values[sample],
                'Time to Maturity': surface['Time to Maturity'].values if not toy_sample else surface['Time to Maturity'].values[sample],
                'Implied Volatility': surface['Implied Volatility'].values if not toy_sample else surface['Implied Volatility'].values[sample],
            }
        }
        surfaces.append(surface_dict)

    if toy_sample:
        entire_size = len(surfaces)
        sample = rng.choice(range(entire_size), size=min(max_surfaces, entire_size), replace=False)  

        return [surfaces[i] for i in sample]

    return surfaces


def split_surfaces(
    data, 
    n_partitions=6,
    toy_sample=False,
    max_points=30,
    max_surfaces=10,
    random_state=0,
):
    # Extract unique sorted timestamps
    unique_timestamps = np.sort(data.index.get_level_values('Datetime').unique())
    
    # Split timestamps into partitions
    partitions = np.array_split(unique_timestamps, n_partitions)
    
    # Initialize lists to hold the final timestamps for each dataset
    train_times = []
    validation_times = []
    test_times = []

    for partition in partitions:
        # Remove the first day of the partition to avoid leakage
        if len(partition) > 1:
            partition = partition[1:]
        
        # Determine the number of timestamps for training, validation, and test sets
        partition_len = len(partition)
        train_len = int(0.8 * partition_len)
        valid_len = int(0.1 * partition_len)
        
        # Assign timestamps to train, validation, and test sets
        train_times.extend(partition[:train_len])
        validation_times.extend(partition[train_len:train_len + valid_len])
        test_times.extend(partition[train_len + valid_len:])
    
    # Now, use the timestamps to filter the data for each set
    train_set = data.query('Datetime in @train_times')
    validation_set = data.query('Datetime in @validation_times')
    test_set = data.query('Datetime in @test_times')

    train_surface = implied_volatility_surfaces(
        train_set,
        toy_sample,
        max_points,
        max_surfaces,
        random_state,
    )
    validation_surface = implied_volatility_surfaces(
        validation_set,
        toy_sample,
        max_points,
        max_surfaces,
        random_state,
    )
    test_surface = implied_volatility_surfaces(
        test_set,
        toy_sample,
        max_points,
        max_surfaces,
        random_state,
    )
    
    return train_surface, validation_surface, test_surface


class IVSurfaceDataset(Dataset):
    def __init__(
        self, 
        data, 
        mask_proportions, 
        random_state=0,
        n_query_points=None
    ):
        self.data = data
        self.mask_proportions = mask_proportions
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.n_query_points = n_query_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        surface_data = self.data[idx]
        
        # Extract the surface coordinates and volatilities
        points_coordinates = np.stack([
            surface_data['Surface']['Log Moneyness'], 
            surface_data['Surface']['Time to Maturity']
        ], axis=1)
        points_volatilities = surface_data['Surface']['Implied Volatility']

        # Select a random mask proportion
        proportion = self.rng.choice(self.mask_proportions)

        # Perform clustering
        n_clusters = int(np.ceil(1 / proportion))
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('kmeans', KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init='auto'))
        ])
        labels = pipeline.fit_predict(points_coordinates)
        masked_indices = []

        for cluster in range(n_clusters):
            cluster_indices = np.where(labels == cluster)[0]
            num_to_mask = int(np.ceil(len(cluster_indices) * proportion))
            masked_indices.extend(self.rng.choice(cluster_indices, size=num_to_mask, replace=False))
        
        masked_indices = np.array(masked_indices)
        unmasked_indices = np.setdiff1d(range(len(labels)), masked_indices)

        # Calculate Total Variance mean and std for unmasked points
        time_to_maturity_unmasked = points_coordinates[unmasked_indices, 1]
        total_variance_unmasked = time_to_maturity_unmasked * np.square(points_volatilities[unmasked_indices])

        tv_mean = np.mean(total_variance_unmasked)
        tv_std = np.std(total_variance_unmasked)

        # Define query indices based on n_query_points
        if self.n_query_points is None:
            query_indices = masked_indices
        else:
            query_indices = self.rng.choice(masked_indices, size=self.n_query_points, replace=False)

        time_to_maturity_query = points_coordinates[query_indices, 1]
        total_variance_query = time_to_maturity_query * np.square(points_volatilities[query_indices])    
            
        data_item = {
            'Datetime': surface_data['Datetime'],
            'Symbol': surface_data['Symbol'],
            'Mask Proportion': proportion,
            'Market Features': {
                'Market Return': torch.tensor(surface_data['Market Features']['Market Return'], dtype=torch.float32),
                'Market Volatility': torch.tensor(surface_data['Market Features']['Market Volatility'], dtype=torch.float32),
                'Treasury Rate': torch.tensor(surface_data['Market Features']['Treasury Rate'], dtype=torch.float32),
                'TV Mean': torch.tensor(tv_mean, dtype=torch.float32),  
                'TV Std.': torch.tensor(tv_std, dtype=torch.float32),  
            },
            'Input Surface': {
                'Log Moneyness': torch.tensor(points_coordinates[unmasked_indices, 0], dtype=torch.float32),
                'Time to Maturity': torch.tensor(time_to_maturity_unmasked, dtype=torch.float32),
                'Total Variance': torch.tensor(total_variance_unmasked, dtype=torch.float32)
            },
            'Query Points': {
                'Log Moneyness': torch.tensor(points_coordinates[query_indices, 0], dtype=torch.float32),
                'Time to Maturity': torch.tensor(time_to_maturity_query, dtype=torch.float32),
                'Total Variance': torch.tensor(total_variance_query, dtype=torch.float32)  
            }
        }

        return data_item

    @staticmethod
    def collate_fn(batch):
        batched_data = {
            'Datetime': [item['Datetime'] for item in batch],
            'Symbol': [item['Symbol'] for item in batch],
            'Mask Proportion': [item['Mask Proportion'] for item in batch],
            'Market Features': {
                'Market Return': default_collate([item['Market Features']['Market Return'] for item in batch]),
                'Market Volatility': default_collate([item['Market Features']['Market Volatility'] for item in batch]),
                'Treasury Rate': default_collate([item['Market Features']['Treasury Rate'] for item in batch]),
                'TV Mean': default_collate([item['Market Features']['TV Mean'] for item in batch]),
                'TV Std.': default_collate([item['Market Features']['TV Std.'] for item in batch]),
            },
            'Input Surface': {
                'Log Moneyness': [item['Input Surface']['Log Moneyness'] for item in batch],
                'Time to Maturity': [item['Input Surface']['Time to Maturity'] for item in batch],
                'Total Variance': [item['Input Surface']['Total Variance'] for item in batch],
            },
            'Query Points': {
                'Log Moneyness': [item['Query Points']['Log Moneyness'].requires_grad_(True) for item in batch],
                'Time to Maturity': [item['Query Points']['Time to Maturity'].requires_grad_(True) for item in batch],
                'Total Variance': [item['Query Points']['Total Variance'] for item in batch],
            }
        }

        return batched_data
