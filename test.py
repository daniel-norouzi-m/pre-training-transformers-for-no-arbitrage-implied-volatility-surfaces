import numpy as np
import pandas as pd
import random
import torch
import gc
# Set the random seed for reproducibility
RANDOM_STATE = 0
N_JOBS = 8
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
HYPERPARAMETERS = {
    'Input Preprocessing' : {
        'Mask Proportions' : [0.1, 0.3, 0.5, 0.7],
        'Number of Query Points' : 1,
        'Batch Size' : 4
    },
    'Surface Embedding' : {
        'Embedding Dimension' : 8,
    },
    'Surface Encoding' : {
        'Number of Heads' : 4,
        'FFN Hidden Dimension' : 16,
        'Attention Dropout' : 0.1,
        'Gate Dropout' : 0.1,
        'FFN Dropout' : 0.1,
        'Number of Blocks' : 2,
        'External Feature Dimension' : 5,
    },
    'Adaptive Loss Weights' : {
        'Asymmetry' : 1.,
    },
    'Trainer' : {
        'Pre-Train' : {
            'Number of Epochs' : 10,
            'Warmup Ratio' : 0.15,
            'Peak Learning Rate' : 1e-3,
            'Minimal Learning Rate' : 1e-5,
            'Gradient Clipping' : 10,
            'Adam Betas' : (0.9, 0.999),
            'Adam Epsilon' : 1e-8,
            'Adam Weight Decay' : 0.05,
            'Layer-Wise Decay' : None,
        },
        'Fine-Tune' : {
            'Number of Epochs' : 10,
            'Warmup Ratio' : 0.1,
            'Peak Learning Rate' : 1e-3,
            'Minimal Learning Rate' : 1e-6,
            'Gradient Clipping' : 0,
            'Adam Betas' : (0.9, 0.999),
            'Adam Epsilon' : 1e-8,
            'Adam Weight Decay' : 0.05,
            'Layer-Wise Decay' : 0.9,
        }
    }
}
## Dataset
aapl_googl_data = pd.read_csv('volatility_surface_AAPL_GOOGL_2013_01_2013_06.csv', parse_dates=True, index_col=[0, 1], date_format="ISO8601")
aapl_googl_data
# import yfinance as yf
# # Load the data
# aapl_googl_data = pd.read_csv('volatility_surface_AAPL_GOOGL_2013_01_2013_06.csv', parse_dates=True, index_col=[0, 1], date_format="ISO8601")
# # Fetch historical close and adjusted close prices for AAPL and GOOGL
# aapl = yf.download('AAPL', start='2013-01-01', end='2013-06-30')
# googl = yf.download('GOOG', start='2013-01-01', end='2013-06-30')
# # Create a dictionary to hold close and adjusted close prices for easy access
# prices = {
#     'AAPL': {'Close': aapl['Close'], 'Adj Close': aapl['Adj Close']},
#     'GOOGL': {'Close': googl['Close'], 'Adj Close': googl['Adj Close']}
# }
# # Define a function to calculate the modified log moneyness
# def modified_log_moneyness(row):
#     symbol = row.name[1]
#     date = row.name[0]
#     close_price = prices[symbol]['Close'][date]
#     adj_close_price = prices[symbol]['Adj Close'][date]
#     log_moneyness = row['Log Moneyness'] + np.log(close_price / adj_close_price)
    
#     treasury_rate = row['Treasury Rate']
#     time_to_maturity = row['Time to Maturity']
#     exponential_treasury_rate = np.log(1 + treasury_rate)
#     discount_factor = np.exp(-exponential_treasury_rate * time_to_maturity)
    
#     return log_moneyness * discount_factor
# # Apply the function to each row
# aapl_googl_data['Log Moneyness'] = aapl_googl_data.apply(modified_log_moneyness, axis=1)
def implied_volatility_surfaces(options_market_data):
    # Group the data by Datetime and Symbol
    grouped_data = options_market_data.groupby(level=['Datetime', 'Symbol'])

    surfaces = []
    for (date, symbol), surface in grouped_data:
        surface_dict = {
            'Datetime': date,
            'Symbol': symbol,
            'Market Features': {
                'Market Return': surface['Market Return'].values[0],
                'Market Volatility': surface['Market Volatility'].values[0],
                'Treasury Rate': surface['Treasury Rate'].values[0],
            },
            'Surface': {
                'Log Moneyness': surface['Log Moneyness'].values[:30],
                'Time to Maturity': surface['Time to Maturity'].values[:30],
                'Implied Volatility': surface['Implied Volatility'].values[:30],
            }
        }
        surfaces.append(surface_dict)

    return surfaces[:30]

surfaces = implied_volatility_surfaces(aapl_googl_data)
surfaces[0]
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import numpy as np

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
        masked_indices = np.array([], dtype=int)

        for cluster in range(n_clusters):
            cluster_indices = np.where(labels == cluster)[0]
            num_to_mask = int(np.ceil(len(cluster_indices) * proportion))
            masked_indices = np.append(masked_indices, [self.rng.choice(cluster_indices, size=num_to_mask, replace=False)])
        
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


# Assuming surfaces is the output from the implied_volatility_surfaces function
# mask_proportions = HYPERPARAMETERS['Input Preprocessing']['Mask Proportions']  
# n_query_points = HYPERPARAMETERS['Input Preprocessing']['Number of Query Points']  
# dataset = IVSurfaceDataset(surfaces, mask_proportions, RANDOM_STATE, n_query_points)
# data_loader = DataLoader(
#     dataset, 
#     batch_size=HYPERPARAMETERS['Input Preprocessing']['Batch Size'], 
#     shuffle=True, 
#     num_workers=0, 
#     collate_fn=IVSurfaceDataset.collate_fn
# )

# Fetch one batch from the DataLoader
# batch = next(iter(data_loader))
# batch
## Surface Embedding
### Components
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

class SurfaceBatchNorm(nn.Module):
    def __init__(
        self, 
        num_features=1, 
        momentum=0.1
    ):
        super(SurfaceBatchNorm, self).__init__()
        self.log_moneyness_bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.time_to_maturity_bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.market_return_bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.market_volatility_bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.treasury_rate_bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.tv_mean_bn = nn.BatchNorm1d(num_features, momentum=momentum)
        self.tv_std_bn = nn.BatchNorm1d(num_features, momentum=momentum)

    def forward(self, batch):
        # Concatenate all tensors from the Input Surface into one tensor for each feature
        input_surface_log_moneyness = torch.cat([x for x in batch['Input Surface']['Log Moneyness']])
        input_surface_time_to_maturity = torch.cat([x for x in batch['Input Surface']['Time to Maturity']])

        # Concatenate Input Surface tensors with Query Points tensors
        total_log_moneyness = torch.cat([input_surface_log_moneyness] + [x for x in batch['Query Points']['Log Moneyness']])
        total_time_to_maturity = torch.cat([input_surface_time_to_maturity] + [x for x in batch['Query Points']['Time to Maturity']])

        # Normalize Log Moneyness and Time to Maturity
        norm_log_moneyness = self.log_moneyness_bn(total_log_moneyness.unsqueeze(1)).squeeze(1)
        norm_time_to_maturity = self.time_to_maturity_bn(total_time_to_maturity.unsqueeze(1)).squeeze(1)

        # Split the normalized results back to corresponding structures
        input_surface_sizes = [len(x) for x in batch['Input Surface']['Log Moneyness']]
        query_points_sizes = [len(x) for x in batch['Query Points']['Log Moneyness']]
        total_input_size = sum(input_surface_sizes)

        # Normalizing Market Features
        market_features = batch['Market Features']
        norm_market_return = self.market_return_bn(market_features['Market Return'].unsqueeze(1)).squeeze(1)
        norm_market_volatility = self.market_volatility_bn(market_features['Market Volatility'].unsqueeze(1)).squeeze(1)
        norm_treasury_rate = self.treasury_rate_bn(market_features['Treasury Rate'].unsqueeze(1)).squeeze(1)
        norm_tv_mean = self.tv_mean_bn(market_features['TV Mean'].unsqueeze(1)).squeeze(1)
        norm_tv_std = self.tv_std_bn(market_features['TV Std.'].unsqueeze(1)).squeeze(1)

        # Reconstructing the batch with normalized data
        output = {
            'Datetime': batch['Datetime'],
            'Symbol': batch['Symbol'],
            'Mask Proportion': batch['Mask Proportion'],
            'Market Features': {
                'Market Return': norm_market_return,
                'Market Volatility': norm_market_volatility,
                'Treasury Rate': norm_treasury_rate,
                'TV Mean': norm_tv_mean,
                'TV Std.': norm_tv_std
            },
            'Input Surface': {
                'Log Moneyness': list(torch.split(norm_log_moneyness[:total_input_size], input_surface_sizes)),
                'Time to Maturity': list(torch.split(norm_time_to_maturity[:total_input_size], input_surface_sizes)),
                'Total Variance': batch['Input Surface']['Total Variance']
            },
            'Query Points': {
                'Log Moneyness': list(torch.split(norm_log_moneyness[total_input_size:], query_points_sizes)),
                'Time to Maturity': list(torch.split(norm_time_to_maturity[total_input_size:], query_points_sizes)),
                'Total Variance': batch['Query Points']['Total Variance']
            }
        }

        # Ensure requires_grad is True for query point values
        # for key in output['Query Points']:
        #     if key != 'Implied Volatility':  # We only set requires_grad for Log Moneyness and Time to Maturity
        #         for tensor in output['Query Points'][key]:
        #             tensor.requires_grad_()

        return output

# Usage
# surfacebatchnorm = SurfaceBatchNorm()
# processed_batch = surfacebatchnorm(batch)
# processed_batch
import torch
import torch.nn as nn
import numpy as np

class EllipticalRBFKernel(nn.Module):
    def __init__(
        self, 
        input_dim, 
        bandwidth, 
        remove_kernel=False
    ):
        super(EllipticalRBFKernel, self).__init__()
        self.bandwidth = bandwidth
        # Initialize the log of the scale vector to zero, which corresponds to scale factors of one
        self.log_scale = nn.Parameter(torch.zeros(input_dim))
        self.remove_kernel = remove_kernel

    def forward(self, distances):
        if self.remove_kernel:
            # Create a mask for the condition check
            all_zeros = torch.all(distances==0.0, dim=-1)
            result = torch.where(
                all_zeros, 
                torch.full(distances.shape[:-1], 1.0, device=distances.device),
                torch.full(distances.shape[:-1], 1e-10, device=distances.device)
            )
            return result
        # Convert log scale to actual scale values
        scale = torch.exp(self.log_scale)
        
        # Calculate the scaled distances
        scaled_distances = (distances ** 2) * scale  # Element-wise multiplication by scale

        # Normalize by the trace of the scale matrix
        trace_scale_matrix = torch.sum(scale)
        normalized_distances = torch.sum(scaled_distances, dim=-1) / trace_scale_matrix

        # Compute the RBF kernel output using the normalized distances
        kernel_values = torch.exp(-normalized_distances / (2 * self.bandwidth ** 2))

        return kernel_values

class SurfaceContinuousKernelPositionalEmbedding(nn.Module):
    def __init__(
        self, 
        d_embedding,
        remove_kernel=False,
        remove_positional_embedding=False
    ):
        super(SurfaceContinuousKernelPositionalEmbedding, self).__init__()
        self.d_embedding = d_embedding
        self.remove_positional_embedding = remove_positional_embedding

        # Initialize multiple RBF kernels, each with a different fixed bandwidth
        self.kernels = nn.ModuleList()
        for i in range(1, d_embedding + 1):
            bandwidth_value = torch.erfinv(torch.tensor(i / (d_embedding + 1))) * np.sqrt(2)
            self.kernels.append(
                EllipticalRBFKernel(
                    bandwidth=bandwidth_value, 
                    input_dim=2, 
                    remove_kernel=remove_kernel
                )
            )

        self.input_surface_layer_norm = nn.LayerNorm(d_embedding)
        self.query_points_layer_norm = nn.LayerNorm(d_embedding)

        # Initialize learnable scaling parameter (the base for positional embedding)
        self.log_scale = nn.Parameter(torch.log(torch.tensor(10000.0)))

    def forward(
        self, 
        input_surface_batch, 
        query_points_batch
    ):
        batch_size = len(input_surface_batch['Log Moneyness'])

        input_surface_embeddings = []
        query_points_embeddings = []

        for i in range(batch_size):
            # Extract the coordinates and implied volatilities for each surface in the batch
            surface_coords = torch.stack([
                input_surface_batch['Log Moneyness'][i], 
                input_surface_batch['Time to Maturity'][i]
            ], dim=-1)
            surface_tvs = input_surface_batch['Total Variance'][i]

            query_coords = torch.stack([
                query_points_batch['Log Moneyness'][i], 
                query_points_batch['Time to Maturity'][i]
            ], dim=-1)

            all_coords = torch.cat((surface_coords, query_coords), dim=0)

            # Compute the pairwise differences between all points and the input surface points
            point_differences = all_coords.unsqueeze(1) - surface_coords.unsqueeze(0)  # (n+m, n, 2)

            # Initialize the output embeddings for the current surface with d_embedding channels
            all_embedded = torch.zeros((all_coords.shape[0], self.d_embedding), dtype=torch.float32, device=surface_coords.device)

            for kernel_idx, kernel in enumerate(self.kernels):
                # Apply the RBF kernel to each distance vector 
                kernel_outputs = kernel(point_differences)

                # Compute the weighted sum of TVs based on the kernel outputs
                weighted_sum = (kernel_outputs * surface_tvs.unsqueeze(0)).sum(dim=1)
                normalization_factor = kernel_outputs.sum(dim=1)

                all_embedded[:, kernel_idx] = weighted_sum / normalization_factor    

            # Split the embeddings into input surface and query points embeddings
            input_surface_embedded = all_embedded[:surface_coords.shape[0], :]
            query_points_embedded = all_embedded[surface_coords.shape[0]:, :]

            # Normalize the embedded surfaces
            input_surface_embedded = self.input_surface_layer_norm(input_surface_embedded)
            query_points_embedded = self.query_points_layer_norm(query_points_embedded)

            # Positional embedding for input surface points
            input_surface_pe = self._compute_positional_embedding(surface_coords)

            # Positional embedding for query points
            query_points_pe = self._compute_positional_embedding(query_coords)

            # Add positional embeddings with a factor of sqrt(2)
            input_surface_final = input_surface_embedded + input_surface_pe * np.sqrt(2)
            query_points_final = query_points_embedded + query_points_pe * np.sqrt(2)

            # Append the encoded surface for this input surface to the batch list
            input_surface_embeddings.append(input_surface_final)
            query_points_embeddings.append(query_points_final)

        # Keep all encoded surfaces as lists to handle variable lengths
        return {
            'Input Surface': input_surface_embeddings,
            'Query Points': query_points_embeddings
        }

    def _compute_positional_embedding(
        self, 
        coords, 
    ):
        positional_embedding = torch.zeros(coords.size(0), self.d_embedding, device=coords.device)

        if not self.remove_positional_embedding:
            for i in range(self.d_embedding // 4):
                div_factor = torch.exp(self.log_scale) ** (4 * i / self.d_embedding)
                positional_embedding[:, 4 * i] = torch.sin(coords[:, 0] / div_factor)
                positional_embedding[:, 4 * i + 1] = torch.cos(coords[:, 0] / div_factor)
                positional_embedding[:, 4 * i + 2] = torch.sin(coords[:, 1] / div_factor)
                positional_embedding[:, 4 * i + 3] = torch.cos(coords[:, 1] / div_factor)

        return positional_embedding

# Example of initializing and using this module
# d_embedding = HYPERPARAMETERS['Surface Embedding']['Embedding Dimension']  # Desired number of output channels

# continuous_kernel_positional_embedding = SurfaceContinuousKernelPositionalEmbedding(d_embedding=d_embedding)
# kernel_positional_embedded_batch = continuous_kernel_positional_embedding(processed_batch['Input Surface'], processed_batch['Query Points'])
# kernel_positional_embedded_batch
## Block
import torch
import torch.nn as nn
import numpy as np

class SurfaceEmbedding(nn.Module):
    def __init__(
        self, 
        d_embedding, 
        momentum=0.1,
        remove_kernel=False,
        remove_positional_embedding=False
    ):
        super(SurfaceEmbedding, self).__init__()
        self.batch_norm = SurfaceBatchNorm(num_features=1, momentum=momentum)
        self.kernel_positional_embedding = SurfaceContinuousKernelPositionalEmbedding(d_embedding, remove_kernel, remove_positional_embedding)
        self.layer_norm = nn.LayerNorm(d_embedding)
        self.mask_token = nn.Parameter(torch.randn(d_embedding))

    def forward(self, batch):
        # Apply batch normalization
        norm_batch = self.batch_norm(batch)

        # Extract market features from processed batch and create external_features_batch tensor
        market_features = norm_batch['Market Features']
        external_features_batch = torch.stack([
            market_features['Market Return'],
            market_features['Market Volatility'],
            market_features['Treasury Rate'],
            market_features['TV Mean'],
            market_features['TV Std.']
        ], dim=-1)  # (batch, features)

        # Compute kernel and positional embeddings
        embeddings = self.kernel_positional_embedding(norm_batch['Input Surface'], norm_batch['Query Points'])

        input_surface_embeddings = embeddings['Input Surface']
        query_points_embeddings = embeddings['Query Points']

        embedded_sequences = []

        for input_surface_embedding, query_points_embedding in zip(input_surface_embeddings, query_points_embeddings):
            # Add mask token to the query point embeddings
            masked_query_points_embedding = query_points_embedding + self.mask_token

            # Combine input surface embeddings and masked query points embeddings
            combined_sequence = torch.cat((input_surface_embedding, masked_query_points_embedding), dim=0)

            # Apply layer normalization
            combined_sequence = self.layer_norm(combined_sequence)

            embedded_sequences.append(combined_sequence)

        return embedded_sequences, external_features_batch


# # Example of initializing and using this module
# d_embedding = HYPERPARAMETERS['Surface Embedding']['Embedding Dimension']  # Desired number of output channels
# surface_embedding = SurfaceEmbedding(d_embedding=d_embedding)
# embedded_sequences_batch, external_features_batch = surface_embedding(batch)
# embedded_sequences_batch
# Surface Encoding
## Encoder
import torch
import torch.nn as nn

class ResidualNorm(nn.Module):
    def __init__(self, d_embedding):
        super(ResidualNorm, self).__init__()
        self.norm = nn.LayerNorm(d_embedding)

    def forward(
        self, 
        x, 
        sublayer_output
    ):
        return self.norm(x + sublayer_output)
    

class GatedAttentionFusion(nn.Module):
    def __init__(
        self, 
        d_embedding,
        gate_dropout,
        weight_initializer_std=0.02,
        bias_initializer_value=10.0,
        remove_external_attention=False,
        remove_gate=False
    ):
        super(GatedAttentionFusion, self).__init__()
        self.gate_layer = nn.Sequential(
            nn.Linear(d_embedding * 2, d_embedding),
            nn.Sigmoid(),
            nn.Dropout(gate_dropout)
        )
        self.remove_external_attention = remove_external_attention
        self.remove_gate = remove_gate

        # Initialize weights and biases
        self._initialize_weights(weight_initializer_std, bias_initializer_value)

    def _initialize_weights(
        self, 
        std, 
        bias_value
    ):
        for module in self.gate_layer:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.constant_(module.bias, bias_value)

    def forward(
        self, 
        self_attention_output, 
        external_attention_output
    ):
        if self.remove_external_attention:

            return self_attention_output

        if self.remove_gate:  

            return self_attention_output + external_attention_output
        # Concatenate self-attention and external attention outputs
        concatenated_output = torch.cat((self_attention_output, external_attention_output), dim=-1)
        # Compute gate values
        gate_values = self.gate_layer(concatenated_output)
        # Calculate gated embedding
        gated_embedding = gate_values * self_attention_output + (1 - gate_values) * external_attention_output

        return gated_embedding
    
    
class FeedForwardNetwork(nn.Module):
    def __init__(
        self, 
        d_embedding, 
        ffn_hidden_dim, 
        ffn_dropout, 
        layer_depth, 
        weight_initializer_std=0.02, 
        bias_initializer_value=0,
    ):
        super(FeedForwardNetwork, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(d_embedding, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(ffn_hidden_dim, d_embedding),
            nn.Dropout(ffn_dropout)
        )

        self.layer_depth = layer_depth
        self._initialize_weights(weight_initializer_std, bias_initializer_value)

    def forward(self, x):
        return self.feedforward(x)
    
    def _initialize_weights(
        self, 
        std, 
        bias_value
    ):
        for i, module in enumerate(self.feedforward):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.constant_(module.bias, bias_value)
                
                # Rescale the output matrices of the last linear projection
                if i == len(self.feedforward) - 2:
                    scale_factor = 1 / (2 * self.layer_depth) ** 0.5
                    module.weight.data *= scale_factor


class Encoder(nn.Module):
    def __init__(
        self, 
        d_embedding, 
        n_heads, 
        ffn_hidden_dim, 
        attention_dropout, 
        gate_dropout,
        ffn_dropout,
        external_dim,
        layer_depth,
        weight_initializer_std=0.02,
        linear_bias_initializer_value=0.0,
        gate_bias_initializer_value=10.0,
        remove_external_attention=False,
        remove_gate=False
    ):
        super(Encoder, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_embedding, 
            num_heads=n_heads, 
            dropout=attention_dropout
        )
        self.residual_norm_self_attention = ResidualNorm(d_embedding)
        self.external_attention = nn.MultiheadAttention(
            embed_dim=d_embedding, 
            num_heads=n_heads, 
            kdim=external_dim, 
            vdim=external_dim, 
            dropout=attention_dropout
        )
        self.residual_norm_external_attention = ResidualNorm(d_embedding)
        self.gated_attention_fusion = GatedAttentionFusion(
            d_embedding, 
            gate_dropout,
            weight_initializer_std,
            gate_bias_initializer_value,
            remove_external_attention, 
            remove_gate,
        )
        self.residual_norm_fusion = ResidualNorm(d_embedding)
        self.feed_forward = FeedForwardNetwork(
            d_embedding, 
            ffn_hidden_dim, 
            ffn_dropout, 
            layer_depth, 
            weight_initializer_std, 
            linear_bias_initializer_value
        )
        self.residual_norm_ffn = ResidualNorm(d_embedding)
        # Initialize self-attention
        self._initialize_attention_weights(self.self_attention, weight_initializer_std, linear_bias_initializer_value, layer_depth)
        # Initialize external-attention
        self._initialize_attention_weights(self.external_attention, weight_initializer_std, linear_bias_initializer_value, layer_depth)

    def _initialize_attention_weights(
        self, 
        attention_module, 
        weight_initializer_std, 
        linear_bias_initializer_value, 
        layer_depth
    ):
        if attention_module._qkv_same_embed_dim:
            nn.init.normal_(attention_module.in_proj_weight, mean=0.0, std=weight_initializer_std)
        else:
            nn.init.normal_(attention_module.q_proj_weight, mean=0.0, std=weight_initializer_std)
            nn.init.normal_(attention_module.k_proj_weight, mean=0.0, std=weight_initializer_std)
            nn.init.normal_(attention_module.v_proj_weight, mean=0.0, std=weight_initializer_std)

        if attention_module.in_proj_bias is not None:
            nn.init.constant_(attention_module.in_proj_bias, linear_bias_initializer_value)
            nn.init.constant_(attention_module.out_proj.bias, linear_bias_initializer_value)
        
        if attention_module.bias_k is not None:
            nn.init.constant_(attention_module.bias_k, linear_bias_initializer_value)
        if attention_module.bias_v is not None:
            nn.init.constant_(attention_module.bias_v, linear_bias_initializer_value)
        
        # Transformer layer rescaling for output weights
        scale_factor = 1 / (2 * layer_depth) ** 0.5
        nn.init.normal_(attention_module.out_proj.weight, mean=0.0, std=weight_initializer_std * scale_factor)

    def forward(
        self, 
        surface_embeddings, 
        external_features,
        output_attention_map=False
    ):
        # Self-Attention
        self_attention_output, self_attention_weights = self.self_attention(surface_embeddings, surface_embeddings, surface_embeddings)
        self_attention_output = self.residual_norm_self_attention(surface_embeddings, self_attention_output)
        # External Attention
        external_attention_output, external_attention_weights = self.external_attention(surface_embeddings, external_features, external_features) 
        external_attention_output = self.residual_norm_external_attention(surface_embeddings, external_attention_output)
        # Gated Attention Fusion
        gated_embedding = self.gated_attention_fusion(self_attention_output, external_attention_output)
        gated_embedding = self.residual_norm_fusion(surface_embeddings, gated_embedding)
        # Feed-Forward Network
        ffn_output = self.feed_forward(gated_embedding)
        # Final Residual Connection and Layer Normalization
        surface_embeddings = self.residual_norm_ffn(gated_embedding, ffn_output)

        if output_attention_map:
            # Remove the batch dimension for attention weights
            return surface_embeddings, self_attention_weights.squeeze(0), external_attention_weights.squeeze(0)
        
        return surface_embeddings, None, None

class SurfaceEncoder(nn.Module):
    def __init__(
        self, 
        d_embedding, 
        num_encoder_blocks,
        n_heads, 
        ffn_hidden_dim,
        attention_dropout, 
        gate_dropout,
        ffn_dropout,
        external_dim,
        weight_initializer_std=0.02,
        linear_bias_initializer_value=0.0,
        gate_bias_initializer_value=10.0,
        remove_external_attention=False,
        remove_gate=False
    ):
        super(SurfaceEncoder, self).__init__()
        self.encoders = nn.ModuleList([
            Encoder(
                d_embedding, 
                n_heads, 
                ffn_hidden_dim, 
                attention_dropout, 
                gate_dropout,
                ffn_dropout,
                external_dim,
                (i + 1),
                weight_initializer_std,
                linear_bias_initializer_value,
                gate_bias_initializer_value,
                remove_external_attention,
                remove_gate
            )
            for i in range(num_encoder_blocks)
        ])

    def forward(
        self, 
        embedded_sequences_batch, 
        external_features_batch,
        output_attention_map=False
    ):
        batch_size = len(embedded_sequences_batch)
        encoded_sequences_batch = []
        self_attention_maps = []
        external_attention_maps = []

        for i in range(batch_size):
            surface_embeddings = embedded_sequences_batch[i].unsqueeze(1) 
            external_features = external_features_batch[i].unsqueeze(0).unsqueeze(0)

            for j, encoder in enumerate(self.encoders):
                if j == len(self.encoders) - 1 and output_attention_map:
                    surface_embeddings, self_attention_map, external_attention_map = encoder(surface_embeddings, external_features, output_attention_map)
                    
                else:
                    surface_embeddings, _, _ = encoder(surface_embeddings, external_features)
                
            encoded_sequences_batch.append(surface_embeddings.squeeze(1))
            if output_attention_map:
                self_attention_maps.append(self_attention_map)
                external_attention_maps.append(external_attention_map)

        if output_attention_map:
            return encoded_sequences_batch, self_attention_maps, external_attention_maps
        
        return encoded_sequences_batch, None, None    

# Example of initializing and using these modules
# torch.manual_seed(RANDOM_STATE)
# n_heads = HYPERPARAMETERS['Surface Encoding']['Number of Heads']
# ffn_hidden_dim = HYPERPARAMETERS['Surface Encoding']['FFN Hidden Dimension']
# attention_dropout = HYPERPARAMETERS['Surface Encoding']['Attention Dropout']
# gate_dropout = HYPERPARAMETERS['Surface Encoding']['Gate Dropout']
# ffn_dropout = HYPERPARAMETERS['Surface Encoding']['FFN Dropout']
# num_encoder_blocks = HYPERPARAMETERS['Surface Encoding']['Number of Blocks']
# external_dim = 5

# surface_encoder = SurfaceEncoder(
#     d_embedding, 
#     num_encoder_blocks,
#     n_heads, 
#     ffn_hidden_dim, 
#     attention_dropout, 
#     gate_dropout, 
#     ffn_dropout, 
#     external_dim, 
# )

# Assume embedded_sequences_batch is the output of the SurfaceEmbedding module and
# external_features is the formatted external market features batch
# encoded_sequences_batch, self_attention_map_batch, external_attention_map_batch = surface_encoder(embedded_sequences_batch, external_features_batch)
# encoded_sequences_batch
# IvySPT
import torch
import torch.nn as nn

class IvySPT(nn.Module):
    def __init__(
        self, 
        d_embedding, 
        num_encoder_blocks,
        n_heads, 
        ffn_hidden_dim,
        attention_dropout, 
        gate_dropout,
        ffn_dropout,
        external_dim,
        weight_initializer_std=0.02,
        linear_bias_initializer_value=0.0,
        gate_bias_initializer_value=10.0,
        remove_kernel=False,
        remove_positional_embedding=False,
        remove_external_attention=False,
        remove_gate=False
    ):
        super(IvySPT, self).__init__()
        self.surface_embedding = SurfaceEmbedding(
            d_embedding, 
            remove_kernel, 
            remove_positional_embedding
        )
        self.surface_encoder = SurfaceEncoder(
            d_embedding, 
            num_encoder_blocks,
            n_heads, 
            ffn_hidden_dim,
            attention_dropout, 
            gate_dropout,
            ffn_dropout,
            external_dim,
            weight_initializer_std,
            linear_bias_initializer_value,
            gate_bias_initializer_value,
            remove_external_attention,
            remove_gate
        )
        self.final_layer = nn.Linear(d_embedding, 1)
        nn.init.normal_(self.final_layer.weight, mean=0.0, std=weight_initializer_std * (1 / (2 * (num_encoder_blocks + 1)) ** 0.5))
        nn.init.constant_(self.final_layer.bias, linear_bias_initializer_value)

    def forward(
        self, 
        batch,
        output_attention_map=False
    ):
        # Obtain the embedded sequences and external features from the SurfaceEmbedding module
        embedded_sequences_batch, external_features_batch = self.surface_embedding(batch)

        # Encode the sequences using the SurfaceEncoder module
        encoded_sequences_batch, self_attention_maps, external_attention_maps = self.surface_encoder(
            embedded_sequences_batch, 
            external_features_batch, 
            output_attention_map
        )

        # List to hold the implied volatility estimates for each query point in the batch
        tv_estimates_batch = []

        query_self_attention_maps = []
        query_external_attention_maps = []

        for i in range(len(encoded_sequences_batch)):
            # Extract the encoded sequence
            encoded_sequence = encoded_sequences_batch[i]

            # Determine the number of query points for this sequence
            num_query_points = len(batch['Query Points']['Log Moneyness'][i])

            # Extract the encoded query points (last num_query_points elements in the sequence)
            encoded_query_points = encoded_sequence[-num_query_points:]

            # Estimate the implied volatility for each query point using the fully connected layer
            tv_estimates = self.final_layer(encoded_query_points).squeeze(-1)

            # Append the estimates to the batch list
            tv_estimates_batch.append(tv_estimates)

            if output_attention_map:
                # Extract the attention maps for the query points
                self_attention_map = self_attention_maps[i][-num_query_points:]
                external_attention_map = external_attention_maps[i][-num_query_points:]

                query_self_attention_maps.append(self_attention_map)
                query_external_attention_maps.append(external_attention_map)

        if output_attention_map:
            return tv_estimates_batch, query_self_attention_maps, query_external_attention_maps
        
        return tv_estimates_batch, None, None

# Example of initializing and using this module
torch.manual_seed(RANDOM_STATE)
n_heads = HYPERPARAMETERS['Surface Encoding']['Number of Heads']
ffn_hidden_dim = HYPERPARAMETERS['Surface Encoding']['FFN Hidden Dimension']
attention_dropout = HYPERPARAMETERS['Surface Encoding']['Attention Dropout']
gate_dropout = HYPERPARAMETERS['Surface Encoding']['Gate Dropout']
ffn_dropout = HYPERPARAMETERS['Surface Encoding']['FFN Dropout']
num_encoder_blocks = HYPERPARAMETERS['Surface Encoding']['Number of Blocks']
d_embedding = HYPERPARAMETERS['Surface Embedding']['Embedding Dimension']  # Desired number of output channels
external_dim = 5

ivy_spt = IvySPT(
    d_embedding, 
    num_encoder_blocks,
    n_heads, 
    ffn_hidden_dim,
    attention_dropout, 
    gate_dropout,
    ffn_dropout,
    external_dim
)

# # Pass the batch through the IvySPT model to get implied volatility estimates
# tv_estimates_batch, self_attention_maps, external_attention_maps = ivy_spt(batch, output_attention_map=False)
# gc.collect()
# tv_estimates_batch
# batch['Query Points']['Total Variance']
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceArbitrageFreeLoss(nn.Module):
    def __init__(self):
        super(SurfaceArbitrageFreeLoss, self).__init__()

    def forward(
        self, 
        tv_estimates_batch, 
        batch,
        testing_mode=False,
        epsilon=1e-5  # Small value to prevent division by zero
    ):
        mspe_loss_sum = 0.0
        calendar_arbitrage_loss_sum = 0.0
        butterfly_arbitrage_loss_sum = 0.0
        total_elements = 0
        loss_records = []

        for total_implied_variance, target_variance, time_to_maturity, log_moneyness in zip(
            tv_estimates_batch, 
            batch['Query Points']['Total Variance'], 
            batch['Query Points']['Time to Maturity'], 
            batch['Query Points']['Log Moneyness']
        ):
            sequence_length = total_implied_variance.size(0)
            total_elements += sequence_length

            # Calculate mean squared percentage error between model estimates and target variances
            percentage_error = (total_implied_variance - target_variance) / (target_variance + epsilon)
            mspe_loss = torch.sum(percentage_error ** 2)
            mspe_loss_sum += mspe_loss

            unit_vectors = torch.eye(sequence_length, device=total_implied_variance.device)

            # Compute gradients needed for arbitrage conditions
            w_t = torch.stack([
                torch.autograd.grad(
                    outputs=total_implied_variance, 
                    inputs=time_to_maturity,
                    grad_outputs=vec, 
                    create_graph=True   
                )[0]
                for vec in unit_vectors
            ]).diag()

            w_x = torch.stack([
                torch.autograd.grad(
                    outputs=total_implied_variance, 
                    inputs=log_moneyness,
                    grad_outputs=vec, 
                    create_graph=True   
                )[0]
                for vec in unit_vectors
            ]).diag()

            w_xx = torch.stack([
                torch.autograd.grad(
                    outputs=w_x, 
                    inputs=log_moneyness, 
                    grad_outputs=vec,
                    create_graph=True   
                )[0]
                for vec in unit_vectors
            ]).diag()

            # Calculate Calendar Arbitrage Loss
            calendar_arbitrage_loss = torch.clamp(-w_t, min=0) ** 2
            calendar_arbitrage_loss_sum += calendar_arbitrage_loss.sum()

            # Calculate Butterfly Arbitrage Loss
            w = total_implied_variance
            g = (1 - log_moneyness * w_x / (2 * w)) ** 2 - w_x / 4 * (1 / w + 1 / 4) + w_xx / 2
            butterfly_arbitrage_loss = torch.clamp(-g, min=0) ** 2
            butterfly_arbitrage_loss_sum += butterfly_arbitrage_loss.sum()
            if testing_mode:
                record = {
                    'MSPE Loss': mspe_loss.mean().item(),
                    'Calendar Arbitrage Loss': calendar_arbitrage_loss.mean().item(),
                    'Butterfly Arbitrage Loss': butterfly_arbitrage_loss.mean().item()
                }
                loss_records.append(record)

        # Calculate mean losses
        mspe_loss = mspe_loss_sum / total_elements
        calendar_arbitrage_loss = calendar_arbitrage_loss_sum / total_elements
        butterfly_arbitrage_loss = butterfly_arbitrage_loss_sum / total_elements

        # Stack losses into a single tensor
        total_losses = torch.stack([mspe_loss, calendar_arbitrage_loss, butterfly_arbitrage_loss])

        if testing_mode:
            loss_records = pd.DataFrame(loss_records)
            loss_records['Datetime'] = batch['Datetime']
            loss_records['Mask Proportion'] = batch['Mask Proportion']
            loss_records.set_index(['Datetime', 'Mask Proportion'], inplace=True)

            return total_losses, loss_records

        return total_losses, None

# surface_arbitrage_free_loss = SurfaceArbitrageFreeLoss()  
# all_losses, loss_records = surface_arbitrage_free_loss(tv_estimates_batch, batch)
# all_losses, loss_records
class AdaptiveLossWeights(torch.nn.Module):
    def __init__(
        self, 
        initial_losses, 
        alpha=1.0, 
        learning_rate=0.01
    ):
        """
        Initializes the adaptive loss weights module.

        Args:
            initial_losses (torch.Tensor): Initial loss values for each task to set the initial loss ratios.
            alpha (float): The strength of the restoring force in balancing training rates.
            learning_rate (float): Learning rate for updating the weights.
        """
        super(AdaptiveLossWeights, self).__init__()
        self.initial_losses = initial_losses
        self.alpha = alpha
        self.weights = torch.nn.Parameter(torch.ones_like(self.initial_losses))
        self.optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
        self.total_weights = self.weights.sum().item()  # Total of weights to maintain normalization

    def forward(
        self, 
        current_losses, 
        final_layer
    ):
        """
        Adjusts and normalizes the weights based on current losses using the GradNorm approach.

        Args:
            current_losses (torch.Tensor): Current computed losses from the main model.
            final_layer (torch.nn.Moduler): The final layer of the model whose parameters are used for 
            gradient norm calculation. 

        Returns:
            None: The updated weights are detached and stored within the module.
        """
        loss_ratios = current_losses / self.initial_losses
        relative_inverse_rates = loss_ratios / loss_ratios.mean()

        # Compute gradient norms for each weighted loss
        gradient_norms = torch.stack([
            torch.norm(torch.autograd.grad(self.weights[i] * loss, final_layer.parameters(), create_graph=True)[0])
            for i, loss in enumerate(current_losses)
        ])

        target_gradient_norms = (gradient_norms.mean() * (relative_inverse_rates ** self.alpha)).detach()
        gradnorm_loss = torch.sum(torch.abs(gradient_norms - target_gradient_norms))

        # Update the weights using the GradNorm loss
        self.optimizer.zero_grad()
        gradnorm_loss.backward()
        self.optimizer.step()

        # Normalize to sum to total_weights, detach, and ensure gradient tracking
        with torch.no_grad():
            normalized_weights = self.weights / self.weights.sum() * self.total_weights
            self.weights.data = normalized_weights.detach()  # Explicitly detach from the graph

        # Re-enable gradient tracking on the updated weights
        self.weights.requires_grad_()
def send_batch_to_device(batched_data, device):
    def move_to_device(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: move_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [move_to_device(item, device) for item in data]
        else:
            return data  # For non-tensor data (e.g., strings), return as is

    return move_to_device(batched_data, device)
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_

class Trainer:
    def __init__(
        self, 
        model, 
        train_data_loader, 
        validate_data_loader, 
        test_data_loader, 
        n_epochs, 
        warmup_ratio, 
        peak_learning_rate, 
        min_learning_rate, 
        gradient_clip, 
        adamw_betas, 
        adamw_epsilon, 
        adamw_weight_decay, 
        layer_wise_decay,
        loss_asymmetry_alpha, 
        device
    ):
        self.model = model.to(device)
        self.train_data_loader = train_data_loader
        self.validate_data_loader = validate_data_loader
        self.test_data_loader = test_data_loader
        self.n_epochs = n_epochs
        self.warmup_epochs = int(warmup_ratio * n_epochs)
        self.loss_asymmetry_alpha = loss_asymmetry_alpha
        self.gradient_clip = gradient_clip
        self.peak_learning_rate = peak_learning_rate
        self.min_learning_rate = min_learning_rate
        self.device = device

        # AdamW Optimizer with Layer-wise decay
        self.optimizer = AdamW(
            self._layer_wise_learning_rate_decay(layer_wise_decay, peak_learning_rate), 
            betas=adamw_betas, 
            eps=adamw_epsilon, 
            weight_decay=adamw_weight_decay
        )

        # Learning Rate Scheduler
        warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / self.warmup_epochs)
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.n_epochs - self.warmup_epochs,
            eta_min=self.min_learning_rate
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

    def train(self):
        self.model.train()
        adaptive_loss_weights = None
        loss_coefficients_history = []
        train_loss_components_history = []
        validate_loss_components_history = []

        for epoch in range(self.n_epochs):
            train_loss_components_sums = torch.zeros(3, device=self.device)  
            total_batches = 0

            for batch in self.train_data_loader:
                batch = send_batch_to_device(batch, self.device)
                tv_estimates_batch, _, _ = self.model(batch)
                train_loss_components, _ = SurfaceArbitrageFreeLoss()(tv_estimates_batch, batch)
                
                if adaptive_loss_weights is None: 
                    adaptive_loss_weights = AdaptiveLossWeights(
                        initial_losses=train_loss_components.detach().clone(),
                        alpha=self.loss_asymmetry_alpha,
                        learning_rate=np.sqrt(self.peak_learning_rate * self.min_learning_rate)
                    )

                # Obtain the current loss coefficients
                loss_coefficients = adaptive_loss_weights.weights.detach().clone()
                train_loss = train_loss_components @ loss_coefficients

                # Record the current loss coefficients
                loss_coefficients_history.append(loss_coefficients.cpu().numpy())

                # Accumulate the loss components
                train_loss_components_sums += train_loss_components.detach().clone()

                self.optimizer.zero_grad()
                train_loss.backward(retain_graph=True)

                adaptive_loss_weights(train_loss_components, self.model.final_layer)

                clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()
                total_batches += 1

                # Free up memory
                del batch, tv_estimates_batch, train_loss_components, loss_coefficients, train_loss
                torch.cuda.empty_cache()
                gc.collect()
                print(epoch)

            # Calculate the average loss components for this epoch
            avg_train_loss_components = train_loss_components_sums / total_batches
            train_loss_components_history.append(avg_train_loss_components.cpu().numpy())    
            
            # Validate after each epoch
            avg_validate_loss_components = self.validate()
            validate_loss_components_history.append(avg_validate_loss_components.cpu().numpy())

            print(f"Epoch {epoch + 1}/{self.n_epochs} - Training Loss: {avg_train_loss_components.cpu().numpy()}, Validation Loss: {avg_validate_loss_components.cpu().numpy()}")

            # Adjust learning rate
            self.scheduler.step()

        return loss_coefficients_history, train_loss_components_history, validate_loss_components_history


    def validate(self):
        self.model.eval()

        validate_loss_components_sums = torch.zeros(3, device=self.device)  
        total_batches = 0

        for batch in self.validate_data_loader:
            batch = send_batch_to_device(batch, self.device)
            tv_estimates_batch, _, _ = self.model(batch)

            validate_loss_components, _ = SurfaceArbitrageFreeLoss()(tv_estimates_batch, batch)
            validate_loss_components_sums += validate_loss_components.detach().clone()
            total_batches += 1

            # Free up memory
            del batch, tv_estimates_batch, validate_loss_components
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate the average loss components for this epoch
        avg_validate_loss_components = validate_loss_components_sums / total_batches  

        return avg_validate_loss_components    
    
    
    def test(
        self,
        output_attention_map=False
    ):
        self.model.eval()

        if output_attention_map:
            with torch.no_grad():
                batch = next(iter(self.test_data_loader))
                batch = send_batch_to_device(batch, self.device)
                tv_estimates_batch, self_attention_maps, external_attention_maps = ivy_spt(batch, output_attention_map=output_attention_map)

                # Free up memory
                del batch, tv_estimates_batch
                torch.cuda.empty_cache()
                gc.collect()

                return self_attention_maps, external_attention_maps

        test_loss_components_sums = torch.zeros(3, device=self.device)  
        total_batches = 0
        test_loss_records = []

        for batch in self.test_data_loader:
            batch = send_batch_to_device(batch, self.device)
            tv_estimates_batch, _, _ = self.model(batch)

            test_loss_components, loss_records = SurfaceArbitrageFreeLoss()(tv_estimates_batch, batch, testing_mode=True)
            test_loss_components_sums += test_loss_components.detach().clone()
            total_batches += 1
            test_loss_records.append(loss_records)

            # Free up memory
            del batch, tv_estimates_batch, test_loss_components, loss_records
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate the average loss components for this epoch
        avg_test_loss_components = test_loss_components_sums / total_batches  
        test_loss_records = pd.concat(test_loss_records) 

        return avg_test_loss_components, test_loss_records   
    
    def _layer_wise_learning_rate_decay(
        self, 
        layer_wise_decay, 
        base_lr
    ):
        params = []

        # Final layer (depth 0)
        params.append({
            'params': self.model.final_layer.parameters(),
            'lr': base_lr
        })

        # Surface Encoder layers (depth from 1 to num_encoder_blocks)
        if layer_wise_decay is not None:
            for i, encoder in enumerate(self.model.surface_encoder.encoders):
                lr = base_lr * (layer_wise_decay ** (i + 1))
                params.append({
                    'params': encoder.parameters(),
                    'lr': lr
                })

            # Surface Embedding layers (highest depth)
            params.append({
                'params': self.model.surface_embedding.parameters(),
                'lr': base_lr * (layer_wise_decay ** (len(self.model.surface_encoder.encoders) + 1))
            })
        else:
            # No decay: All layers use the base learning rate
            params.extend([
                {'params': self.model.surface_encoder.parameters(), 'lr': base_lr},
                {'params': self.model.surface_embedding.parameters(), 'lr': base_lr},
            ])

        return params

hyperparameters = {
    'Input Preprocessing' : {
        'Mask Proportions' : [0.1, 0.3, 0.5, 0.7],
        'Number of Query Points' : 1,
        'Batch Size' : 2
    },
    'Surface Embedding' : {
        'Embedding Dimension' : 8,
    },
    'Surface Encoding' : {
        'Number of Heads' : 4,
        'FFN Hidden Dimension' : 16,
        'Attention Dropout' : 0.1,
        'Gate Dropout' : 0.1,
        'FFN Dropout' : 0.1,
        'Number of Blocks' : 2,
        'External Feature Dimension' : 5,
    },
    'Adaptive Loss Weights' : {
        'Asymmetry' : 1.,
    },
    'Trainer' : {
        'Pre-Train' : {
            'Number of Epochs' : 10,
            'Warmup Ratio' : 0.15,
            'Peak Learning Rate' : 1e-3,
            'Minimal Learning Rate' : 1e-5,
            'Gradient Clipping' : 10,
            'Adam Betas' : (0.9, 0.999),
            'Adam Epsilon' : 1e-8,
            'Adam Weight Decay' : 0.05,
            'Layer-Wise Decay' : None,
        },
        'Fine-Tune' : {
            'Number of Epochs' : 10,
            'Warmup Ratio' : 0.1,
            'Peak Learning Rate' : 1e-3,
            'Minimal Learning Rate' : 1e-6,
            'Gradient Clipping' : 0,
            'Adam Betas' : (0.9, 0.999),
            'Adam Epsilon' : 1e-8,
            'Adam Weight Decay' : 0.05,
            'Layer-Wise Decay' : 0.9,
        }
    }
}
dataset = IVSurfaceDataset(
    surfaces, 
    hyperparameters['Input Preprocessing']['Mask Proportions'], 
    RANDOM_STATE, 
    hyperparameters['Input Preprocessing']['Number of Query Points'] 
)
data_loader = DataLoader(
    dataset, 
    batch_size=hyperparameters['Input Preprocessing']['Batch Size'], 
    shuffle=True, 
    num_workers=0, 
    collate_fn=IVSurfaceDataset.collate_fn
)
torch.manual_seed(RANDOM_STATE)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
model_pre_train = IvySPT(
    hyperparameters['Surface Embedding']['Embedding Dimension'], 
    hyperparameters['Surface Encoding']['Number of Blocks'],
    hyperparameters['Surface Encoding']['Number of Heads'], 
    hyperparameters['Surface Encoding']['FFN Hidden Dimension'],
    hyperparameters['Surface Encoding']['Attention Dropout'], 
    hyperparameters['Surface Encoding']['Gate Dropout'],
    hyperparameters['Surface Encoding']['FFN Dropout'],
    hyperparameters['Surface Encoding']['External Feature Dimension'],
)
pre_trainer = Trainer(
    model_pre_train,
    data_loader,
    data_loader,
    data_loader,
    hyperparameters['Trainer']['Pre-Train']['Number of Epochs'],
    hyperparameters['Trainer']['Pre-Train']['Warmup Ratio'],
    hyperparameters['Trainer']['Pre-Train']['Peak Learning Rate'],
    hyperparameters['Trainer']['Pre-Train']['Minimal Learning Rate'],
    hyperparameters['Trainer']['Pre-Train']['Gradient Clipping'],
    hyperparameters['Trainer']['Pre-Train']['Adam Betas'],
    hyperparameters['Trainer']['Pre-Train']['Adam Epsilon'],
    hyperparameters['Trainer']['Pre-Train']['Adam Weight Decay'],
    hyperparameters['Trainer']['Pre-Train']['Layer-Wise Decay'],
    hyperparameters['Adaptive Loss Weights']['Asymmetry'],
    device
)
pre_trainer.train()