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

        return output


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

