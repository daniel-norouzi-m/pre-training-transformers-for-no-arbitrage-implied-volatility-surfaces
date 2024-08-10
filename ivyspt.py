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
