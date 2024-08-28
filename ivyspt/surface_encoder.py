import torch
import torch.nn as nn

class ResidualNorm(nn.Module):
    def __init__(self, d_embedding):
        super(ResidualNorm, self).__init__()
        self.norm = nn.LayerNorm(d_embedding, elementwise_affine=True)

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
