---

# Transformer Model for Implied Volatility Surface Modelling

## Introduction

This project aims to create a transformer-based model for modeling implied volatility surfaces, uniquely incorporating key market features such as VIX, S&P returns, and asset returns., and ensuring adherence to no-arbitrage constraints. The model uses advanced techniques such as Conditional Layer Normalization (CLN), parametric continuous convolutional networks (PCCN), and a Gaussian Error Linear Unit (GELU) activation function. The project also employs transfer learning to enhance model performance and adaptability. Pre-trained on high-liquidity options, this model exemplifies the power of transfer learning in financial modeling, allowing seamless fine-tuning for specific, less liquid options to deliver precise and reliable predictions.

## Model Pipeline

![image](https://github.com/daniel-norouzi-m/implied-volatility-surface-with-flow-based-generative-models/assets/108014662/060efec1-8ed4-4300-98c4-d08d03a073b1)

#### Input Embedding Section

1. **Surface Embedding Block**:
   - **Purpose**: Encode the implied volatility surface into a fixed grid.
   - **Method**: Uses PCCN and 1x1 convolutions to embed the surface.
   - **Additional Steps**: Adds positional encodings based on each grid point's M and T values to incorporate the temporal and moneyness structure.

2. **Pre Encoder Blocks**:
   - **Purpose**: Refine the grid embeddings to prepare for the encoder blocks.
   - **Method**: Utilizes dynamically generated convolutional filters (conditioned on market features and grid point M and T values) and Conditional Layer Normalization (CLN) with residual connections.
   - **Structure**: Multiple blocks can be stacked to enhance the representation.

#### Surface Encoding Section

1. **Encoder Blocks**:
   - **Purpose**: Capture relationships within the encoded volatility surface.
   - **Method**: Uses conditional self-attention mechanisms and feed-forward layers, along with CLN and residual connections.
   - **Structure**: Multiple blocks can be stacked to deepen the model's capacity.

#### Query Embedding Section

1. **Query Embedding Block**:
   - **Purpose**: Process the query point input.
   - **Method**: Uses a learnable embedding for the query point (similar to the [MASK] token in NLP), adding positional encoding specific to the query point.

2. **Pre Decoder Blocks**:
   - **Purpose**: Enhance query point embeddings with market features.
   - **Method**: Similar to pre encoder blocks, but focused on query point embeddings.
   - **Structure**: Multiple blocks can be stacked for richer embedding refinement.

#### Surface Decoding

1. **Decoder Blocks**:
   - **Purpose**: Generate outputs by attending to encoded surface data and conditioned query points.
   - **Method**: Uses conditional cross-attention mechanisms and CLN with residual connections.
   - **Additional Features**: Stores cross-attention weights for later visual analysis using Gaussian kernel smoothing.
   - **Structure**: Multiple blocks can be stacked to increase the model's depth.

2. **Output Mapping Block**:
   - **Purpose**: Map decoder outputs to implied volatility values.
   - **Method**: Uses a fully connected layer to produce the final output.

#### Soft No-Arbitrage Constraints

1. **Calendar Spread Constraint**:
   - **Purpose**: Ensure that implied volatilities do not decrease with increasing maturity.
   - **Method**: Utilizes relevant formulas with derivatives calculated using autograd.

2. **Butterfly Spread Constraint**:
   - **Purpose**: Ensure convexity of the implied volatility surface.
   - **Method**: Utilizes relevant formulas with derivatives calculated using autograd.

#### Activation Function

- **GELU (Gaussian Error Linear Unit)**:
  - **Usage**: Applied throughout the model for non-linear transformations, providing smoother and more effective activation compared to ReLU.

#### Transfer Learning

- **Purpose**: Enhance model adaptability and performance.
- **Method**: Pre-train the model on high-liquidity options, then freeze and add new components to fine-tune with new data.

### Input Embedding Section

#### Surface Embedding Block 

##### Inputs
1. **Grid Points $\mathbf{x}_j = (M_j, T_j)$**: Reference points where the encoded values are to be computed, representing moneyness (M) and time to maturity (T).

##### Processing
1. **Parametric Continuous Convolution**:
   - Compute the encoded surface values $h_j$ using PCCN:

```math
h_{j} = \sum_{i=1}^{N} g(\mathbf{y}_i - \mathbf{x}_j; \theta) \cdot f_i
```
   
   Where $g$ is parameterized by an MLP, $\mathbf{y}_i$ are the input data points, and $f_i$ are the corresponding IV values.

2. **1x1 Convolution**:
   - Project the 1-channel encoded surface to a higher dimensional space using a 1x1 convolution:
```math
H = \text{Conv1x1}(h, \mathbf{W}_{1x1}, b)
```

3. **2D Positional Encoding**:
   - Add positional encoding to the output of the 1x1 convolution. The positional encoding for each dimension M and T is defined as follows for a dimension size $d_{\text{embedding}}$:

```math
PE(M_j, T_j, 2i) = \sin\left(\frac{M_j}{10000^{4i/d_{\text{embedding}}}}\right)
```
```math
PE(M_j, T_j, 2i+1) = \cos\left(\frac{M_j}{10000^{4i/d_{\text{embedding}}}}\right)
```
```math
PE(M_j, T_j, 2j+\frac{d_{\text{embedding}}}{2}) = \sin\left(\frac{T_j}{10000^{4j/d_{\text{embedding}}}}\right)
```
```math
PE(M_j, T_j, 2j+1+\frac{d_{\text{embedding}}}{2}) = \cos\left(\frac{T_j}{10000^{4j/d_{\text{embedding}}}}\right)
```
   Where $i, j$ are integers in the range $[0, d_{\text{embedding}}/4)$.

   - The full positional encoding $\mathbf{PE}(M_j, T_j)$ is concatenated or added to the embedding vector $H$ from the 1x1 convolution for each grid point:
```math
H_{\text{final}} = H + \mathbf{PE}(M_j, T_j)
```

4. **Layer Normalization**:
   - Apply layer normalization to the final embedding vector to ensure it is properly normalized for input into the subsequent transformer layers:
```math
H_{\text{norm}} = \text{LayerNorm}(H_{\text{final}})
```

##### Output
- **Encoded and Normalized Surface Embeddings $H_{\text{norm}}$**: These embeddings are now ready to be fed into the encoder blocks of the transformer.

#### Pre Encoder Blocks
Refines the grid embeddings with dynamically generated convolutional filters and Conditional Layer Normalization (CLN).

**Mathematical Formulation**:

First, apply the convolutional layer conditioned on market features:

```math
\mathbf{X}_{\text{conv}} = \text{ReLU}(\mathbf{W}_{\text{conv}}(z) * \mathbf{X}_{\text{grid}} + b)
```

Where $\mathbf{W}_{\text{conv}}(z)$ represents the convolutional filters conditioned on market features.

Then, apply Conditional Layer Normalization (CLN) based on the market features:

```math
\mathbf{Y} = \text{CLN}(\mathbf{X}_{\text{conv}}, z)
```

The CLN operation can be detailed as:

```math
\text{CLN}(\mathbf{X}_{\text{conv}}, z) = \gamma(z) \left(\frac{\mathbf{X}_{\text{conv}} - \mu}{\sigma}\right) + \beta(z)
```

Where $\mu$ and $\sigma$ are the mean and standard deviation of $\mathbf{X}_{\text{conv}}$, and $\gamma(z)$ and $\beta(z)$ are scale and shift parameters conditioned on market features, computed as:

```math
\gamma(z) = W_\gamma z + b_\gamma
```
```math
\beta(z) = W_\beta z + b_\beta
```

### Surface Encoding

#### Encoder Blocks
Captures relationships within the encoded volatility surface using self-attention and feed-forward layers, conditioned by market features.

**Self-Attention Mechanism**:
```math
\mathbf{Q}' = \mathbf{Q}
```
```math
\mathbf{K}' = W_k \mathbf{Z} + b_k
```
```math
\mathbf{V}' = W_v \mathbf{Z} + b_v
```
```math
\text{Attention}(Q', K', V') = \text{softmax}\left(\frac{Q' K'^T}{\sqrt{d_k}} + g(\mathbf{E}_z)\right) V'
```

**Layer Normalization**:
```math
\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{SelfAttention}(Q, K, V, \mathbf{E}_z))
```
```math
\mathbf{Y} = \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}'), \mathbf{E}_z)
```

### Query Embedding Section

#### Query Embedding
Processes the query point inputs, adding positional encodings to the learnable embeddings.

**Mathematical Formulation**:
```math
\mathbf{Q}_{\text{embed}} = \text{LearnableEmbedding}(M, T) + \text{PositionalEncoding}(M, T)
```

#### Pre Decoder Blocks
Prepares the query point embeddings for the decoder by enhancing them with market features.

**Mathematical Formulation**:
```math
\mathbf{Q}_{\text{processed}} = \text{CLN}(\text{ReLU}(\text{FC}(\mathbf{Q}_{\text{embed}})), \mathbf{E}_z)
```

### Surface Decoding

#### Decoder Blocks
Generates the output by attending to the encoded surface data and conditioned query points using cross-attention and Conditional Layer Normalization.

**Cross-Attention Mechanism**:
```math
\mathbf{Q}' = \mathbf{Q}_{\text{processed}}
```
```math
\mathbf{K}' = W_k \mathbf{X}_{\text{encoded}} + b_k
```
```math
\mathbf{V}' = W_v \mathbf{X}_{\text{encoded}} + b_v
```
```math
\text{Attention}(Q', K', V') = \text{softmax}\left(\frac{Q' K'^T}{\sqrt{d_k}} + g(\mathbf{E}_z)\right) V'
```

**Layer Normalization**:
```math
\mathbf{Y}' = \text{LayerNorm}(\mathbf{Y} + \text{CrossAttention}(Q, K, V, \mathbf{E}_z))
```
```math
\mathbf{Y}_{\text{final}} = \text{LayerNorm}(\mathbf{Y}' + \text{FFN}(\mathbf{Y}'), \mathbf{E}_z)
```

#### Output Mapping
Maps the decoder output to the target implied volatility value using a fully connected layer.

**Mathematical Formulation**:
```math
\text{IV}_{\text{pred}} = \text{FC}(\mathbf{Y}_{\text{final}})
```

## Summary

This model integrates advanced transformer techniques with market-conditioned mechanisms to enhance modelling accuracy and robustness for implied volatility surfaces. The structured approach leverages transfer learning, making it adaptable to various market conditions and specific option datasets. This project showcases a novel application of transformers in financial modeling, offering significant contributions to the field.

---
