---

# Transformer Model for Implied Volatility Surface Modelling

## Introduction

This project aims to create a transformer-based model for modeling implied volatility surfaces, uniquely incorporating key market features such as VIX, S&P returns, and asset returns., and ensuring adherence to no-arbitrage constraints. The model uses advanced techniques such as Conditional Layer Normalization (CLN), parametric continuous convolutional networks (PCCN), and a Gaussian Error Linear Unit (GELU) activation function. The project also employs transfer learning to enhance model performance and adaptability. Pre-trained on high-liquidity options, this model exemplifies the power of transfer learning in financial modeling, allowing seamless fine-tuning for specific, less liquid options to deliver precise and reliable predictions.

## Model Pipeline

![image](https://github.com/daniel-norouzi-m/implied-volatility-surface-with-flow-based-generative-models/assets/108014662/060efec1-8ed4-4300-98c4-d08d03a073b1)

#### Input Embedding Section

1. **Surface Embedding**:
   - **Purpose**: Encode the implied volatility surface into a fixed grid.
   - **Method**: Uses PCCN conditioned on market features (like VIX, S&P returns, and asset returns) and grid point M and T values.
   - **Additional Steps**: Adds positional encodings based on each grid point's M and T values to incorporate the temporal and moneyness structure.

2. **Pre Encoder Blocks**:
   - **Purpose**: Refine the grid embeddings to prepare for the encoder blocks.
   - **Method**: Utilizes dynamically generated convolutional filters (conditioned on market features and grid point M and T values) and Conditional Layer Normalization (CLN) with residual connections.
   - **Structure**: Multiple blocks can be stacked to enhance the representation.

#### Surface Encoding

1. **Encoder Blocks**:
   - **Purpose**: Capture relationships within the encoded volatility surface.
   - **Method**: Uses conditional self-attention mechanisms and feed-forward layers, along with CLN and residual connections.
   - **Structure**: Multiple blocks can be stacked to deepen the model's capacity.

#### Query Embedding Section

1. **Query Embedding**:
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

2. **Output Mapping**:
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

#### Surface Embedding
Encodes the implied volatility surface into a fixed grid using parametric continuous convolutional filters dynamically generated based on market features.

**Mathematical Formulation**:

Let $\mathbf{X}$ be the input surface data and $z$ be the market features. The embedding is computed as:

```math
\mathbf{X}_{\text{grid}} = \text{PCCN}(\mathbf{X}, z)
```

Where PCCN is the Parametric Continuous Convolutional Network conditioned on market features.

The PCCN dynamically generates convolutional filters based on market features and the position values $M$ and $T$:

```math
\mathbf{W}_{\text{PCCN}} = f(z, M, T)
```

The convolutional operation within the PCCN can be formulated as:

```math
\mathbf{X}_{\text{grid}, i, j} = \sum_{k,l} \mathbf{W}_{\text{PCCN}, i-k, j-l} \cdot \mathbf{X}_{k, l} + b
```

Where $\mathbf{W}_{\text{PCCN}}$ are the dynamically generated filters and $b$ is the bias term.

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
