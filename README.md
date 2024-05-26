---

# Transformer Model for Implied Volatility Surface Modelling

## Introduction

This project introduces an innovative transformer-based model for modeling implied volatility surfaces, uniquely incorporating key market features such as VIX, S&P returns, and asset returns. By employing cutting-edge techniques like conditional layer normalization and parametric continuous convolutional networks, the model dynamically adjusts its behavior to reflect current market conditions. This adaptive approach significantly enhances the model's robustness and accuracy, especially for less liquid options or those with sparse data. Pre-trained on high-liquidity options, this model exemplifies the power of transfer learning in financial modeling, allowing seamless fine-tuning for specific, less liquid options to deliver precise and reliable predictions.

## Model Pipeline

![image](https://github.com/daniel-norouzi-m/implied-volatility-surface-with-flow-based-generative-models/assets/108014662/060efec1-8ed4-4300-98c4-d08d03a073b1)


- **Input Embedding Section**:
  - **Surface Embedding**: Encodes the implied volatility surface into a fixed grid using parametric continuous convolutional filters dynamically generated based on market features.
  - **Pre Encoder Blocks**: Refines the grid embeddings with dynamically generated convolutional filters and Conditional Layer Normalization.

- **Surface Encoding**:
  - **Encoder Blocks**: Captures relationships within the encoded volatility surface using self-attention and feed-forward layers, conditioned by market features.

- **Query Embedding Section**:
  - **Query Embedding**: Processes the query point inputs, adding positional encodings to the embeddings.
  - **Pre Decoder Blocks**: Prepares the query point embeddings for the decoder by enhancing them with market features.

- **Surface Decoding**:
  - **Decoder Blocks**: Generates the output by attending to the encoded surface data and conditioned query points using cross-attention and Conditional Layer Normalization.
  - **Output Mapping**: Maps the decoder output to the target implied volatility value using a fully connected layer.

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
