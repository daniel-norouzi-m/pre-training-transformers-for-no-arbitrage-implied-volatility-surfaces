---
---
# Transformer Model for Implied Volatility Surface Modelling

## Introduction

This project aims to create a transformer-based model for modeling implied volatility surfaces, uniquely incorporating key market features such as VIX, S&P returns, and asset returns., and ensuring adherence to no-arbitrage constraints. The model uses advanced techniques such as Conditional Layer Normalization (CLN), parametric continuous convolutional networks (PCCN), and a Gaussian Error Linear Unit (GELU) activation function. The project also employs transfer learning to enhance model performance and adaptability. Pre-trained on high-liquidity options, this model exemplifies the power of transfer learning in financial modeling, allowing seamless fine-tuning for specific, less liquid options to deliver precise and reliable predictions.

## Model Pipeline

![image](https://github.com/daniel-norouzi-m/implied-volatility-surface-with-flow-based-generative-models/assets/108014662/060efec1-8ed4-4300-98c4-d08d03a073b1)

#### Surface Preprocessing

1. **Dataset Creation**:
   - **Purpose**: Construct a dataset from raw options data that includes varying market conditions and data completeness.
   - **Method**: Features such as log moneyness, time to maturity, and implied volatility are used along with market features. Data points are clustered and selectively masked to simulate incomplete data scenarios, training the model to predict missing information.
   - **Proportional Sampling**: Implements variable masking proportions to simulate different levels of data availability, enhancing the model's robustness and ability to generalize.

2. **Custom Batch Normalization**:
   - **Purpose**: Normalize features across the dataset to stabilize neural network training.
   - **Method**: Custom batch normalization layers are used to standardize `Input Surface` and `Query Point` features, while separate normalization layers adjust `Market Features`.
   - **Data Integrity**: Ensures non-numeric data like `Datetime` and `Symbol` are preserved unchanged, maintaining essential information for modeling.


#### Input Embedding Section

1. **Surface Embedding Block**:
   - **Purpose**: Encode the implied volatility surface into a fixed grid.
   - **Method**: Uses Continuous Kernel Embedding and 1x1 convolutions to embed the surface.
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

### Dataset

#### Dataset Creation

The dataset is systematically constructed from raw options trading data, which captures key variables such as log moneyness, time to maturity, implied volatility, and relevant market features.

##### Clustering and Masking

To simulate real-world scenarios where all data points might not be available:

1. **Clustering**:
   - Surfaces are divided based on unique datetime and symbol combinations, followed by clustering the points within each surface. This helps to segment the data into meaningful groups, reflecting potential market segmentations.
   - Mathematical formulation:
     ```math
     \text{labels} = \text{KMeans}(n_{\text{clusters}}, \text{random\_state}).\text{fit\_predict}(\text{data points})
     ```

2. **Masking**:
   - Within each cluster, a random subset of points is masked, simulating missing data. This prepares the model to infer missing information effectively.
   - Proportional masking varies to challenge the model under different scenarios:
     ```math
     \text{masked indices} = \text{random choice}(\text{cluster indices}, \text{mask proportion})
     ```

##### Proportional Sampling

Adjusting the proportion of masked data across training instances allows the model to adapt to various levels of data availability, enhancing its robustness and predictive capabilities.

### Input Embedding Section

#### Surface Embedding Block 

##### Inputs
1. **Grid Points $\mathbf{x}_j = (M_j, T_j)$**: Reference points where the encoded values are to be computed, representing moneyness (M) and time to maturity (T).

#### Custom Batch Normalization

To ensure that the model inputs are normalized and stable for training:

1. **Feature Normalization**:
   - Each feature, including log moneyness, time to maturity, and implied volatility, is normalized across the batch to ensure zero mean and unit variance, essential for effective model training.
   - Batch normalization is applied separately to the concatenated features of `Input Surface` and `Query Point`:
     ```math
     \text{norm\_feature} = \text{BatchNorm}(\text{concatenated feature})
     ```

2. **Market Features Normalization**:
   - Market features such as return, volatility, and treasury rates are also normalized using batch normalization, catering to their dynamic ranges and distributions:
     ```math
     \text{norm\_market\_feature} = \text{BatchNorm1d}(\text{market feature})
     ```

3. **Preservation of Data Integrity**:
   - Non-numeric data such as `Datetime` and `Symbol` remain unchanged to preserve essential indexing and categorical information, ensuring the contextual relevance of the model’s output.

This preprocessing step ensures the dataset is not only tailored for effective learning but also mirrors the practical challenges and conditions of financial markets, preparing the model for real-world deployment.

##### Processing

1. **Surface Continuous Kernel Embedding**:
   Compute the embedded surface values using the Elliptical RBF Kernel:

```math
h_{j} = \sum_{i=1}^{N} k(\mathbf{y}_i - \mathbf{x}_j; \sigma) \cdot f_i
```

   Where $k$ is an elliptical radial basis function kernel, $\mathbf{y}_i$ are the input data points, $\mathbf{x}_j$ represents points on a transformed grid, and $f_i$ are the corresponding implied volatility values. The kernel $k$ is defined as:

```math
k(\mathbf{d}; \sigma) = \exp\left(-\frac{1}{2} \sum_{d=1}^{D} \left(\frac{d_d}{\sigma_d}\right)^2\right)
```

   Here, $\mathbf{d} = \mathbf{y}_i - \mathbf{x}_j$ is the vector of differences between input data points and grid points, $D$ is the dimension of the input data (e.g., Log Moneyness, Time to Maturity), and $\sigma$ contains the learnable bandwidth parameters for each dimension, enhancing the flexibility to adapt to different scales of data features.

   **Grid Point Transformation**:
     The grid points $\mathbf{x}_j$ are created to span a regular grid within the normalized feature space. The transformation of these grid points from a uniform distribution to a more natural distribution tailored to the characteristics of financial data is performed using the inverse cumulative distribution function (CDF) of the normal distribution:

```math
\mathbf{x}_j = \text{erfinv}(2 \mathbf{u}_j - 1) \sqrt{2}
```

   Where $\mathbf{u}_j$ are uniformly distributed points on the interval (0, 1) excluding the endpoints, transformed to follow the distribution of the input features more closely.

   **Embedding Calculation**:
     The embedding for each grid point $\mathbf{x}_j$ is calculated by summing the products of the kernel evaluations and the implied volatilities for each input data point $\mathbf{y}_i$. This results in a 2D grid of embedded values, each representing a localized interpretation of the input surface, shaped by the kernel's response to the distance between grid points and data points. The final output is a 2D image-like representation where each pixel corresponds to the embedded value at a specific grid location.


2. **Surface Projection Embedding with 1x1 Convolution and Layer Normalization**:
   - Project the 1-channel encoded surface to a higher dimensional space using a 1x1 convolution and apply layer normalization to the embedding vector to ensure it is properly normalized:
```math
H = \text{LayerNorm}(\text{Conv1x1}(h, \mathbf{W}_{1x1}, b))
```

3. **2D Positional Encoding**:
   - Add positional encoding to the output of the 1x1 convolution. The positional encoding for each dimension $M$ and $T$ is defined as follows for a dimension size $d_{\text{embedding}}$:

```math
PE(M_j, T_j, 4i) = \sin\left(\frac{M_j}{\sigma_{\text{scale}}^{4i/d_{\text{embedding}}}}\right)
```
```math
PE(M_j, T_j, 4i+1) = \cos\left(\frac{M_j}{\sigma_{\text{scale}}^{4i/d_{\text{embedding}}}}\right)
```
```math
PE(M_j, T_j, 4i+2) = \sin\left(\frac{T_j}{\sigma_{\text{scale}}^{4i/d_{\text{embedding}}}}\right)
```
```math
PE(M_j, T_j, 4i+3) = \cos\left(\frac{T_j}{\sigma_{\text{scale}}^{4i/d_{\text{embedding}}}}\right)
```
   Where $i$ is an integer in the range $[0, d_{\text{embedding}}/4)$, and $\sigma_{\text{scale}}$ is a learnable parameter initially set to 10000 but adjustable during training.

   - The full positional encoding $\mathbf{PE}(M_j, T_j)$ multiplied by a learnable factor of $\alpha$ initialized to 1, is added to the embedding vector $H$ from the 1x1 convolution for each grid point and finally sent through a layer normalization:
```math
H_{\text{final}} = \text{LayerNorm}(H + \alpha \mathbf{PE}(M_j, T_j))
```

##### Output
- **Encoded and Normalized Surface Embeddings $H_{\text{final}}$**: These embeddings are now ready to be fed into the encoder blocks of the transformer.

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
---
