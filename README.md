---
---
# Transformer Model for Implied Volatility Surface Modelling

### Introduction

This project aims to create a transformer-based model for modeling implied volatility surfaces, uniquely incorporating key market features such as VIX, S&P returns, and treasury rates, and ensuring adherence to no-arbitrage constraints. The project also employs transfer learning to enhance model performance and adaptability. Pre-trained on high-liquidity options, this model exemplifies the power of transfer learning in financial modeling, allowing seamless fine-tuning for specific, less liquid options to deliver precise and reliable predictions.

## Model Pipeline

#### Dataset

- **Dataset Creation**: Constructed from raw options trading data capturing key variables.
- **Clustering and Masking**: 
  - Clustering surfaces by datetime and symbol, segmenting data meaningfully.
  - Masking random subsets of points within clusters to simulate missing data.

#### Input Embedding

- **Surface Embedding Block**: 
  - Embeds surface values using Elliptical RBF Kernel.
  - Applies custom batch normalization for feature stability.
  - Projects to higher-dimensional space using 1x1 convolution and layer normalization.
  - Adds 2D positional encoding for embedding vectors.

#### Surface Encoding

- **Encoder Blocks**: 
  - Utilizes self-attention mechanisms, external attention with market features, and feed-forward networks.
  - Applies residual connections and layer normalization to refine surface embeddings.

#### Query Embedding

- **Point Embedding**: 
  - Generates embeddings for query points using learnable and positional encodings.
  - Refines embeddings through pre-decoder blocks with feed-forward networks and residual normalization.

#### Surface Decoding

- **Decoder Blocks**: 
  - Processes query embeddings and encoded surface data using cross-attention mechanisms and feed-forward networks.
  - Refines query embeddings for final prediction tasks.

#### Output Mapping

- **Fully Connected Layer**: Maps decoder output to target implied volatility values.

#### Surface Arbitrage Free Loss

- **Overview**: Ensures model predictions adhere to no-arbitrage conditions using soft constraints in the loss function.
- **Loss Components**:
  - **MSE Loss**: Measures the difference between model estimates and target volatilities.
  - **Calendar Arbitrage Condition**: Ensures total implied variance does not decrease with time.
  - **Butterfly Arbitrage Condition**: Prevents butterfly arbitrage in the volatility surface.
  - **Total Loss Calculation**: Combines MSE loss with arbitrage constraints, weighted by predefined coefficients. 

This approach ensures that the model predictions are not only accurate but also theoretically sound and robust against arbitrage opportunities.

![image](https://github.com/daniel-norouzi-m/implied-volatility-surface-with-flow-based-generative-models/assets/108014662/060efec1-8ed4-4300-98c4-d08d03a073b1)

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
Refines the grid embeddings using a multi-branch convolutional architecture inspired by Inception-A from Inception v3. This block is designed to capture complex feature interactions at various scales, followed by normalization and a learnable scaled residual connection to integrate the original input features effectively.

**Architectural Details**:
- **Branches**: The block consists of four distinct branches that process the input embeddings in parallel:
  1. **1x1 Convolution Branch**: Applies a straightforward 1x1 convolution to capture local features.
  2. **1x1 followed by 3x3 Convolution Branch**: Begins with a 1x1 convolution to reduce dimensionality, followed by a 3x3 convolution to capture spatial correlations.
  3. **1x1 followed by two 3x3 Convolutions (5x5 equivalent)**: Utilizes a 1x1 convolution for dimensionality reduction, followed by two successive 3x3 convolutions, effectively increasing the receptive field similar to a 5x5 convolution.
  4. **3x3 Max Pooling followed by 1x1 Convolution**: Applies a 3x3 max pooling to reduce spatial size and enhance feature extraction, followed by a 1x1 convolution to transform the pooled features.
- Each branch includes Batch Normalization and GELU activation functions to ensure stable and non-linear transformation of features.

**Concatenation and Reduction**:
- The outputs from all branches are concatenated along the channel dimension, combining diverse feature maps into a single tensor.
- A 1x1 convolution is then applied to reduce the concatenated features back to the original number of channels, ensuring that the output tensor matches the input dimensions for residual addition.

**Residual Connection and Normalization**:
- A learnable scaling parameter is introduced in the residual connection, allowing the model to adjust the influence of the input embeddings on the final output dynamically.
- After adding the scaled residual to the transformed features, a GELU activation function is applied to introduce non-linearity.
- Layer Normalization is performed across the channels, height, and width of the output tensor to normalize the features before passing to subsequent blocks.

**Mathematical Formulation**:
- The transformations in each branch can be summarized as follows:
  ```math
  \mathbf{B}_1 = \text{GELU}(\text{BatchNorm}(\text{Conv}_{1\times1}(\mathbf{X}))) \\
  \mathbf{B}_2 = \text{GELU}(\text{BatchNorm}(\text{Conv}_{3\times3}(\text{GELU}(\text{BatchNorm}(\text{Conv}_{1\times1}(\mathbf{X}))))))) \\
  \mathbf{B}_3 = \text{GELU}(\text{BatchNorm}(\text{Conv}_{3\times3}(\text{GELU}(\text{BatchNorm}(\text{Conv}_{3\times3}(\text{GELU}(\text{BatchNorm}(\text{Conv}_{1\times1}(\mathbf{X})))))))))) \\
  \mathbf{B}_4 = \text{GELU}(\text{BatchNorm}(\text{Conv}_{1\times1}(\text{MaxPool}_{3\times3}(\mathbf{X})))) \\
  ```
- Concatenation and reduction:
  ```math
  \mathbf{X}_{\text{concat}} = \text{Concat}(\mathbf{B}_1, \mathbf{B}_2, \mathbf{B}_3, \mathbf{B}_4) \\
  \mathbf{X}_{\text{reduced}} = \text{BatchNorm}(\text{Conv}_{1\times1}(\mathbf{X}_{\text{concat}})) \\
  ```
- Residual connection with learnable scale:
  ```math
  \mathbf{X}_{\text{output}} = \text{LayerNorm}(\text{GELU}(\mathbf{X} + \alpha \cdot \mathbf{X}_{\text{reduced}}))
  ```

### Surface Encoding

#### Overview

The Surface Encoding module is designed to process surface embeddings generated by the Input Embedding section, further refining these embeddings to capture complex relationships within the data. It utilizes a series of encoder blocks, each equipped with self-attention mechanisms, external attention with respect to market features, and feed-forward networks. This structure allows for deep integration of both intra-surface relationships and contextual market conditions into the surface representation.

#### Encoder Blocks

Each encoder block within the Surface Encoding module performs the following operations sequentially:

1. **Self-Attention**:
   - Captures dependencies between different positions in the surface embedding.
   - Mathematically represented as:
     ```math
     \text{SA}(X) = \text{softmax}\left(\frac{XQ (XK)^T}{\sqrt{d_k}}\right) XV
     ```
   - Where \(X\) is the input sequence, \(Q\), \(K\), and \(V\) are the query, key, and value projections of \(X\), respectively.

2. **Residual Connection and Layer Normalization**:
   - Applies a scaled residual connection followed by layer normalization:
     ```math
     X = \text{LayerNorm}(X + \alpha \cdot \text{SA}(X))
     ```
   - \(\alpha\) is a learnable scaling factor to modulate the contribution of the self-attention output.

3. **Feed-Forward Network**:
   - A two-layer network with GELU activation and dropout applied between layers:
     ```math
     \text{FFN}(X) = W_2 \cdot \text{GELU}(W_1X + b_1) + b_2
     ```

4. **External Attention**:
   - Incorporates external market features into the encoding process:
     ```math
     \text{EA}(X, M) = \text{softmax}\left(\frac{XQ (MK)^T}{\sqrt{d_k}}\right) MV
     ```
   - \(M\) denotes the external market features, treated as additional key and value inputs to the attention mechanism.

5. **Final Residual Connection and Layer Normalization**:
   - Similar to step 2, integrates the external attention output:
     ```math
     X = \text{LayerNorm}(X + \alpha \cdot \text{EA}(X, M))
     ```

#### Sequential Processing

- The `SurfaceEncoding` module initializes a specified number of these encoder blocks and processes the input through each, sequentially.
- The input to the module is a batch of tokenized, positional embedded surface data, alongside a batch of market features formatted as external features.
- The output is a sequence of encoded tokens enriched with contextual information drawn from both the surface data and external market features, making it ready for downstream tasks such as prediction or further analysis.


### Query Embedding

#### Overview

The Query Embedding module is responsible for generating embeddings for the query points, which are specific points on the volatility surface for which predictions are made. This module leverages the learnable embeddings and positional encodings to create a rich representation of each query point. Additionally, it refines these embeddings through a series of pre-decoder blocks.

#### Point Embedding

The Point Embedding component performs the following operations:

1. **Learnable Embedding**:
   - Each query point has a learnable embedding vector associated with it.
   - This vector is initialized randomly and optimized during the training process.

2. **Positional Encoding**:
   - Positional encodings are calculated based on the log moneyness and time to maturity of the query points with the same method and shared learnable parameter from the Surface Embedding section.

3. **Scaled Residual Norm**:
   - Combines the learnable embedding and positional encoding using a scaled residual norm:
     ```math
     \mathbf{Q}_{\text{final}} = \text{LayerNorm}(\mathbf{Q} + \alpha \cdot \text{PE}(\mathbf{q}_j))
     ```
   - Ensures the query embeddings are normalized and properly scaled.

#### Pre-Decoder Blocks

Each pre-decoder block refines the query embeddings further:

1. **Feed-Forward Network**:
   - A two-layer feed-forward network with GELU activation and dropout:
     ```math
     \text{FFN}(\mathbf{Q}) = W_2 \cdot \text{GELU}(W_1 \mathbf{Q} + b_1) + b_2
     ```

2. **Residual Connection and Layer Normalization**:
   - Applies a scaled residual connection followed by layer normalization:
     ```math
     \mathbf{Q} = \text{LayerNorm}(\mathbf{Q} + \alpha \cdot \text{FFN}(\mathbf{Q}))
     ```

#### Sequential Processing

- The `QueryEmbedding` module initializes the point embedding and a specified number of pre-decoder blocks.
- The input to the module is a batch of query points, which are processed to generate the initial embeddings.
- These embeddings are then sequentially passed through each pre-decoder block.
- The final output is a batch of refined query embeddings with the shape `(batch, 1, embedding)`, making them suitable for subsequent processing in the Surface Decoding module.


### Surface Decoding

#### Overview

The Surface Decoding module is designed to process the query embeddings generated by the Query Embedding module and the encoded surface data from the Surface Encoding module. It utilizes a series of decoder blocks, each equipped with cross-attention mechanisms and feed-forward networks, to refine the query embeddings based on the encoded surface data.

#### Decoder Blocks

Each decoder block within the Surface Decoding module performs the following operations sequentially:

1. **Cross-Attention**:
   - Captures dependencies between the query embeddings and the encoded surface data.
   - The query embeddings serve as the query (\(Q\)), while the encoded surface data serves as the keys (\(K\)) and values (\(V\)).
   - Mathematically represented as:
     ```math
     \text{CA}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
     ```

2. **Residual Connection and Layer Normalization**:
   - Applies a scaled residual connection followed by layer normalization:
     ```math
     Q = \text{LayerNorm}(Q + \alpha \cdot \text{CA}(Q, K, V))
     ```

3. **Feed-Forward Network**:
   - A two-layer feed-forward network with GELU activation and dropout:
     ```math
     \text{FFN}(Q) = W_2 \cdot \text{GELU}(W_1 Q + b_1) + b_2
     ```

4. **Final Residual Connection and Layer Normalization**:
   - Similar to the previous residual connection, applies scaled residual connection and layer normalization:
     ```math
     Q = \text{LayerNorm}(Q + \alpha \cdot \text{FFN}(Q))
     ```

#### Sequential Processing

- The `SurfaceDecoder` module initializes a specified number of decoder blocks and processes the input through each, sequentially.
- The input to the module includes the query embeddings and the encoded surface data.
- The output is a refined sequence of query embeddings enriched with contextual information from the encoded surface data, ready for final prediction tasks.

#### Output Mapping
Maps the decoder output to the target implied volatility value using a fully connected layer.

**Mathematical Formulation**:
```math
\text{IV}_{\text{pred}} = \text{FC}(\mathbf{Y}_{\text{final}})
```

### Surface Arbitrage Free Loss

#### Overview

The Surface Arbitrage Free Loss module is designed to ensure that the model's predictions of implied volatility surfaces adhere to no-arbitrage conditions. By incorporating these conditions as soft constraints in the loss function, the model minimizes the potential for generating arbitrage opportunities, thereby improving the reliability and robustness of its predictions. This module calculates the total loss, which includes the mean squared error (MSE) between the model's predictions and the actual target volatilities, along with the soft constraint losses for calendar and butterfly arbitrage conditions. Gradient calculations necessary for these arbitrage conditions are efficiently handled using Torch's automatic differentiation (autograd) capabilities.

#### Loss Components

1. **Mean Squared Error (MSE) Loss**:
   - Measures the difference between the model's implied volatility estimates and the actual target volatilities.
   - Formulated as:
     ```math
     \text{MSE Loss} = \frac{1}{N} \sum_{i=1}^{N} (\sigma_{\text{estimated}} - \sigma_{\text{target}})^2
     ```
   - Where \(\sigma_{\text{estimated}}\) is the implied volatility predicted by the model, and \(\sigma_{\text{target}}\) is the actual implied volatility from the dataset.

2. **Calendar Arbitrage Condition**:
   - Ensures that the total implied variance \(w(X, t) = t \cdot \sigma^2(X, t)\) does not decrease with respect to time to maturity.
   - Mathematically represented as:
     ```math
     L_{cal} = \left\| \max \left( 0, -\frac{\partial w}{\partial t} \right) \right\|^2
     ```

3. **Butterfly Arbitrage Condition**:
   - Ensures that the implied volatility surface does not exhibit butterfly arbitrage.
   - Defined as:
     ```math
     g(X, t) = \left( 1 - \frac{X w'}{2w} \right)^2 - \frac{w'}{4} \left( \frac{1}{w} + \frac{1}{4} \right) + \frac{w''}{2}
     ```
   - The butterfly arbitrage loss is then calculated as:
     ```math
     L_{but} = \left\| \max (0, -g) \right\|^2
     ```

4. **Total Loss Calculation**:
   - The total loss combines the MSE loss with the arbitrage constraints, weighted by predefined coefficients using \(\lambda_{cal}\) and \(\lambda_{but}\) for calendar and butterfly conditions, respectively.
   - Formulated as:
     ```math
     \text{Total Loss} = \text{MSE Loss} + \lambda_{cal} \cdot L_{cal} + \lambda_{but} \cdot L_{but}
     ```
   - These coefficients are configured to balance the influence of each component on the model's training process, ensuring both predictive accuracy and adherence to financial theory.

This comprehensive approach to loss calculation helps train models that not only fit the data well but also respect crucial financial principles, contributing to more robust and dependable predictions in practical applications.

## Architecture Search

This section explains how to find the optimal architecture without training by using rank scores of the NTK condition number and the Fisher-Rao norm at initialization.

#### NTK Condition Number

The Neural Tangent Kernel (NTK) condition number measures the stability of the network during training. At initialization, we compute the NTK \(\Theta\) and its condition number \(\kappa(\Theta)\):

```math
\Theta(x, x') = \nabla_\theta f(x; \theta)^\top \nabla_\theta f(x'; \theta)
```

The condition number is given by the ratio of the largest to the smallest eigenvalue of the NTK:

```math
\kappa(\Theta) = \frac{\lambda_{\max}(\Theta)}{\lambda_{\min}(\Theta)}
```

#### Fisher-Rao Norm

The Fisher-Rao norm measures the expressivity of the model by quantifying the sensitivity of the output distribution with respect to parameter changes:

```math
F(\theta) = \mathbb{E} \left[ \left( \frac{\partial \log p(X; \theta)}{\partial \theta} \right) \left( \frac{\partial \log p(X; \theta)}{\partial \theta} \right)^\top \right]
```

The Fisher-Rao norm is calculated as:

```math
\| \theta \|_{\text{FR}} = \sqrt{\theta^\top F(\theta) \theta}
```

#### Ranking and Selection

Architectures are ranked based on their NTK condition number and Fisher-Rao norm. The optimal architecture minimizes the NTK condition number and maximizes the Fisher-Rao norm:

```math
R_C(i) = \alpha R_{\kappa}(i) + (1 - \alpha) R_{\text{FR}}(i)
```

where \(R_{\kappa}(i)\) and \(R_{\text{FR}}(i)\) are the rankings based on the NTK condition number and Fisher-Rao norm, respectively, and \(\alpha\) is a weight balancing the two criteria.

## Evaluating the Trained Model

This section describes the evaluation of the trained model for robustness, sensitivity, trainability, and expressivity.

#### Gradients and Hessians

The gradients and Hessians of the loss function with respect to model parameters and input features (initial embedded surface grid and external features) are computed.

- **Gradient Norm**:
  ```math
  \|\nabla_{\theta} L\|_2 = \sqrt{\sum_{i=1}^p \left( \frac{\partial L}{\partial \theta_i} \right)^2}
  ```

- **Hessian Condition Number**:
  ```math
  H = \left[ \frac{\partial^2 L}{\partial \theta_i \partial \theta_j} \right]
  ```
  ```math
  \kappa(H) = \frac{\lambda_{\max}(H)}{\lambda_{\min}(H)}
  ```

#### NTK Condition Number and Trace

For the final model, compute the NTK and its condition number and trace:

```math
\Theta(x, x') = \nabla_\theta f(x; \theta)^\top \nabla_\theta f(x'; \theta)
```

- **Condition Number**:
  ```math
  \kappa(\Theta) = \frac{\lambda_{\max}(\Theta)}{\lambda_{\min}(\Theta)}
  ```

- **Trace**:
  ```math
  \text{Tr}(\Theta) = \sum_{i=1}^{n} \lambda_i(\Theta)
  ```

#### Fisher-Rao Norm

Compute the Fisher-Rao norm for the final model:

```math
\| \theta \|_{\text{FR}} = \sqrt{\theta^\top F(\theta) \theta}
```

## Component Analysis

Analyze each component of the trained model using gradients, Hessians, NTK, and Fisher-Rao norm.

#### Gradients and Hessians

- **Gradient Norm**:
  ```math
  \|\nabla_{\theta^l} L\|_2 = \sqrt{\sum_{i=1}^p \left( \frac{\partial L}{\partial \theta^l_i} \right)^2}
  ```

- **Hessian Condition Number**:
  ```math
  H^l = \left[ \frac{\partial^2 L}{\partial \theta^l_i \partial \theta^l_j} \right]
  ```
  ```math
  \kappa(H^l) = \frac{\lambda_{\max}(H^l)}{\lambda_{\min}(H^l)}
  ```

#### NTK Condition Number and Trace

For each component, compute the NTK and its condition number and trace:

```math
\Theta^l(x, x') = \nabla_{\theta^l} f(x; \theta^l)^\top \nabla_{\theta^l} f(x'; \theta^l)
```

- **Condition Number**:
  ```math
  \kappa(\Theta^l) = \frac{\lambda_{\max}(\Theta^l)}{\lambda_{\min}(\Theta^l)}
  ```

- **Trace**:
  ```math
  \text{Tr}(\Theta^l) = \sum_{i=1}^{n} \lambda_i(\Theta^l)
  ```

#### Fisher-Rao Norm

Compute the Fisher-Rao norm for each component:

```math
\| \theta^l \|_{\text{FR}} = \sqrt{\theta^l^\top F^l(\theta^l) \theta^l}
```

## Out of Sample and Stress Testing

Perform out-of-sample and stress testing by creating sequential blocks of data used as train and test sets, applying De Prado's embargo method to prevent leakage from autocorrelation. Report the out-of-sample results (MSE, MAE, calendar arbitrage, butterfly arbitrage) for the entire test set and for test blocks considered as stress times.


## Summary

This model integrates advanced transformer techniques with market-conditioned mechanisms to enhance modelling accuracy and robustness for implied volatility surfaces. The structured approach leverages transfer learning, making it adaptable to various market conditions and specific option datasets. This project showcases a novel application of transformers in financial modeling, offering significant contributions to the field.

---
---
