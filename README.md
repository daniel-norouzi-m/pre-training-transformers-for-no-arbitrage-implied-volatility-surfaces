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
  - Embeds surface values using RBF Kernel.
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

### Dataset

#### Dataset Creation

The dataset is systematically constructed from raw options trading data, which captures key variables such as log moneyness, time to maturity, implied volatility, and relevant market features.

##### Dynamic Clustering and Masking

To simulate real-world scenarios where all data points might not be available, dynamic clustering and masking are performed during batch creation instead of preprocessing:

1. **Clustering**:
   - Surfaces are divided based on unique datetime and symbol combinations, followed by clustering the points within each surface during batch creation. This helps to segment the data into meaningful groups, reflecting potential market segmentations.
   - Mathematical formulation:
     ```math
     \text{labels} = \text{KMeans}(n_{\text{clusters}}, \text{random\_state}).\text{fit\_predict}(\text{data points})
     ```

2. **Dynamic Masking**:
   - Within each cluster, a random subset of points is masked dynamically every time a surface is fed to the model. This prepares the model to infer missing information effectively.
   - Proportional masking varies to challenge the model under different scenarios:
     ```math
     \text{masked indices} = \text{random choice}(\text{cluster indices}, \text{mask proportion})
     ```

##### Proportional Sampling

Adjusting the proportion of masked data across training epochs allows the model to adapt to various levels of data availability, enhancing its robustness and predictive capabilities. This dynamic approach ensures that the model consistently encounters new patterns of missing data, promoting greater generalization and adaptability.


### Surface Embedding Section

#### Surface Embedding Section 

##### Inputs
1. **Surface Points $\mathbf{x}_j = (M_j, T_j)$**: Available surface points with coordinates representing moneyness (M) and time to maturity (T) and their corresponding implied volatilities (IV).

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
Compute the embedded surface values using multiple Elliptical RBF Kernels with different fixed bandwidths: The embedding for each grid point is calculated by summing the products of the kernel evaluations and the implied volatilities for each input data point across multiple elliptical RBF kernels, resulting in a multi-channel representation.

**Fixed Bandwidth Calculation**: Based on $d$ (d_embedding), we set a range for the Gaussian PDF integral: from $\frac{1}{d+1}$ to $\frac{d}{d+1}$. This gives us $d$ values for the integral solutions $i$. The bandwidth value of the RBF kernel is then set using the formula $\text{CDF}^{-1} \left( \frac{i + 1}{2} \right)$. Mathematically, for each RBF kernel, where $k_{k}$ is the k-th RBF kernel with a fixed bandwidth, $\mathbf{y}_i$ and $\mathbf{x}_j$ represent input surface points, and $f_i$ are the corresponding implied volatility values. Here, $\mathbf{d} = \mathbf{y}_i - \mathbf{x}_j$ is the vector of differences between input data points, and $\sigma_{k}$ is the fixed bandwidth for the k-th RBF kernel. Additionally, we introduce a learnable scale vector $\mathbf{s}$, initialized to 1, which serves as the diagonal scale matrix $A$ to calculate the distance $((\mathbf{y}_i - \mathbf{x}_j)^T A (\mathbf{y}_i - \mathbf{x}_j))$.

For embedding calculation, the embedding at each point where a surface point is present is the average volatility value weighted by the kernel outputs. The same holds for the query points but their volatility values are not considered. This means we should divide the sum by the sum of the kernel outputs, ensuring that the sum is taken over the input surface points (excluding the query points), but each grid point can still be a query point:

```math
h_{j}^{(k)} = \frac{\sum_{i=1}^{N} k_{k}(\mathbf{y}_i, \mathbf{x}_j) \cdot f_i}{\sum_{i=1}^{N} k_{k}(\mathbf{y}_i, \mathbf{x}_j)}
```
     
```math
k_{k}(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{(\mathbf{x} - \mathbf{y})^T \mathbf{A} (\mathbf{x} - \mathbf{y})}{2 \cdot \text{trace}(\mathbf{A})}\right)
```

**Embedding Calculation**:
The embedding for each surface point is calculated by averaging the implied volatilities of all input surface points, weighted by their 2D distances using multiple elliptical RBF kernels with different bandwidths. This process is also applied to the masked points, except their IV values are masked out and not considered in the calculation. This results in a multi-channel representation where each channel corresponds to the embedded value computed using a different RBF kernel, providing a localized interpretation of the input surface.


Here is the revised README section based on your new implementation details:

---

**Normalization**:
After computing the embeddings using multiple Elliptical RBF kernels, layer normalization is applied across the embedding dimension to ensure stability and consistency in the feature distributions:

```math
H = \text{LayerNorm}(H_{\text{multi-channel}})
```

2. **2D Positional Encoding**:
   - Positional encoding is added to the output of the Elliptical RBF kernel embeddings. The positional encoding for each dimension $M$ and $T$ is defined as follows for a dimension size $d_{\text{embedding}}$:

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

   - The full positional encoding $\mathbf{PE}(M_j, T_j)$ is added to the embedding vector $H$ from the Elliptical RBF kernel embeddings with a constant factor $\sqrt{2}$ to account for the standard deviation of the sinusoidal functions:
```math
H_{\text{final}} = \text{LayerNorm}(H + \sqrt{2} \mathbf{PE}(M_j, T_j))
```

3. **Mask Token**:
   - A learnable mask token is added to the query points before layer norm to handle any masked values.

##### Output
- **Encoded and Normalized Surface Embeddings $H_{\text{final}}$**: These embeddings are now ready to be fed into the encoder blocks of the transformer.

### Surface Encoding

#### Overview

The Surface Encoding module processes surface embeddings generated by the Surface Embedding section, refining these embeddings to capture complex relationships within the data. It utilizes a series of encoder blocks, each equipped with self-attention mechanisms, external attention with respect to market features, and feed-forward networks. This structure allows for deep integration of both intra-surface relationships and contextual market conditions into the surface representation.

#### Encoder Blocks

Each encoder block within the Surface Encoding module performs the following operations sequentially:

1. **Self-Attention**:
   - Captures dependencies between different positions in the surface embedding.
   - Mathematically represented as:
   ```math
   \text{SA}(X) = \text{softmax}\left(\frac{XQ (XK)^T}{\sqrt{d_k}}\right) XV
   ```
   - Where $X$ is the input sequence, $Q$, $K$, and $V$ are the query, key, and value projections of $X$, respectively.

2. **Cross-Attention**:
   - Incorporates external market features into the encoding process:
   ```math
   \text{EA}(X, M) = \text{softmax}\left(\frac{XQ (MK)^T}{\sqrt{d_k}}\right) MV
   ```
   - $M$ denotes the external market features, treated as additional key and value inputs to the attention mechanism.

3. **Gated Attention Fusion**:
   - Combines the outputs of self-attention and cross-attention using a gating mechanism.
   - The attention outputs are concatenated and passed through a linear layer followed by a sigmoid activation to compute the gating values:
   ```math
   \text{Gate}(X) = \sigma\left(W_g \cdot \left[\text{SA}(X); \text{EA}(X, M)\right] + b_g\right)
   ```

   - The final gated embedding is calculated as a weighted average of the self-attention and cross-attention outputs:
   ```math
   \text{Gated Embedding}(X) = \text{Gate}(X) \cdot \text{SA}(X) + (1 - \text{Gate}(X)) \cdot \text{EA}(X, M)
   ```

4. **Residual Connection and Layer Normalization**:
   - Applies a residual connection followed by layer normalization to the gated embedding:
   ```math
   X = \text{LayerNorm}(X + \text{Gated Embedding}(X))
   ```

5. **Feed-Forward Network**:
   - A two-layer network with GELU activation and dropout applied between layers:
   ```math
   \text{FFN}(X) = W_2 \cdot \text{GELU}(W_1X + b_1) + b_2
   ```

6. **Final Residual Connection and Layer Normalization**:
   - Similar to step 4, integrates the feed-forward network output:
   ```math
   X = \text{LayerNorm}(X + \text{FFN}(X))
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
   - The query embeddings serve as the query ($Q$), while the encoded surface data serves as the keys ($K$) and values ($V$).
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
   - Where $\sigma_{\text{estimated}}$ is the implied volatility predicted by the model, and $\sigma_{\text{target}}$ is the actual implied volatility from the dataset.

2. **Calendar Arbitrage Condition**:
   - Ensures that the total implied variance $w(X, t) = t \cdot \sigma^2(X, t)$ does not decrease with respect to time to maturity.
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
   - The total loss combines the MSE loss with the arbitrage constraints, weighted by predefined coefficients using $\lambda_{cal}$ and $\lambda_{but}$ for calendar and butterfly conditions, respectively.
   - Formulated as:
```math
\text{Total Loss} = \text{MSE Loss} + \lambda_{cal} \cdot L_{cal} + \lambda_{but} \cdot L_{but}
```
   - These coefficients are configured to balance the influence of each component on the model's training process, ensuring both predictive accuracy and adherence to financial theory.

This comprehensive approach to loss calculation helps train models that not only fit the data well but also respect crucial financial principles, contributing to more robust and dependable predictions in practical applications.

## Architecture Search

This section explains how to find the optimal architecture without training by using rank scores of the NTK condition number and the Fisher-Rao norm at initialization.

#### NTK Condition Number

The Neural Tangent Kernel (NTK) condition number measures the stability of the network during training. At initialization, we compute the NTK $\Theta$ and its condition number $\kappa(\Theta)$:

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

where $R_{\kappa}(i)$ and $R_{\text{FR}}(i)$ are the rankings based on the NTK condition number and Fisher-Rao norm, respectively, and $\alpha$ is a weight balancing the two criteria.

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
\| \theta^l \|_{\text{FR}} = \sqrt{{\theta^l}^\top F^l(\theta^l) \theta^l} 
```

## Out of Sample and Stress Testing

Perform out-of-sample and stress testing by creating sequential blocks of data used as train and test sets, applying De Prado's embargo method to prevent leakage from autocorrelation. Report the out-of-sample results (MSE, MAE, calendar arbitrage, butterfly arbitrage) for the entire test set and for test blocks considered as stress times.


## Summary

This model integrates advanced transformer techniques with market-conditioned mechanisms to enhance modelling accuracy and robustness for implied volatility surfaces. The structured approach leverages transfer learning, making it adaptable to various market conditions and specific option datasets. This project showcases a novel application of transformers in financial modeling, offering significant contributions to the field.

---
---
