---
---
# A Surface is Worth Hundreds of Options: Pre-Training Transformers for No-Arbitrage Implied Volatility Surfaces

## Abstract
Accurate modeling of implied volatility surfaces is crucial for options pricing and risk management. This paper introduces IvySPT, a transformer-based architecture designed to integrate key market features—such as VIX levels, SP 500 returns, and treasury rates—into estimating these surfaces while providing interpretable insights into the complex relationships between market conditions and volatility estimates. Our approach leverages dynamic data masking and a dual-phase training strategy: the model is pre-trained on high liquidity options and subsequently fine-tuned to adapt to sparse volatility surfaces. The model's robustness is further enhanced by a dynamic masking strategy that simulates real-world scenarios, improving generalization and inference of missing information under varying data availability conditions. To ensure the financial soundness of the estimations, no-arbitrage conditions are embedded as soft constraints within the loss function. IvySPT not only sets new standards in the estimation accuracy of financial models but also provides a comprehensive framework for understanding the intricate dynamics of market conditions on volatility estimates.

## Introduction

The IvySPT project introduces an innovative transformer-based model designed to accurately model implied volatility surfaces, integrating essential market features and adhering to rigorous financial constraints. This model strategically incorporates market dynamics through features such as VIX levels, S&P returns, and treasury rates, alongside the mean and standard deviation of surface implied volatility values, providing a comprehensive understanding of the market conditions influencing volatility estimates.

Central to our approach is the utilization of transfer learning, which significantly enhances the model's performance and adaptability. Initially pre-trained on options with high liquidity, the model leverages learned patterns to facilitate rapid and efficient fine-tuning on specific, less liquid options. This dual-phase training approach—consisting of a robust pre-training phase followed by targeted fine-tuning—ensures that the model can provide precise and reliable predictions across a wide range of market scenarios.

Additionally, the model's architecture is carefully designed to ensure compliance with no-arbitrage conditions, integrating these constraints as soft penalties within the loss function. This not only reinforces the financial validity of the model's predictions but also instills confidence in its use for practical trading and risk management applications. By embracing a sophisticated blend of machine learning techniques and deep financial insights, the IvySPT project sets a new standard for quantitative modeling in finance, aiming to deliver unmatched accuracy and insight into the complex dynamics of implied volatility surfaces.


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

Adjusting the proportion of masked data randomly across training iterations allows the model to adapt to various levels of data availability, enhancing its robustness and predictive capabilities. This dynamic approach ensures that the model consistently encounters new patterns of missing data, promoting greater generalization and adaptability.


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
   - Market features (S&P returns, VIX, treasury rates, mean of surface IVs, and std. of surface IVs) are also normalized using batch normalization, catering to their dynamic ranges and distributions:
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
     
$$\text{SA}(X) = \text{softmax}\left(\frac{XQ (XK)^T}{\sqrt{d_k}}\right) XV$$
   
   - Where $X$ is the input sequence, $Q$, $K$, and $V$ are the query, key, and value projections of $X$, respectively.
     
   - Applies a residual connection followed by layer normalization to the gated embedding:

$$\text{SA}(X) = \text{LayerNorm}(X + \text{SA}(X))$$

3. **Cross-Attention**:
   - Incorporates external market features (S&P returns, VIX, treasury rates, mean of surface IVs, std. of surface IVs) into the encoding process:

$$\text{EA}(X, M) = \text{softmax}\left(\frac{XQ (MK)^T}{\sqrt{d_k}}\right) MV$$

   - $M$ denotes the external market features, treated as additional key and value inputs to the attention mechanism.
   - Applies a residual connection followed by layer normalization to the gated embedding:

$$\text{EA}(X, M) = \text{LayerNorm}(X + \text{EA}(X, M))$$

4. **Gated Attention Fusion**:
   - Combines the outputs of self-attention and cross-attention using a gating mechanism.
   - The attention outputs are concatenated and passed through a linear layer followed by a sigmoid activation to compute the gating values:

$$\text{Gate}(X) = \sigma\left(W_g \cdot \left[\text{SA}(X); \text{EA}(X, M)\right] + b_g\right)$$

   - The final gated embedding is calculated as a weighted average of the self-attention and cross-attention outputs:

$$\text{Gated Embedding}(X) = \text{Gate}(X) \cdot \text{SA}(X) + (1 - \text{Gate}(X)) \cdot \text{EA}(X, M)$$

   - Applies a residual connection followed by layer normalization to the gated embedding:

$$X = \text{LayerNorm}(X + \text{Gated Embedding}(X))$$

5. **Feed-Forward Network**:
   - A two-layer network with GELU activation and dropout applied between layers:

$$\text{FFN}(X) = W_2 \cdot \text{GELU}(W_1X + b_1) + b_2$$

   - Integrates the feed-forward network output:\
     
$$X = \text{LayerNorm}(X + \text{FFN}(X))$$

#### Sequential Processing

- The `SurfaceEncoding` module initializes a specified number of these encoder blocks and processes the input through each, sequentially.
- The input to the module is a batch of tokenized, positional embedded surface data, alongside a batch of market features formatted as external features.
- The output is a sequence of encoded tokens enriched with contextual information drawn from both the surface data and external market features, making it ready for downstream tasks such as prediction or further analysis.

### Model Initialization

The IvySPT model parameters are initialized using the following strategy:

1. **Random Initialization**:
   - All weights are randomly initialized from a normal distribution $N(0, 0.02)$ and biases are zero.

2. **Transformer Layer Rescaling**:
   - For the $l$-th Transformer layer, the output matrices (i.e., the last linear projection within each sub-layer) of the self-attention module and the feed-forward network are rescaled by $\frac{1}{\sqrt{2l}}$. This ensures stable gradient flow and improves convergence.

3. **Gate Bias Initialization**:
   - The gate bias in the gated attention fusion mechanism is initialized to a high value (e.g., 10). This initialization biases the model to initially ignore external features, allowing it to focus on learning the primary self-attention relationships before integrating external information.


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
\text{Total Loss} = \lambda_{mse} \text{MSE Loss} + \lambda_{cal} \cdot L_{cal} + \lambda_{but} \cdot L_{but}
```
   - These coefficients are configured to balance the influence of each component on the model's training process, ensuring both predictive accuracy and adherence to financial theory.

This comprehensive approach to loss calculation helps train models that not only fit the data well but also respect crucial financial principles, contributing to more robust and dependable predictions in practical applications.


## Empirical Study

### Data Splitting and Masking Strategy

Due to the infeasibility of cross-validating the entire dataset, we adopt a partition-based approach for data splitting and masking. This method ensures robust training, validation, and testing phases while addressing potential data leakage issues.

#### Data Splitting

1. **Partitioning the Data**:
   - The data time horizon is divided into $ N $ partitions.
   - For each partition, the first 80% of the data span is designated as the training set, the next 10% as the validation set, and the final 10% as the test set.

2. **Purging to Avoid Data Leakage**:
   - When the training data trails the test data, the first day of the training set is removed to prevent data leakage from autocorrelation.
   - This technique, known as purging, is recommended by Lopez de Prado to ensure the integrity of the training process.

#### Dynamic Data Masking

To simulate the task of recovering masked surface points, we employ a dynamic masking strategy with varying proportions of masked data:

1. **Masking Proportions**:
   - At each iteration, a random masking proportion is selected from the set \([0.1, 0.3, 0.5, 0.7]\).

2. **Random Selection of Masked Points**:
   - The masked points are randomly chosen based on the selected masking proportion, ensuring diverse and comprehensive training scenarios.

By adopting this data splitting and masking strategy, we ensure that the model is trained and validated on robust, non-overlapping datasets while effectively simulating real-world conditions where data sparsity and recovery are critical challenges.


### Baseline Comparison and Empirical Results

The IvySPT model was rigorously tested against a variety of baselines to demonstrate its effectiveness and efficiency in estimating implied volatilities from market data. The comparative analysis involved two key phases: pre-training and fine-tuning, targeting different market conditions and data availability.

#### Baseline Models:
The baseline models included:
1. **Dense Neural Network**: A fully connected neural network that estimates implied volatility values from given moneyness (M), time to maturity (T), and market features.
2. **Random Forest**: A random forest regressor that predicts implied volatility from the moneyness and time to maturity of a surface.
3. **Polynomial Regression**: A polynomial regression model for implied volatility estimation from surface data points.
4. **Cubic Spline**: A spline-based method applied directly to interpolate and extrapolate surface points for volatility estimation.

#### Testing Phases:
1. **Pre-training Phase**: 
   - The models were evaluated on options with abundant surface points to assess their ability to learn from dense data.
   - Metrics: Mean Squared Error (MSE) and, where applicable, arbitrage-related losses (Butterfly and Calendar Arbitrage Losses) were calculated, alongside computational time and memory usage.

2. **Fine-tuning Phase**:
   - Both pre-trained IvySPT and fine-tuned models are evaluated on options characterized by sparser surface data.
   - This phase further included stress-testing by evaluating the models on a subset of the test set corresponding to extreme market conditions to gauge robustness and reliability under stress.

#### Comparative Analysis:
The evaluation focused on:
- **Performance Metrics**: MSE and arbitrage losses provided quantitative measures of prediction accuracy and adherence to financial no-arbitrage conditions.
- **Stress Testing Results**: Specific insights into model performance during market extremes were highlighted to understand the resilience and stability of the models.
- **Resource Utilization**: Reports on computation time and memory consumption offered a perspective on the scalability and practical deployment potential of each model.

By detailed comparison against these benchmarks, the IvySPT model's unique advantages in handling sparse data and complex market dynamics, while ensuring computational efficiency, were underscored.


### Ablation Study

To assess the impact of different components of the IvySPT model, we conducted an ablation study by systematically removing or modifying specific elements. The following variants of the model were evaluated:

1. **Removing the Continuous Kernel**:
   - The elliptical RBF kernel-based continuous kernel embeddings were removed. This variant helps understand the contribution of the continuous kernel to the overall performance. 

2. **Removing the Positional Embedding**:
   - Positional embeddings were excluded from the model. This variant assesses the importance of positional information in representing the surface embeddings.

3. **Removing the External Attention**:
   - The external attention mechanism, which incorporates external market features, was removed. This variant evaluates the impact of external market features on the model's predictive accuracy.

4. **Removing the Gated Fusion**:
   - The gated attention fusion mechanism was replaced with a simple addition of the self-attention and external attention outputs. This variant highlights the effectiveness of the gating mechanism in combining different attention outputs.

5. **Removing the No Arbitrage Conditions Soft Constraints**:
   - The soft constraints ensuring no arbitrage conditions were removed from the loss function. This variant demonstrates the importance of enforcing financial constraints on the model's predictions.

By comparing the performance of these ablated variants with the full IvySPT model, we can quantify the contribution of each component to the overall performance.


### Hyperparameter Search

To optimize the performance of the IvySPT model, a comprehensive hyperparameter search was conducted. This search involved training and validating the model on subsets of the available data, ensuring efficient and effective identification of the best hyperparameters.

#### Methodology:
- **Training and Validation Subsets**:
  - **Training Set**: 10% of the entire dataset was used for training during the hyperparameter search.
  - **Validation Set**: 10% of the entire dataset was used for validation, allowing us to evaluate model performance on unseen data.

#### Hyperparameters Considered:
- **Number of Encoder Blocks**: The depth of the Transformer model.
- **Embedding Dimensions**: The dimensionality of the embeddings used in the model.
- **Learning Rate**: The step size used during gradient descent optimization.
- **Batch Size**: The number of samples processed before the model is updated.
- **Dropout Rates**: The rates at which neurons are randomly dropped during training to prevent overfitting.

#### Evaluation Metrics:
- **Mean Squared Error (MSE)**: Measures the average squared difference between estimated and actual values.
- **Butterfly Arbitrage Loss**: Ensures the model's predictions adhere to no-arbitrage conditions related to volatility smiles.
- **Calendar Arbitrage Loss**: Ensures the model's predictions adhere to no-arbitrage conditions related to term structures.

#### Search Strategy:
- A grid search or random search approach was employed to explore combinations of hyperparameters.
- The **average rankings** of the models were used as a robust metric for selecting the best hyperparameters. This involved averaging the rankings of the models across all evaluation metrics, ensuring a balanced consideration of performance aspects.

#### Reporting:
- **Loss Plots**: For each combination of the number of encoder blocks and embedding dimensions, the three losses (MSE, Butterfly Arbitrage Loss, and Calendar Arbitrage Loss) were plotted. These visualizations provided insights into how different configurations impacted model performance.
- The selected hyperparameters were those that consistently performed well across all metrics, ensuring a robust and generalizable model.

By systematically exploring the hyperparameter space and evaluating the models on subsets of the data, the IvySPT model's performance was fine-tuned, resulting in an optimized configuration ready for deployment and further analysis.


### Sensitivity Analysis to Market Features

#### Overview:
To understand how the IvySPT model's predictions are influenced by market features, we conducted a sensitivity analysis using neural network gradients on the validation set. This analysis helps in interpreting the model's responsiveness to changes in market conditions, providing insights into the robustness and reliability of the predictions.

#### Methodology:
- **Gradient Computation**:
  - For each estimated implied volatility (IV) value, the gradient with respect to each market feature (e.g., market return, market volatility, treasury rate) is calculated.
  - These gradients indicate the direction and magnitude of change in the IV estimates in response to changes in the market features.

#### Interpretation:
- **Magnitude of Gradients**:
  - Large gradient values indicate that the model's predictions are highly sensitive to changes in the corresponding market feature.
  - Small gradient values suggest that the model's predictions are relatively stable with respect to changes in that market feature.

- **Positive vs. Negative Gradients**:
  - Positive gradients imply that an increase in the market feature leads to an increase in the estimated IV.
  - Negative gradients suggest that an increase in the market feature results in a decrease in the estimated IV.

#### Reporting:
- **Aggregate Statistics**:
  - Report the mean and standard deviation of the gradients for each market feature across the entire test set. This provides a summary of the overall sensitivity.
  - Present histograms or box plots to visualize the distribution of gradients for each market feature, offering a detailed view of the variability in sensitivity.

- **Implications**:
  - Discuss the practical implications of the sensitivity analysis, such as the reliability of predictions during periods of market turbulence.
  - Explain how understanding sensitivity can guide risk management and decision-making processes for users of the model.

By thoroughly analyzing and reporting the sensitivity of the IvySPT model to market features, we can provide a transparent and comprehensive evaluation of its performance, addressing potential concerns regarding robustness and reliability in various market conditions.


### Flat Local Minima Analysis

#### Overview:
To demonstrate that the IvySPT model is in a flat local minimum after training, we analyze the gradients and Hessians of the loss landscape using the validation set. A flat local minimum indicates that the model is in a region of the parameter space where the loss surface is relatively flat, suggesting robustness and generalizability of the model.

#### Methodology:
- **Gradient Analysis**:
  - Compute the gradients of the loss with respect to the model parameters.
  - A flat local minimum is indicated by small gradient magnitudes, suggesting that small perturbations in the parameter space do not significantly change the loss.

- **Hessian Analysis**:
  - Compute the Hessian matrix, which contains second-order partial derivatives of the loss with respect to the model parameters.
  - The eigenvalues of the Hessian provide information about the curvature of the loss surface. A flat local minimum is characterized by small eigenvalues, indicating that the loss landscape is flat in various directions.

  ```math
  H = \left[ \frac{\partial^2 L}{\partial \theta_i \partial \theta_j} \right]
  ```

#### Reporting:
- **Gradient Magnitude**:
  - Report the mean and standard deviation of the gradient magnitudes across all model parameters.
  - Present histograms or box plots to visualize the distribution of gradient magnitudes, providing a detailed view of the variability.

- **Hessian Eigenvalues**:
  - Report the eigenvalues of the Hessian matrix, focusing on their magnitude.
  - Present a histogram or a plot of the eigenvalues to illustrate the curvature of the loss landscape.
  - Calculate and report the ratio of small to large eigenvalues to quantify the flatness of the local minimum.

- **Implications**:
  - Discuss the practical implications of finding a flat local minimum, such as improved model robustness and generalizability.
  - Explain how the flatness of the local minimum can lead to better performance on unseen data, as the model is less sensitive to small perturbations in the parameter space.

By conducting a thorough analysis of the gradients and Hessians, we can provide strong evidence that the IvySPT model is in a flat local minimum, reinforcing its robustness and reliability.


### Demonstrating Model Convergence

To demonstrate the convergence of the IvySPT model, we monitor and visualize various metrics throughout the training process. This includes tracking the training and validation losses for each of the three losses in our problem, as well as analyzing the distribution of the Neural Tangent Kernel (NTK) eigenvalues at initialization.

#### Loss Tracking During Training

We track and plot the following losses during the training process to assess the convergence of the model:
1. **Mean Squared Error (MSE) Loss**:
   - This loss measures the difference between the predicted and actual implied volatility values.
2. **Calendar Arbitrage Loss**:
   - This loss ensures that the model's predictions do not exhibit calendar arbitrage violations.
3. **Butterfly Arbitrage Loss**:
   - This loss ensures that the model's predictions do not exhibit butterfly arbitrage violations.

For each loss, we plot both the training and validation losses against the number of epochs. This helps in understanding how well the model is learning and generalizing to unseen data.

#### Neural Tangent Kernel (NTK) Eigenvalues Distribution

At initialization, we analyze the distribution of the NTK eigenvalues using 10% of the entire training set. The NTK provides insights into the training dynamics and convergence properties of the model. By plotting the distribution of the NTK eigenvalues, we can assess the initial conditions of the model and ensure they are conducive to effective training.

```math
\Theta(x, x') = \nabla_\theta f(x; \theta)^\top \nabla_\theta f(x'; \theta)
```

#### Visualization

- **Training and Validation Loss Curves**:
  - Plot the training and validation losses for MSE, calendar arbitrage, and butterfly arbitrage losses against the number of epochs.
  - These plots help in understanding the convergence behavior of the model and identifying any potential overfitting.

- **NTK Eigenvalues Distribution**:
  - Plot the distribution of the NTK eigenvalues at initialization.
  - This plot provides insights into the model's training dynamics and helps in assessing the initial conditions of the model.

By visualizing these metrics, we can effectively demonstrate the convergence of the IvySPT model and ensure that it is learning appropriately from the data.


### Dynamics of Each Loss Contribution During Training

To understand the contribution of each loss to the overall training process, we employ the GradNorm method. GradNorm is an adaptive algorithm that dynamically adjusts the weights of each task's loss function to balance the training rates of different tasks. This ensures that no single task dominates the training process, leading to a more balanced and efficient training regime.

#### GradNorm Method

GradNorm works by normalizing the gradient magnitudes across tasks and adjusting the loss weights to balance the training rates. The algorithm aims to achieve the following goals:

1. **Common Scale for Gradient Magnitudes**:
   - Establish a common scale for gradient magnitudes across tasks.
   - This is achieved by using the average gradient norm as a baseline.

2. **Balancing Training Rates**:
   - Adjust gradient norms so that different tasks train at similar rates.
   - This is done by dynamically adjusting the loss weights based on the relative training rates of the tasks.

The GradNorm algorithm can be summarized by the following key steps:

1. **Define Gradient Norms and Training Rates**:
   - $G_W^{(i)}(t) = \| \nabla_W [w_i(t) L_i(t)] \|_2$: L2 norm of the gradient of the weighted single-task loss.
   - $G_W(t) = \mathbb{E}_{\text{task}} [G_W^{(i)}(t)]$: Average gradient norm across all tasks.
   - $\tilde{L}_i(t) = \frac{L_i(t)}{L_i(0)}$: Loss ratio for task $i$, representing the inverse training rate.
   - $`r_i(t) = \frac{\tilde{L}_i(t)}{\mathbb{E}_{\text{task}} [\tilde{L}_i(t)]}`$: Relative inverse training rate.

2. **Adjust Gradient Norms**:
   - Target gradient norm for each task $i$ is adjusted as:
     
$$G_W^{(i)}(t) \rightarrow G_W(t) \times [r_i(t)]^\alpha$$

   where $\alpha$ is a hyperparameter that sets the strength of the balancing force.

3. **Compute Gradient Loss**:
   - The gradient loss $ L_{\text{grad}} $ is defined as:
     
$$L_{\text{grad}}(t; w_i(t)) = \sum_i \left| G_W^{(i)}(t) - G_W(t) \times [r_i(t)]^\alpha \right|_1$$

   This loss penalizes the network when gradient magnitudes are too large or too small compared to the desired target.

4. **Update Loss Weights**:
   - The loss weights $w_i(t)$ are updated using the gradient of $L_{\text{grad}}$ to balance the training rates.

By applying GradNorm, we ensure that the training process is balanced and that each task contributes effectively to the overall learning.

#### Visualization of Loss Coefficients

To visualize the dynamics of each loss contribution during training, we plot the coefficients of each loss function at each iteration. These plots help us understand how the model adjusts the importance of each task's loss over time.

- **Loss Coefficients Plot**:
  - The plot shows the evolution of the coefficients for the mean squared error (MSE) loss, calendar arbitrage loss, and butterfly arbitrage loss during training.
  - This visualization helps in understanding how the GradNorm algorithm dynamically adjusts the loss weights to achieve balanced training.

By analyzing these plots, we can gain insights into the training dynamics and the effectiveness of the GradNorm method in balancing the loss contributions.


### Loss Decomposition Analysis

To gain a deeper understanding of the model's performance, we decompose the losses based on their corresponding year and mask proportion. This decomposition allows us to analyze how the losses vary across different time periods and masking proportions, providing insights into the model's behavior under different conditions.

#### Loss Decomposition by Year

By decomposing the losses based on the year, we can observe how the model's performance evolves over time. This is particularly useful for identifying any temporal patterns or trends in the losses, which may indicate the model's sensitivity to changes in market conditions or other temporal factors.

#### Loss Decomposition by Mask Proportion

Decomposing the losses based on the mask proportion helps us understand how the model performs under different levels of data sparsity. By analyzing the losses for different masking proportions, we can evaluate the model's robustness to missing data and its ability to generalize from incomplete surfaces.

#### Visualization of Loss Decomposition

To visualize the loss decomposition, we create plots that show the distribution of each loss type (MSE, calendar arbitrage, butterfly arbitrage) based on the year and mask proportion. These plots provide a detailed view of the model's performance across different scenarios.

- **Loss Decomposition by Year**:
  - Plot the MSE, calendar arbitrage loss, and butterfly arbitrage loss for each year.
  - This helps identify any temporal patterns or trends in the losses.

- **Loss Decomposition by Mask Proportion**:
  - Plot the MSE, calendar arbitrage loss, and butterfly arbitrage loss for each mask proportion.
  - This helps evaluate the model's robustness to missing data and its ability to generalize from incomplete surfaces.

By analyzing these plots, we can gain valuable insights into the model's performance across different time periods and data sparsity levels, helping us understand the strengths and weaknesses of the model under various conditions.


### Visualizing Attention Maps

To better understand the behavior and decision-making process of the IvySPT model, we can visualize the attention maps. Specifically, we focus on the self-attention and external attention mechanisms from the last Transformer layer. These visualizations help us see which parts of the input data the model is focusing on during its predictions.

1. **Self-Attention Maps**:
   - The self-attention mechanism captures dependencies between different positions within the surface embeddings.
   - Visualizing these maps shows how the model correlates various points in the input surface to make its predictions.

2. **External Attention Maps**:
   - The external attention mechanism incorporates external market features into the encoding process.
   - Visualizing these maps helps us understand how external market conditions influence the model's decisions.

---
---
