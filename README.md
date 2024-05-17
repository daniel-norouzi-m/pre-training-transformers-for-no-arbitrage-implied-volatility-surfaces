---

# Variational Inference with Planar Flows for Implied Volatility Surfaces

## Introduction

This repository contains an implementation of a variational autoencoder (VAE) with planar flows to model and fill implied volatility (IV) surfaces. Implied volatility surfaces are crucial in financial markets for pricing options and managing risk. This project aims to improve the flexibility and accuracy of modeling IV surfaces by leveraging the power of normalizing flows and variational inference.

## Core Idea

The core idea of this project is to use planar flows within a variational inference framework to learn a flexible posterior distribution for the latent variables. The model is conditioned on a grid of IVs and additional asset and market features, allowing it to generate and fill missing IV points accurately. The generative model uses the learned posterior samples along with specific conditioning features such as strike price and time to maturity to decode to a single volatility point.

## Project Structure

The project is structured into several key components:

1. **Data Generation**: Synthetic data generation for implied volatility surfaces and associated features.
2. **Planar Flow Layer**: Implementation of planar flows to ensure invertible transformations.
3. **Encoder Network**: Neural network to encode the input IV grid and features into latent variables.
4. **Decoder Network**: Neural network to decode latent variables and conditioning features into implied volatility points.
5. **Variational Autoencoder (VAE)**: Integration of encoder, planar flows, and decoder into a VAE.
6. **Training Loop**: Training procedure to optimize the model parameters.

## Mathematical Background

### Variational Inference (VI)

Variational Inference (VI) is a method used to approximate complex posterior distributions in Bayesian inference. Instead of directly computing the posterior $p(z|x)$, which is often intractable, VI optimizes a simpler distribution $q_{\phi}(z|x)$ to be close to the true posterior. This is achieved by minimizing the Kullback-Leibler (KL) divergence between the approximate posterior and the true posterior:

```math
\text{KL}(q_{\phi}(z|x) \| p(z|x)) = \mathbb{E}_{q_{\phi}(z|x)} \left[ \log \frac{q_{\phi}(z|x)}{p(z|x)} \right]
```

Minimizing this divergence is equivalent to maximizing the Evidence Lower Bound (ELBO):

$$\mathcal{L}(q) = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - \text{KL}(q_{\phi}(z|x) \| p(z))$$

### Normalizing Flows

Normalizing Flows are a series of invertible transformations applied to a simple initial distribution (e.g., Gaussian) to obtain a more complex distribution. These transformations allow us to model flexible posterior distributions in VI. Each transformation must be invertible and differentiable to ensure that we can compute the Jacobian determinant for the change of variables.

### Planar Flows

Planar Flows are a specific type of normalizing flow where each transformation is defined as:

$$z_k = z_{k-1} + u h(w^T z_{k-1} + b)$$

Here, $u$, $w$, and $b$ are learnable parameters, and $h$ is a non-linear activation function, typically $\tanh$. The invertibility condition for planar flows is:

$$w^T u \geq -1$$

To ensure this condition is met, we modify $u$ as follows:

$$\tilde{u} = u + \left( m(w^T u) - w^T u \right) \frac{w}{\|w\|^2}$$

where $m(x) = -1 + \log(1 + e^x)$.

The log-determinant of the Jacobian for the transformation is:

$$\ln \left| \det \frac{\partial z_k}{\partial z_{k-1}} \right| = \ln \left| 1 + u^T h'(w^T z_{k-1} + b) w \right|$$

### Evidence Lower Bound (ELBO)

The ELBO in the context of our model with planar flows can be written as:

$$\mathcal{L}(q) = \mathbb{E}_{q_0(z_0)} \left[ \log p_{\theta}(x|z_K) + \log p(z_K) - \log q_0(z_0) - \sum_{k=1}^K \log \left| \det \frac{\partial z_k}{\partial z_{k-1}} \right| \right]$$

Where $z_K$ is the final latent variable obtained after applying $K$ planar flows to the initial latent variable $z_0$.

### ELBO Loss Function

The ELBO loss function combines the reconstruction loss (e.g., mean squared error) and the KL divergence between the approximate posterior and the prior:

$$\text{ELBO} = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - \text{KL}(q_{\phi}(z|x) \| p(z))$$

The KL divergence term regularizes the approximate posterior to be close to the prior, while the reconstruction term measures how well the model can reconstruct the input data from the latent variables.

## Implementation

### Data Generation

We generate synthetic sample data to simulate IV surfaces and additional asset and market features. The IV surfaces are represented as grids with some missing values to mimic real-world scenarios.

### Planar Flow Layer

Planar flows modify their parameters to ensure invertibility, which is crucial for the flow-based variational inference framework. The planar flow layer includes the condition $w^T u \geq -1$ to maintain invertibility.

### Encoder Network

The encoder network takes the IV grid and additional features as input and outputs the parameters of the initial approximate posterior distribution. This network is a crucial part of the variational inference process.

### Decoder Network

The decoder network takes the latent variable $z$ and conditioning features (including strike price and time to maturity) to generate a single implied volatility point. This network enables the generative capabilities of the model.

### Variational Autoencoder (VAE)

The VAE integrates the encoder, multiple planar flows, and decoder. It uses the encoded input and planar flows to learn a flexible posterior distribution, and the decoder generates the IV points.

### Training Loop

The training loop optimizes the model parameters by iterating through the data, performing forward and backward passes, and updating the model using the Adam optimizer. The loss function used is the Evidence Lower Bound (ELBO), which combines the reconstruction term and the KL divergence term.

## Getting Started

### Prerequisites

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Matplotlib (for visualization)

### Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/iv-planar-flows.git
cd iv-planar-flows
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Code

To generate the data and train the model, you can run the provided Jupyter notebook:

```bash
jupyter notebook iv_planar_flows.ipynb
```

### Usage

The main components of the project can be used as follows:

1. **Data Generation**: Create synthetic data for IV surfaces and features.
2. **Planar Flow Layer**: Implement and use planar flow layers in your model.
3. **Encoder and Decoder Networks**: Define and train the encoder and decoder networks.
4. **VAE with Planar Flows**: Integrate all components into a VAE and train the model.
5. **Training**: Use the provided training loop to optimize your model.

## Future Work

- **Model Refinement**: Experiment with different types of normalizing flows and network architectures.
- **Data Augmentation**: Incorporate real-world financial data and enhance the data augmentation techniques.
- **Hyperparameter Tuning**: Optimize hyperparameters for better performance.
- **Applications**: Apply the model to other financial time series and market signals.

## Contributing

We welcome contributions to this project. If you have any ideas, suggestions, or improvements, please open an issue or submit a pull request. Let's make this project better together!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- This project is inspired by research in variational inference and normalizing flows.
- We thank the PyTorch community for providing an excellent deep learning framework.

---
