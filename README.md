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

## Implementation

### Data Generation

We generate synthetic sample data to simulate IV surfaces and additional asset and market features. The IV surfaces are represented as grids with some missing values to mimic real-world scenarios.

### Planar Flow Layer

Planar flows modify their parameters to ensure invertibility, which is crucial for the flow-based variational inference framework. The planar flow layer includes the condition \( w^T u \geq -1 \) to maintain invertibility.

### Encoder Network

The encoder network takes the IV grid and additional features as input and outputs the parameters of the initial approximate posterior distribution. This network is a crucial part of the variational inference process.

### Decoder Network

The decoder network takes the latent variable \( z \) and conditioning features (including strike price and time to maturity) to generate a single implied volatility point. This network enables the generative capabilities of the model.

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
