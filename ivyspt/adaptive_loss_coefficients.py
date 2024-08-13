import torch


class AdaptiveLossCoefficients(torch.nn.Module):
    def __init__(
        self, 
        initial_losses, 
        alpha=1.0, 
        learning_rate=0.01,
        remove_multi_loss=False,
        device='cpu',
        epsilon=1e-8  # Small value to avoid division by zero
    ):
        """
        Initializes the adaptive loss weights module.

        Args:
            initial_losses (torch.Tensor): Initial loss values for each task to set the initial loss ratios.
            alpha (float): The strength of the restoring force in balancing training rates.
            learning_rate (float): Learning rate for updating the weights.
            remove_multi_loss (bool): If True, only use the first loss component and ignore the others.
            device (str): The device to which the weights tensor will be moved.
            epsilon (float): A small value to prevent division by zero.
        """
        super(AdaptiveLossCoefficients, self).__init__()
        self.initial_losses = (initial_losses + epsilon).to(device)
        self.alpha = alpha
        self.remove_multi_loss = remove_multi_loss

        if remove_multi_loss:
            self.weights = torch.zeros_like(self.initial_losses).to(device)
            self.weights[0] = 1.0  # Set the first loss weight to 1
        else:
            self.weights = torch.nn.Parameter(torch.ones_like(self.initial_losses).to(device))
            self.optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
            self.total_weights = self.weights.sum().item()  # Total of weights to maintain normalization


    def forward(
        self, 
        current_losses, 
        final_layer,
    ):
        """
        Adjusts and normalizes the weights based on current losses using the GradNorm approach.
        If remove_multi_loss is True, this function does nothing.

        Args:
            current_losses (torch.Tensor): Current computed losses from the main model.
            final_layer (torch.nn.Module): The final layer of the model whose parameters are used for 
            gradient norm calculation. 

        Returns:
            None: The updated weights are detached and stored within the module.
        """
        if self.remove_multi_loss:
            return  # Do nothing if multi-loss is removed

        loss_ratios = current_losses / self.initial_losses
        relative_inverse_rates = loss_ratios / loss_ratios.mean()

        # Compute gradient norms for each weighted loss
        gradient_norms = torch.stack([
            torch.norm(torch.autograd.grad(self.weights[i] * loss, final_layer.parameters(), create_graph=True)[0])
            for i, loss in enumerate(current_losses)
        ])

        target_gradient_norms = (gradient_norms.mean() * (relative_inverse_rates ** self.alpha)).detach()
        gradnorm_loss = torch.sum(torch.abs(gradient_norms - target_gradient_norms))

         # Manually calculate gradients with respect to self.weights
        grad_weights = torch.autograd.grad(gradnorm_loss, self.weights, create_graph=False)[0]
        self.weights.grad = grad_weights  # Set the gradients manually

        # Update the weights using the optimizer
        self.optimizer.step()

        # Normalize to sum to total_weights, detach, and ensure gradient tracking
        with torch.no_grad():
            normalized_weights = self.weights / self.weights.sum() * self.total_weights
            self.weights.data = normalized_weights.detach()  # Explicitly detach from the graph

        # Re-enable gradient tracking on the updated weights
        self.weights.requires_grad_()