import torch


class AdaptiveLossCoefficients(torch.nn.Module):
    def __init__(
        self, 
        initial_losses, 
        alpha=1.0, 
        learning_rate=0.01
    ):
        """
        Initializes the adaptive loss weights module.

        Args:
            initial_losses (torch.Tensor): Initial loss values for each task to set the initial loss ratios.
            alpha (float): The strength of the restoring force in balancing training rates.
            learning_rate (float): Learning rate for updating the weights.
        """
        super(AdaptiveLossCoefficients, self).__init__()
        self.initial_losses = initial_losses
        self.alpha = alpha
        self.weights = torch.nn.Parameter(torch.ones_like(self.initial_losses))
        self.optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
        self.total_weights = self.weights.sum().item()  # Total of weights to maintain normalization

    def forward(
        self, 
        current_losses, 
        final_layer
    ):
        """
        Adjusts and normalizes the weights based on current losses using the GradNorm approach.

        Args:
            current_losses (torch.Tensor): Current computed losses from the main model.
            final_layer (torch.nn.Moduler): The final layer of the model whose parameters are used for 
            gradient norm calculation. 

        Returns:
            None: The updated weights are detached and stored within the module.
        """
        loss_ratios = current_losses / self.initial_losses
        relative_inverse_rates = loss_ratios / loss_ratios.mean()

        # Compute gradient norms for each weighted loss
        gradient_norms = torch.stack([
            torch.norm(torch.autograd.grad(self.weights[i] * loss, final_layer.parameters(), create_graph=True)[0])
            for i, loss in enumerate(current_losses)
        ])

        target_gradient_norms = (gradient_norms.mean() * (relative_inverse_rates ** self.alpha)).detach()
        gradnorm_loss = torch.sum(torch.abs(gradient_norms - target_gradient_norms))

        # Update the weights using the GradNorm loss
        self.optimizer.zero_grad()
        gradnorm_loss.backward()
        self.optimizer.step()

        # Normalize to sum to total_weights, detach, and ensure gradient tracking
        with torch.no_grad():
            normalized_weights = self.weights / self.weights.sum() * self.total_weights
            self.weights.data = normalized_weights.detach()  # Explicitly detach from the graph

        # Re-enable gradient tracking on the updated weights
        self.weights.requires_grad_()