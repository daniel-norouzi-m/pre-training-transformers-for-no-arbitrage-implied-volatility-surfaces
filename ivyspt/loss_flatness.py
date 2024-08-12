import torch
import torch.autograd as autograd
import gc

from ivyspt.surface_arbitrage_free_loss import SurfaceArbitrageFreeLoss
from ivyspt.trainer import send_batch_to_device


class OptimalLossFlatness:
    def __init__(
        self, 
        model, 
        validate_data_loader, 
        device
    ):
        self.model = model.to(device)
        self.validate_data_loader = validate_data_loader
        self.device = device

    def compute_gradients(
        self, 
        loss, 
        final_layer_params
    ):
        # Compute gradients of the loss with respect to final layer parameters
        grads = autograd.grad(loss, final_layer_params, create_graph=True, retain_graph=True)
        flat_grads = torch.cat([grad.view(-1) for grad in grads])
        return flat_grads

    def compute_hessian_diag(
        self, 
        flat_grads, 
        final_layer_params
    ):
        # Compute the diagonal of the Hessian matrix using the computed gradients
        hessian_diag = torch.zeros_like(flat_grads, device=self.device)
        for i in range(flat_grads.size(0)):
            second_derivatives = autograd.grad(flat_grads[i], final_layer_params, retain_graph=True, create_graph=True)
            second_derivatives_flat = torch.cat([sd.view(-1) for sd in second_derivatives])
            hessian_diag[i] = second_derivatives_flat[i].detach().clone()
        return hessian_diag

    def calculate_flatness(self):
        self.model.eval()
        final_layer_params = list(self.model.final_layer.parameters())
        num_params = sum(p.numel() for p in final_layer_params)
        
        gradients_sum = torch.zeros(3, num_params, device=self.device)
        hessian_diag_sum = torch.zeros(3, num_params, device=self.device)
        total_batches = 0

        for batch in self.validate_data_loader:
            batch = send_batch_to_device(batch, self.device)
            tv_estimates_batch, _, _ = self.model(batch)
            validate_loss_components, _ = SurfaceArbitrageFreeLoss()(tv_estimates_batch, batch)

            for i, loss_component in enumerate(validate_loss_components):
                flat_grads = self.compute_gradients(loss_component, final_layer_params)
                hessian_diag = self.compute_hessian_diag(flat_grads, final_layer_params)

                gradients_sum[i] += flat_grads.detach().clone()
                hessian_diag_sum[i] += hessian_diag

            total_batches += 1

            # Free up memory
            del batch, tv_estimates_batch, validate_loss_components
            torch.cuda.empty_cache()
            gc.collect()

        # Averaging the gradients and Hessian diagonals over all batches
        avg_gradients = gradients_sum / total_batches
        avg_hessian_diag = hessian_diag_sum / total_batches

        # Computing the eigenvalues of the diagonal Hessian
        eigenvalues = torch.sort(avg_hessian_diag, dim=1)[0]

        return avg_gradients, eigenvalues