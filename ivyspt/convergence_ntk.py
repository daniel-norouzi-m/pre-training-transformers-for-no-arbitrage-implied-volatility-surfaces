import torch
import torch.autograd as autograd
import torch.linalg as linalg
from ivyspt.trainer import send_batch_to_device
import gc

class ConvergenceNTK:
    def __init__(
        self, 
        model, 
        validate_data_loader, 
        device
    ):
        self.model = model.to(device)
        self.validate_data_loader = validate_data_loader
        self.device = device

    def compute_ntk_matrix(
        self, 
        gradients_matrix
    ):
        # Compute NTK matrix as Gram matrix: NTK = G * G^T
        ntk_matrix = torch.mm(gradients_matrix, gradients_matrix.t())
        return ntk_matrix

    def calculate_ntk_eigenvalues(self):
        self.model.eval()
        model_params = list(self.model.parameters())
        gradients_list = []

        for batch in self.validate_data_loader:
            batch = send_batch_to_device(batch, self.device)
            tv_estimates_batch, _, _ = self.model(batch)

            for tv_estimate in tv_estimates_batch:
                for output in tv_estimate:
                    # Compute gradients of each output with respect to the final layer parameters
                    grads = autograd.grad(output, model_params, retain_graph=True, create_graph=True)
                    flat_grads = torch.cat([grad.view(-1) for grad in grads])
                    gradients_list.append(flat_grads.detach().clone())  # Detach to avoid unnecessary graph building

            # Free up memory
            del batch, tv_estimates_batch
            torch.cuda.empty_cache()
            gc.collect()

        # Convert gradients list to a tensor
        gradients_matrix = torch.stack(gradients_list)  # Shape: (num_samples, num_params)

        # Compute NTK matrix
        ntk_matrix = self.compute_ntk_matrix(gradients_matrix)

        # Calculate eigenvalues of the NTK matrix
        eigenvalues = linalg.eigvalsh(ntk_matrix)  # Use symmetric eigenvalue computation

        return eigenvalues