import pandas as pd
import torch
import gc

from ivyspt.trainer import send_batch_to_device

class MarketSensitivityAnalysis:
    def __init__(
        self, 
        model, 
        validation_data_loader, 
        device
    ):
        """
        Initialize the MarketSensitivityAnalysis module.
        
        Args:
            model (torch.nn.Module): The trained model to analyze.
            validation_data_loader (torch.utils.data.DataLoader): DataLoader for validation data.
            device (torch.device): Device to run the analysis on (CPU or GPU).
        """
        self.model = model.to(device)
        self.validation_data_loader = validation_data_loader
        self.device = device

    def compute_sensitivity(self):
        """
        Compute the sensitivity of each model output with respect to market features.
        
        Returns:
            sensitivities (pd.Dataframe): A dataframe where row contains
                                          gradients for each market feature.
        """
        self.model.eval()  # Set the model to evaluation mode
        
        sensitivities = []

        # Iterate over each batch
        for batch in self.validation_data_loader:
            batch = send_batch_to_device(batch, self.device)
            
            # Enable gradient computation for market features
            for feature_name in batch['Market Features']:
                batch['Market Features'][feature_name].requires_grad = True
            
            # Forward pass to compute the model's output
            tv_estimates_batch, _, _ = self.model(batch)

            # Iterate over each output tensor in the batch
            for i, tv_estimates in enumerate(tv_estimates_batch):
                # Iterate over each element in the output tensor
                for j in range(tv_estimates.size(0)):
                    tv_output = tv_estimates[j]

                    # Compute gradients with respect to market features
                    grads = {}
                    for market_feature in batch['Market Features']:
                        grad = torch.autograd.grad(
                            outputs=tv_output, 
                            inputs=batch['Market Features'][market_feature], 
                            create_graph=False, 
                            retain_graph=True
                        )[0][i].item()
                        grads[market_feature] = grad

                    sensitivities.append(grads)

            # Free up memory
            del batch, tv_estimates_batch, tv_estimates, tv_output
            torch.cuda.empty_cache()
            gc.collect()

        return pd.DataFrame.from_records(sensitivities)
