import gc
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from ivyspt.adaptive_loss_coefficients import AdaptiveLossCoefficients
from ivyspt.surface_arbitrage_free_loss import SurfaceArbitrageFreeLoss

def send_batch_to_device(batched_data, device):
    def move_to_device(data, device):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {key: move_to_device(value, device) for key, value in data.items()}
        elif isinstance(data, list):
            return [move_to_device(item, device) for item in data]
        else:
            return data  # For non-tensor data (e.g., strings), return as is

    return move_to_device(batched_data, device)

class Trainer:
    def __init__(
        self, 
        model, 
        train_data_loader, 
        validate_data_loader, 
        test_data_loader, 
        n_epochs, 
        warmup_ratio, 
        peak_learning_rate, 
        min_learning_rate, 
        gradient_clip, 
        adamw_betas, 
        adamw_epsilon, 
        adamw_weight_decay, 
        layer_wise_decay,
        loss_asymmetry_alpha, 
        device,
        remove_multi_loss=False
    ):
        self.model = model.to(device)
        self.train_data_loader = train_data_loader
        self.validate_data_loader = validate_data_loader
        self.test_data_loader = test_data_loader
        self.n_epochs = n_epochs
        self.warmup_epochs = int(warmup_ratio * n_epochs)
        self.loss_asymmetry_alpha = loss_asymmetry_alpha
        self.gradient_clip = gradient_clip
        self.peak_learning_rate = peak_learning_rate
        self.min_learning_rate = min_learning_rate
        self.device = device
        self.remove_multi_loss = remove_multi_loss

        # AdamW Optimizer with Layer-wise decay
        self.optimizer = AdamW(
            self._layer_wise_learning_rate_decay(layer_wise_decay, peak_learning_rate), 
            betas=adamw_betas, 
            eps=adamw_epsilon, 
            weight_decay=adamw_weight_decay
        )

        # Learning Rate Scheduler
        warmup_scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / self.warmup_epochs)
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.n_epochs - self.warmup_epochs,
            eta_min=self.min_learning_rate
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_epochs]
        )

    def _layer_wise_learning_rate_decay(
        self, 
        layer_wise_decay, 
        base_lr
    ):
        params = []

        # Final layer (depth 0)
        params.append({
            'params': self.model.final_layer.parameters(),
            'lr': base_lr
        })

        # Surface Encoder layers (depth from 1 to num_encoder_blocks)
        if layer_wise_decay is not None:
            for i, encoder in enumerate(self.model.surface_encoder.encoders):
                lr = base_lr * (layer_wise_decay ** (i + 1))
                params.append({
                    'params': encoder.parameters(),
                    'lr': lr
                })

            # Surface Embedding layers (highest depth)
            params.append({
                'params': self.model.surface_embedding.parameters(),
                'lr': base_lr * (layer_wise_decay ** (len(self.model.surface_encoder.encoders) + 1))
            })
        else:
            # No decay: All layers use the base learning rate
            params.extend([
                {'params': self.model.surface_encoder.parameters(), 'lr': base_lr},
                {'params': self.model.surface_embedding.parameters(), 'lr': base_lr},
            ])

        return params    

    def train(
        self, 
        experiment_name=None
    ):
        # Initialize TensorBoard writer if an experiment name is provided
        if experiment_name:
            writer = SummaryWriter(log_dir=f"runs/{experiment_name}")
        else:
            writer = None

        self.model.train()
        adaptive_loss_weights = None
        loss_coefficients_history = []
        train_loss_components_history = []
        validate_loss_components_history = []

        for epoch in range(self.n_epochs):
            train_loss_components_sums = torch.zeros(3, device=self.device)  
            total_batches = 0

            for batch_idx, batch in enumerate(self.train_data_loader):
                batch = send_batch_to_device(batch, self.device)
                tv_estimates_batch, _, _ = self.model(batch)
                train_loss_components, _ = SurfaceArbitrageFreeLoss(
                    remove_multi_loss=self.remove_multi_loss
                )(tv_estimates_batch, batch)
                
                if adaptive_loss_weights is None: 
                    adaptive_loss_weights = AdaptiveLossCoefficients(
                        initial_losses=train_loss_components.detach().clone(),
                        alpha=self.loss_asymmetry_alpha,
                        learning_rate=self.peak_learning_rate,
                        remove_multi_loss=self.remove_multi_loss,
                        device=self.device
                    )

                # Obtain the current loss coefficients
                loss_coefficients = adaptive_loss_weights.weights.detach().clone()
                train_loss = train_loss_components @ loss_coefficients

                # Record the current loss coefficients
                loss_coefficients_history.append(loss_coefficients.cpu().numpy())

                # Accumulate the loss components
                train_loss_components_sums += train_loss_components.detach().clone()

                # Reset gradients 
                self.optimizer.zero_grad()
                train_loss.backward(retain_graph=True)

                adaptive_loss_weights(train_loss_components, self.model.final_layer)
                total_batches += 1

                # Calculate and print the gradient norm before clipping
                # total_norm = 0
                # for p in self.model.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2)
                #         total_norm += param_norm.item() ** 2
                # total_norm = total_norm ** 0.5
                # print(f"Gradient Norm before clipping: {total_norm:.4f}")  

                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:  # Ensure the parameter has a gradient
                #         grad_min = param.grad.min().item()  # Minimum gradient value
                #         grad_max = param.grad.max().item()  # Maximum gradient value
                #         print(f"Layer: {name} | Gradient Interval: [{grad_min}, {grad_max}]")      

                # Apply gradient clipping if gradient_clip is set, otherwise do nothing
                if self.gradient_clip is not None:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), self.gradient_clip)

                # clipped_norm = 0
                # for p in self.model.parameters():
                #     if p.grad is not None:
                #         param_norm = p.grad.data.norm(2)
                #         clipped_norm += param_norm.item() ** 2
                # clipped_norm = clipped_norm ** 0.5
                # print(f"Gradient Norm after clipping: {clipped_norm:.4f}")    

                # initial_params = {}
                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:  # Only consider parameters that require gradients
                #         initial_params[name] = param.clone().detach()  # Clone the parameter to avoid any references


                # Perform optimizer step
                self.optimizer.step()    

                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:  # Only consider parameters that require gradients
                #         # Calculate the absolute change
                #         absolute_change = torch.abs(param - initial_params[name])
                        
                #         # Calculate the mean of the absolute changes
                #         avg_absolute_change = absolute_change.mean().item()
                        
                #         # Print the average absolute change for this parameter
                #         print(f"Parameter {name} average absolute change: {avg_absolute_change:.6f}")

                    
                if writer:
                    # Log the train loss and loss components to TensorBoard
                    writer.add_scalars('Train Loss', {
                        'MSE Loss': train_loss_components[0].item(),
                        'Calendar Arbitrage Loss': train_loss_components[1].item(),
                        'Butterfly Arbitrage Loss': train_loss_components[2].item(),
                    }, epoch * len(self.train_data_loader) + batch_idx)

                    writer.add_scalars('Loss Coeffficients', {
                        'MSE Loss': loss_coefficients[0].item(),
                        'Calendar Arbitrage Loss': loss_coefficients[1].item(),
                        'Butterfly Arbitrage Loss': loss_coefficients[2].item(),
                    }, epoch * len(self.train_data_loader) + batch_idx)

                # Free up memory
                del batch, tv_estimates_batch, train_loss_components, loss_coefficients, train_loss
                torch.cuda.empty_cache()
                gc.collect()
            
            # Calculate the average loss components for this epoch
            avg_train_loss_components = train_loss_components_sums / total_batches
            train_loss_components_history.append(avg_train_loss_components.cpu().numpy())    
            
            # Validate after each epoch
            avg_validate_loss_components = self.validate()
            validate_loss_components_history.append(avg_validate_loss_components.cpu().numpy())

            if writer:
                print(f"Epoch {epoch + 1}/{self.n_epochs} - Training Loss: {avg_train_loss_components.cpu().numpy()}, Validation Loss: {avg_validate_loss_components.cpu().numpy()}")
                 # Log the train loss and loss components to TensorBoard
                writer.add_scalars('Validation Loss', {
                    'MSE Loss': avg_validate_loss_components[0].item(),
                    'Calendar Arbitrage Loss': avg_validate_loss_components[1].item(),
                    'Butterfly Arbitrage Loss': avg_validate_loss_components[2].item(),
                }, epoch)
                
            # Adjust learning rate
            self.scheduler.step()


        if writer:
            writer.close()    

        return loss_coefficients_history, train_loss_components_history, validate_loss_components_history


    def validate(self):
        self.model.eval()

        validate_loss_components_sums = torch.zeros(3, device=self.device)  
        total_batches = 0

        for batch in self.validate_data_loader:
            batch = send_batch_to_device(batch, self.device)
            tv_estimates_batch, _, _ = self.model(batch)

            validate_loss_components, _ = SurfaceArbitrageFreeLoss()(tv_estimates_batch, batch)
            validate_loss_components_sums += validate_loss_components.detach().clone()
            total_batches += 1

            # Free up memory
            del batch, tv_estimates_batch, validate_loss_components
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate the average loss components for this epoch
        avg_validate_loss_components = validate_loss_components_sums / total_batches  

        return avg_validate_loss_components    
    
    
    def test(
        self,
        output_attention_map=False
    ):
        self.model.eval()

        if output_attention_map:
            with torch.no_grad():
                batch = next(iter(self.test_data_loader))
                batch = send_batch_to_device(batch, self.device)
                tv_estimates_batch, self_attention_maps, external_attention_maps = self.model(batch, output_attention_map=output_attention_map)

                return self_attention_maps, external_attention_maps

        test_loss_components_sums = torch.zeros(3, device=self.device)  
        total_batches = 0
        test_loss_records = []

        for batch in self.test_data_loader:
            batch = send_batch_to_device(batch, self.device)
            tv_estimates_batch, _, _ = self.model(batch)

            test_loss_components, loss_records = SurfaceArbitrageFreeLoss()(tv_estimates_batch, batch, testing_mode=True)
            test_loss_components_sums += test_loss_components.detach().clone()
            total_batches += 1
            test_loss_records.append(loss_records)

            # Free up memory
            del batch, tv_estimates_batch, test_loss_components, loss_records
            torch.cuda.empty_cache()
            gc.collect()

        # Calculate the average loss components for this epoch
        test_loss_components = test_loss_components_sums / total_batches  
        test_loss_records = pd.concat(test_loss_records) 

        return test_loss_components, test_loss_records   
    
