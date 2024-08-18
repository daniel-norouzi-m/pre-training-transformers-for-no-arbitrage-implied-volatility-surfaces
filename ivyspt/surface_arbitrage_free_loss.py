import pandas as pd
import torch
import torch.nn as nn

class SurfaceArbitrageFreeLoss(nn.Module):
    def __init__(
        self,
        remove_multi_loss=False
    ):
        super(SurfaceArbitrageFreeLoss, self).__init__()
        self.remove_multi_loss = remove_multi_loss

    def forward(
        self, 
        tv_estimates_batch, 
        batch,
        testing_mode=False,
    ):
        mse_loss_sum = 0.0
        calendar_arbitrage_loss_sum = 0.0
        butterfly_arbitrage_loss_sum = 0.0
        total_elements = 0
        loss_records = []

        for total_implied_variance, target_variance, time_to_maturity, log_moneyness in zip(
            tv_estimates_batch, 
            batch['Query Points']['Total Variance'], 
            batch['Query Points']['Time to Maturity'], 
            batch['Query Points']['Log Moneyness']
        ):
            sequence_length = total_implied_variance.size(0)
            total_elements += sequence_length

            # Calculate mean squared error between model estimates and target variances
            mse_loss = torch.sum((total_implied_variance - target_variance) ** 2)
            mse_loss_sum += mse_loss

            # If remove_multi_loss is True, skip additional loss calculations
            if self.remove_multi_loss:
                continue

            # unit_vectors = torch.eye(sequence_length, device=total_implied_variance.device)

            # # Compute gradients needed for arbitrage conditions
            # w_t = torch.stack([
            #     torch.autograd.grad(
            #         outputs=total_implied_variance, 
            #         inputs=time_to_maturity,
            #         grad_outputs=vec, 
            #         create_graph=True   
            #     )[0]
            #     for vec in unit_vectors
            # ]).diag()

            # w_x = torch.stack([
            #     torch.autograd.grad(
            #         outputs=total_implied_variance, 
            #         inputs=log_moneyness,
            #         grad_outputs=vec, 
            #         create_graph=True   
            #     )[0]
            #     for vec in unit_vectors
            # ]).diag()

            # w_xx = torch.stack([
            #     torch.autograd.grad(
            #         outputs=w_x, 
            #         inputs=log_moneyness, 
            #         grad_outputs=vec,
            #         create_graph=True   
            #     )[0]
            #     for vec in unit_vectors
            # ]).diag()

            # Sum the outputs
            sum_total_implied_variance = total_implied_variance.sum()

            # Calculate the gradient of the sum of outputs with respect to time_to_maturity
            w_t = torch.autograd.grad(
                outputs=sum_total_implied_variance, 
                inputs=time_to_maturity,
                create_graph=True   
            )[0]

            # Calculate the gradient of the sum of outputs with respect to log_moneyness
            w_x = torch.autograd.grad(
                outputs=sum_total_implied_variance, 
                inputs=log_moneyness,
                create_graph=True   
            )[0]

            # Calculate the second-order gradient of the sum of outputs with respect to log_moneyness
            w_xx = torch.autograd.grad(
                outputs=w_x.sum(), 
                inputs=log_moneyness, 
                create_graph=True   
            )[0]

            # Calculate Calendar Arbitrage Loss
            calendar_arbitrage_loss = torch.clamp(-w_t, min=0) ** 2
            calendar_arbitrage_loss_sum += calendar_arbitrage_loss.sum()

            # Calculate Butterfly Arbitrage Loss
            w = total_implied_variance
            g = (1 - log_moneyness * w_x / (2 * w)) ** 2 - w_x / 4 * (1 / w + 1 / 4) + w_xx / 2
            butterfly_arbitrage_loss = torch.clamp(-g, min=0) ** 2
            butterfly_arbitrage_loss_sum += butterfly_arbitrage_loss.sum()
            if testing_mode:
                record = {
                    'MSE Loss': mse_loss.mean().item(),
                    'Calendar Arbitrage Loss': calendar_arbitrage_loss.mean().item(),
                    'Butterfly Arbitrage Loss': butterfly_arbitrage_loss.mean().item()
                }
                loss_records.append(record)

        # Calculate mean losses
        mse_loss = mse_loss_sum / total_elements
        # If remove_multi_loss is True, return the MSE loss
        if self.remove_multi_loss:
            return torch.stack([mse_loss, torch.tensor(0.0), torch.tensor(0.0)]), None

        calendar_arbitrage_loss = calendar_arbitrage_loss_sum / total_elements
        butterfly_arbitrage_loss = butterfly_arbitrage_loss_sum / total_elements

        # Stack losses into a single tensor
        total_losses = torch.stack([mse_loss, calendar_arbitrage_loss, butterfly_arbitrage_loss])

        if testing_mode:
            loss_records = pd.DataFrame(loss_records)
            loss_records['Datetime'] = batch['Datetime']
            loss_records['Mask Proportion'] = batch['Mask Proportion']

            return total_losses, loss_records

        return total_losses, None

