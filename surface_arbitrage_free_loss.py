import torch
import torch.nn as nn
import torch.nn.functional as F

class SurfaceArbitrageFreeLoss(nn.Module):
    def __init__(self):
        super(SurfaceArbitrageFreeLoss, self).__init__()

    def forward(
        self, 
        tv_estimates_batch, 
        batch,
        testing_mode=False,
        epsilon=1e-5  # Small value to prevent division by zero
    ):
        mspe_loss_sum = 0.0
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

            # Calculate mean squared percentage error between model estimates and target variances
            percentage_error = (total_implied_variance - target_variance) / (target_variance + epsilon)
            mspe_loss = torch.sum(percentage_error ** 2)
            mspe_loss_sum += mspe_loss

            unit_vectors = torch.eye(sequence_length, device=total_implied_variance.device)

            # Compute gradients needed for arbitrage conditions
            w_t = torch.stack([
                torch.autograd.grad(
                    outputs=total_implied_variance, 
                    inputs=time_to_maturity,
                    grad_outputs=vec, 
                    create_graph=True   
                )[0]
                for vec in unit_vectors
            ]).diag()

            w_x = torch.stack([
                torch.autograd.grad(
                    outputs=total_implied_variance, 
                    inputs=log_moneyness,
                    grad_outputs=vec, 
                    create_graph=True   
                )[0]
                for vec in unit_vectors
            ]).diag()

            w_xx = torch.stack([
                torch.autograd.grad(
                    outputs=w_x, 
                    inputs=log_moneyness, 
                    grad_outputs=vec,
                    create_graph=True   
                )[0]
                for vec in unit_vectors
            ]).diag()

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
                    'MSPE Loss': mspe_loss.mean().item(),
                    'Calendar Arbitrage Loss': calendar_arbitrage_loss.mean().item(),
                    'Butterfly Arbitrage Loss': butterfly_arbitrage_loss.mean().item()
                }
                loss_records.append(record)

        # Calculate mean losses
        mspe_loss = mspe_loss_sum / total_elements
        calendar_arbitrage_loss = calendar_arbitrage_loss_sum / total_elements
        butterfly_arbitrage_loss = butterfly_arbitrage_loss_sum / total_elements

        # Stack losses into a single tensor
        total_losses = torch.stack([mspe_loss, calendar_arbitrage_loss, butterfly_arbitrage_loss])

        if testing_mode:
            loss_records = pd.DataFrame(loss_records)
            loss_records['Datetime'] = batch['Datetime']
            loss_records['Mask Proportion'] = batch['Mask Proportion']
            loss_records.set_index(['Datetime', 'Mask Proportion'], inplace=True)

            return total_losses, loss_records

        return total_losses, None

