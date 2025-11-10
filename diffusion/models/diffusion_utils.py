import torch
from typing import Optional, Dict, Tuple
from .ddpm import normalize_pcd, unnormalize_pcd

class DiffusionTimeHandler:
    def __init__(self, num_timesteps: int, sampling_timesteps: int):
        times = torch.linspace(-1, num_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        self._time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    def get_timesteps(self, time: float) -> Tuple[int, int]:
        """
        :param time: Value of tau, the diffusion time parameter which runs from one to zero during denoising.
        :return: Tuple of (current, next) integer timestep for the diffusion model.
        """
        time = 1. - time
        assert 0. <= time <= 1.
        time_idx = int(round(time * len(self._time_pairs)))
        time_idx = min(time_idx, len(self._time_pairs) - 1)
        time_idx = max(time_idx, 0)

        time, time_next = self._time_pairs[time_idx]
        return time, time_next

class DiffusionRegulariser:
    """
    Main class for using the denoising diffusion model to regularise ACE.
    """
    def __init__(self, diffusion_model, device):
        self._diffusion_model = diffusion_model
        self._time_handler = DiffusionTimeHandler(num_timesteps=200, sampling_timesteps=diffusion_model.sampling_timesteps)
        self._device = device
        self.centroid_pc = diffusion_model.centroid_pc
        self.scale_pc = diffusion_model.scale_pc
        self.dynamic_norm = False

    def get_diff_loss(self, pcd_, time):
        pcd = pcd_.clone()
        if self.dynamic_norm:
            centroid_pc = torch.mean(pcd, dim=1).detach()
            scale_pc = torch.max(torch.sqrt(torch.sum((pcd - centroid_pc) ** 2, dim=1))).detach()
            # avoid zero scale
            scale_pc = torch.max(scale_pc, torch.tensor(1e-3, device=scale_pc.device))
        else:
            centroid_pc = self.centroid_pc.to(pcd.device)
            scale_pc = self.scale_pc.to(pcd.device)
        pcd = normalize_pcd(pcd, centroid_pc, scale_pc)

        pcd = pcd.permute(0, 2, 1)  # (B, 3, N)
        time, time_next = self._time_handler.get_timesteps(time)

        sigma_lambda = (1. - self._diffusion_model.alphas_cumprod[time]).sqrt()
        assert sigma_lambda > 0.
        model_predictions = self._diffusion_model.model_predictions(
            x=pcd,
            t=torch.Tensor([time], ).to(torch.int64).to(self._device),
            cond=None,
            clip_x_start=True
        )

        grad_log_prior_prob = -model_predictions.pred_noise.detach() * (1. / sigma_lambda)

        multiplier = 1 / torch.linalg.norm(grad_log_prior_prob.detach())
        diffusion_pseudo_loss = multiplier * grad_log_prior_prob * pcd

        pred_x0 = unnormalize_pcd(model_predictions.pred_x_start.permute(0, 2, 1), centroid_pc, scale_pc)

        return diffusion_pseudo_loss, pred_x0
