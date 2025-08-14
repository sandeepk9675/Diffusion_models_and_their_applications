from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
    
        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        One step denoising function of DDPM: x_t -> x_{t-1}.

        Input:
            x_t (`torch.Tensor [B,C,H,W]`): samples at arbitrary timestep t.
            t (`int`): current timestep in a reverse process.
            eps_theta (`torch.Tensor [B,C,H,W]`): predicted noise from a learned model.
        Ouptut:
            sample_prev (`torch.Tensor [B,C,H,W]`): one step denoised sample. (= x_{t-1})
        """

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM reverse step.
        B, C, H, W  = x_t.shape
        t = torch.tensor(t, device=x_t.device)
        
        t  =  t.expand(B,)
        # t = t.reshape(-1, 1, 1, 1)
        sample_prev = None
        #######################
        alpha_t = self._get_teeth(self.alphas, t)                  # α_t
        alpha_bar_t = self._get_teeth(self.alphas_cumprod, t)      # α̅_t
        beta_t = self._get_teeth(self.betas, t)                    # β_t

        # alpha_t shape: [batch_size, 1, 1, 1]
        # alpha_bar_t shape: [batch_size, 1, 1, 1]
        # beta_t shape: [batch_size, 1, 1, 1]

        # 3. Compute the mean μ_t
        mean_coeff = 1.0 / torch.sqrt(alpha_t) # [batch_size, 1, 1, 1]
        eps_coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) # [batch_size, 1, 1, 1]

        mu_t = mean_coeff * (x_t - eps_coeff * eps_theta) # [batch_size, C, H, W]

        sigma_t_sq = beta_t # *sigma_t_sq = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t))
        # sigma_t_sq: [batch_size, 1, 1, 1]

        sample_prev = mu_t + torch.sqrt(sigma_t_sq) * torch.randn_like(x_t)
        

        return sample_prev
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        # print(f"consts shape: {consts.shape}, t shape: {t.shape}")
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
        
        if eps is None:
            eps = torch.randn(x_0.shape, device='cuda')

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        B, C, H, W = x_0.shape
        print(f"hello world t shape: {t.shape}")
        alphas_prod_t = self._get_teeth(self.alphas_cumprod, t)
        # alphas_prod_t shape: [batch_size, 1, 1, 1]
        ## alphas_prod_t = alphas_prod_t.expand(B, C, H, W)
        ## alphas_prod_t shape: [batch_size, C, H, W]
    
        xt = torch.sqrt(alphas_prod_t ) * x0 + torch.sqrt(1 - alphas_prod_t) * eps
        # xt shape: [batch_size, 2]
        x_t = xt
        #######################

        return x_t, eps
