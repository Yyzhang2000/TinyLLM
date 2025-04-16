import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext

from config import MoEConfig


class MoEManager:
    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []

    def reset_aux_loss(self):
        self.aux_loss = []

    def reset_router_z_loss(self):
        self.router_z_loss = []

    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)

    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)

    def aggregate_aux_loss(self):
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)


class Router(nn.Module):
    def __init__(self, config: MoEConfig, manager: MoEManager):
        super().__init__()

        self.config = config

        self.gate = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.w_noise = (
            nn.Linear(config.n_embd, config.num_experts, bias=False)
            if self.config.use_noisy_top_k
            else None
        )

        self.manager = manager

    def forward(self, x):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        ctx = (
            nullcontext()
            if not self.router_use_full_precision
            else torch.autocast(device_type=device_type, enabled=False)
        )

        with ctx:
            B, T, D = x.shape
            num_tokens = B * T
            logits = self.gate(x)

            if self.w_noise is not None:
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(logits)
                logits += noise

            if self.config.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                self.manager.add_router_z_loss(z_loss)

            # Find top k experts
            top_k_logits, top_k_indices = torch.topk(logits, self.config.top_k, dim=-1)

            # Normalize
            router_probs = torch.full_like(logits, float("-inf"))
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)

            if self.config.use_aux_loss:
                aux_loss = self.compute_aux_loss(logits)
                self.manager.add_aux_loss(aux_loss)

            # Get expert capacity
            exp_capacity = self.get_capacity(num_tokens)

            exp_mask = F.one_hot(top_k_indices, num_classes=self.config.num_experts)
            exp_mask = exp_mask.view(
                num_tokens, self.config.top_k, self.config.num_experts
            )
            exp_mask = exp_mask.permute(1, 0, 2)


class MLPExperts(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()

        self.bias = config.bias

        self.c_fc = nn.Parameter(
            torch.empty(config.num_experts, config.n_embd, config.n_embd * 4)
        )

        self.c_proj = nn.Parameter(
            torch.empty(config.num_experts, config.n_embd * 4, config.n_embd)
        )

        self.fc_bias = nn.Parameter(
            torch.empty(config.num_experts, 1, config.n_embd * 4)
        )
        self.proj_bias = nn.Parameter(torch.empty(config.num_experts, 1, config.n_embd))

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)

        if self.bias:
            x = x + self.fc_bias

        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x = x + self.proj_bias

        x = self.dropout(x)
        return x


class MoE(nn.Module):
    def __init__(self, config: MoEConfig, manager: MoEManager):
        super().__init__()
        self.config = config

        self.router = Router(config, manager)
        self.experts = MLPExperts(config)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape

        num_tokens = B * T

        used_capacity, exp_weight, exp_mask = self.router(x)

        x = x.view(num_tokens, D)

        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        exp_out = self.experts(exp_batches)

        exp_weight = exp_weight.view(
            num_tokens,
            -1,
        )
        output = exp_out.vi @ exp_out

        return output.view(B, T, D)
