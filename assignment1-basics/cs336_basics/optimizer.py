import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.99, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m = state["m"]
                v = state["v"]
                t = state["t"]

                # compute adjusted lr for step t
                alpha_t = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)

                # apply weight_decay
                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                # update moment estimate
                m.mul_(beta1).add_((1 - beta1) * grad)
                v.mul_(beta2).add_((1 - beta2) * grad**2)

                # apply weight update
                # p <- p - alpha_t * (m/(sqrt(v)+eps))
                p.addcdiv_(m, torch.sqrt(v) + eps, value=-alpha_t)

                state["t"] += 1

        return loss


def zeropower_via_newtonschulz(
    grad: torch.Tensor,
    ns_coefficients=(3.4445, -4.7750, 2.0315),
    ns_steps: int = 5,
    eps: float = 1e-7,
):
    a, b, c = ns_coefficients
    ortho_grad = grad.to(dtype=torch.bfloat16, copy=True)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # 归一化，确保谱范数 ≤ 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # 执行 NS 迭代
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        nesterov: bool = True,
        ns_steps: int = 5,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not isinstance(ns_steps, int) or ns_steps <= 0:
            raise ValueError(f"ns_steps must be a positive integer: {ns_steps}")
        if len(ns_coefficients) != 3:
            raise ValueError("ns_coefficients must contain exactly three values")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
            ns_coefficients=ns_coefficients,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_coefficients = group["ns_coefficients"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim < 2:
                    raise ValueError(
                        "Muon only supports matrix parameters; use AdamW for vectors such as norm scales and biases"
                    )

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                # Muon uses an exponential moving average of the gradient.
                buf.lerp_(g, 1 - momentum)
                update = g.lerp(buf, momentum) if group["nesterov"] else buf

                original_shape = update.shape
                if update.ndim > 2:
                    update = update.reshape(update.shape[0], -1)

                # 正交化
                update = zeropower_via_newtonschulz(update, ns_coefficients, ns_steps)

                # 形状相关的学习率调整
                update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
                update = update.reshape(original_shape)

                # 权重衰减与更新
                p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for hidden matrices and AdamW for the remaining parameters.

    Parameter groups must include ``use_muon``.  Muon groups use ``momentum``;
    auxiliary Adam groups use ``betas`` and ``eps``.  This keeps both update
    rules and their state in one optimizer, which makes the existing trainer
    and checkpoint format usable without special cases.
    """

    def __init__(self, param_groups):
        if not isinstance(param_groups, list) or not param_groups:
            raise ValueError("param_groups must be a non-empty list")

        normalized_groups = []
        for group in param_groups:
            group = dict(group)
            if "use_muon" not in group:
                raise ValueError("every MuonWithAuxAdam parameter group needs use_muon")
            if not group["params"]:
                continue

            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("nesterov", True)
                group.setdefault("ns_steps", 5)
                group.setdefault("ns_coefficients", (3.4445, -4.7750, 2.0315))
                if not 0.0 <= group["lr"]:
                    raise ValueError(f"Invalid learning rate: {group['lr']}")
                if not 0.0 <= group["momentum"] < 1.0:
                    raise ValueError(f"Invalid momentum value: {group['momentum']}")
                if not 0.0 <= group["weight_decay"]:
                    raise ValueError(f"Invalid weight_decay value: {group['weight_decay']}")
                if not isinstance(group["ns_steps"], int) or group["ns_steps"] <= 0:
                    raise ValueError(f"ns_steps must be a positive integer: {group['ns_steps']}")
                if len(group["ns_coefficients"]) != 3:
                    raise ValueError("ns_coefficients must contain exactly three values")
            else:
                group.setdefault("lr", 1e-3)
                group.setdefault("betas", (0.9, 0.999))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
                beta1, beta2 = group["betas"]
                if not 0.0 <= group["lr"]:
                    raise ValueError(f"Invalid learning rate: {group['lr']}")
                if not 0.0 <= group["eps"]:
                    raise ValueError(f"Invalid epsilon value: {group['eps']}")
                if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
                    raise ValueError(f"Invalid beta parameters: {group['betas']}")
                if not 0.0 <= group["weight_decay"]:
                    raise ValueError(f"Invalid weight_decay value: {group['weight_decay']}")

            normalized_groups.append(group)

        if not normalized_groups:
            raise ValueError("MuonWithAuxAdam received no parameters")
        super().__init__(normalized_groups, {})

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adam_group(group)
        return loss

    def _step_muon_group(self, group):
        momentum = group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.ndim < 2:
                raise ValueError(
                    "Muon only supports matrix parameters; use AdamW for vectors such as norm scales and biases"
                )

            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            buf = state["momentum_buffer"]
            buf.lerp_(p.grad, 1 - momentum)
            update = p.grad.lerp(buf, momentum) if group["nesterov"] else buf

            original_shape = update.shape
            if update.ndim > 2:
                update = update.reshape(update.shape[0], -1)
            update = zeropower_via_newtonschulz(update, group["ns_coefficients"], group["ns_steps"])
            update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
            update = update.reshape(original_shape)

            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update, alpha=-group["lr"])

    def _step_adam_group(self, group):
        beta1, beta2 = group["betas"]
        for p in group["params"]:
            if p.grad is None:
                continue

            state = self.state[p]
            if len(state) == 0:
                state["t"] = 1
                state["m"] = torch.zeros_like(p)
                state["v"] = torch.zeros_like(p)

            t = state["t"]
            m = state["m"]
            v = state["v"]
            alpha_t = group["lr"] * math.sqrt(1 - beta2**t) / (1 - beta1**t)

            if group["weight_decay"] != 0:
                p.mul_(1 - group["lr"] * group["weight_decay"])
            m.mul_(beta1).add_((1 - beta1) * p.grad)
            v.mul_(beta2).add_((1 - beta2) * p.grad.square())
            p.addcdiv_(m, torch.sqrt(v) + group["eps"], value=-alpha_t)
            state["t"] += 1


def get_lr_cosine_schedule(it: int, lr_max: float, lr_min: float, warmup_iters: int, cosine_cycle_iters: int):
    assert it >= 0, "Iteration must be non-negative"
    assert warmup_iters > 0, "Warmup iterations must be positive"
    assert cosine_cycle_iters > 0, "Cosine cycle iterations must be positive"
    assert lr_max > lr_min, "Maximum learning rate must be greater than minimum learning rate"
    assert cosine_cycle_iters > warmup_iters, "Cosine cycle iterations must be greater than warmup iterations"

    if it < warmup_iters:
        lr = it / warmup_iters * lr_max
    elif it <= cosine_cycle_iters:
        lr = lr_min + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (
            lr_max - lr_min
        )
    else:
        lr = lr_min

    return lr


def get_lr_wsd_schedule(
    it: int,
    lr_max: float,
    lr_min: float,
    warmup_iters: int,
    total_iters: int,
    decay_iters: int,
) -> float:
    assert it >= 0, "Iteration must be non-negative"
    assert total_iters > 0, "Total iterations must be positive"
    assert decay_iters > 0, "Decay iterations must be positive"
    assert warmup_iters >= 0, "Warmup iterations must be non-negative"
    assert lr_max > lr_min, "Maximum learning rate must be greater than minimum learning rate"
    assert warmup_iters + decay_iters < total_iters, "WSD requires warmup_iters + decay_iters < total_iters"

    decay_start = total_iters - decay_iters
    if warmup_iters > 0 and it < warmup_iters:
        return it / warmup_iters * lr_max
    if it < decay_start:
        return lr_max
    if it >= total_iters:
        return lr_min
    progress = (it - decay_start) / decay_iters
    return lr_min + 0.5 * (1 + math.cos(progress * math.pi)) * (lr_max - lr_min)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    parameters_with_grad = [p for p in parameters if p.grad is not None]

    total_norm = torch.sqrt(sum((p.grad**2).sum() for p in parameters_with_grad))
    if total_norm > max_l2_norm:
        for p in parameters_with_grad:
            p.grad.mul_(max_l2_norm / (total_norm + eps))
