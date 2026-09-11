import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.llm import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.training import get_batch, load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument_group("model")
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--vocab_size", type=int, default=32768)
    parser.add_argument("--context_length", type=int, default=256)
    # RoPE parameters
    parser.add_argument("--theta", type=float, default=10000.0)

    # data parameters
    parser.add_argument_group("data")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)

    # training parameters
    parser.add_argument_group("training")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=400)
    parser.add_argument("--total_iters", type=int, default=40000)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--log_path",
        type=Path,
        default=None,
        help="optional JSONL path for train/validation metrics (defaults next to the checkpoint)",
    )

    parser.add_argument("--resume", action="store_true")

    #
    parser.add_argument(
        "--ablation", type=str, default="baseline", choices=["baseline", "no_rms", "post_norm", "nope", "silu"]
    )

    return parser.parse_args()


def select_device(device):
    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
        if requested.index is not None and requested.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device does not exist: {device}")
    elif requested.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    elif requested.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {device}")
    return requested


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.manual_seed(seed)
        np.random.seed(seed)


def load_data(path):
    return np.load(path, mmap_mode="r")


def validate_args(args):
    positive = {
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "hidden_size": args.hidden_size,
        "d_ff": args.d_ff,
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "total_iters": args.total_iters,
        "eval_batch_size": args.eval_batch_size,
        "eval_iters": args.eval_iters,
        "save_interval": args.save_interval,
        "eval_interval": args.eval_interval,
        "log_interval": args.log_interval,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {', '.join(invalid)}")
    if args.hidden_size % args.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    if args.learning_rate <= 0 or args.lr_min < 0 or args.learning_rate <= args.lr_min:
        raise ValueError("require learning_rate > lr_min >= 0")
    if args.warmup_iters >= args.total_iters:
        raise ValueError("warmup_iters must be smaller than total_iters")
    if not Path(args.train_path).exists():
        raise FileNotFoundError(args.train_path)
    if not Path(args.valid_path).exists():
        raise FileNotFoundError(args.valid_path)


def train_model(model, batch_data, optimizer, max_grad_norm, eps):
    optimizer.zero_grad()

    x, y = batch_data
    logits = model(x)
    loss = cross_entropy(logits, y)

    loss.backward()
    gradient_clipping(model.parameters(), max_grad_norm, eps)
    optimizer.step()

    return loss


@torch.no_grad()
def eval_model(model, valid_data, eval_iters, eval_batch_size, context_length, device):
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(valid_data, eval_batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def main():
    args = parse_args()
    validate_args(args)

    set_seed(args.seed)
    device = select_device(args.device)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.log_path or checkpoint_path.with_suffix(".jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # baseline (default)
    ablation_kwargs = {
        "use_rmsnorm": True,
        "norm_type": "pre",
        "use_rope": True,
        "ffn_type": "swiglu",
    }
    if args.ablation == "no_rms":
        ablation_kwargs["use_rmsnorm"] = False
    elif args.ablation == "post_norm":
        ablation_kwargs["norm_type"] = "post"
    elif args.ablation == "nope":
        ablation_kwargs["use_rope"] = False
    elif args.ablation == "silu":
        ablation_kwargs["ffn_type"] = "silu"

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        #
        **ablation_kwargs,
    )
    model.to(device)

    # load data
    train_data = load_data(args.train_path)
    valid_data = load_data(args.valid_path)

    # optimizer
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, betas=args.betas, eps=args.eps, weight_decay=args.weight_decay
    )

    start_step = 0
    if args.resume:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"cannot resume; checkpoint does not exist: {checkpoint_path}")
        start_step = load_checkpoint(checkpoint_path, model, optimizer)

    run_started = time.perf_counter()

    def write_metric(step, learning_rate, train_loss, valid_loss=None):
        record = {
            "step": step,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "wall_time_seconds": time.perf_counter() - run_started,
        }
        if valid_loss is not None:
            record["valid_loss"] = valid_loss
        with log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    for step in range(start_step + 1, args.total_iters + 1):
        lr = get_lr_cosine_schedule(step, args.learning_rate, args.lr_min, args.warmup_iters, args.total_iters)

        update_lr(optimizer, lr)

        batch_data = get_batch(train_data, args.batch_size, args.context_length, device)

        loss = train_model(model, batch_data, optimizer, args.max_grad_norm, args.eps)

        train_loss = loss.item()
        valid_loss = None
        if step % args.eval_interval == 0:
            valid_loss = eval_model(
                model,
                valid_data,
                args.eval_iters,
                args.eval_batch_size,
                args.context_length,
                device,
            )
            model.train()

        if step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, checkpoint_path)

        if step % args.log_interval == 0 or valid_loss is not None:
            write_metric(step, lr, train_loss, valid_loss)
            message = f"step={step} train_loss={train_loss:.6f} lr={lr:.6g}"
            if valid_loss is not None:
                message += f" valid_loss={valid_loss:.6f}"
            print(message)

    # Always leave a checkpoint at the requested final iteration, even when
    # total_iters is not an exact multiple of save_interval.
    if start_step < args.total_iters and args.total_iters % args.save_interval != 0:
        save_checkpoint(model, optimizer, args.total_iters, checkpoint_path)
    print(f"saved checkpoint: {checkpoint_path}")
    print(f"saved metrics: {log_path}")


if __name__ == "__main__":
    main()
