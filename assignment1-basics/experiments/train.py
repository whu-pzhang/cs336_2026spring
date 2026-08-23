import argparse

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
    parser.add_argument("--seed", type=int, default=42, required=True)

    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


def select_device(device):
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


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
    for i in range(eval_iters):
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

    set_seed(args.seed)
    device = select_device(args.device)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
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
        start_step = load_checkpoint(args.checkpoint_path, model, optimizer)

    for step in range(start_step + 1, args.total_iters + 1):
        lr = get_lr_cosine_schedule(step, args.learning_rate, args.lr_min, args.warmup_iters, args.total_iters)

        update_lr(optimizer, lr)

        batch_data = get_batch(train_data, args.batch_size, args.context_length, device)

        loss = train_model(model, batch_data, optimizer, args.max_grad_norm, args.eps)

        if step % args.log_interval == 0:
            print(f"Step {step} loss: {loss.item()}")

        if step % args.save_interval == 0:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)

        if step % args.eval_interval == 0:
            loss_eval = eval_model(
                model,
                valid_data,
                args.eval_iters,
                args.eval_batch_size,
                args.context_length,
                device,
            )
            print(f"Step {step} eval loss: {loss_eval}")
            model.train()


if __name__ == "__main__":
    main()
