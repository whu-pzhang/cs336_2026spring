import torch


def decode_one_step(logits, temperature: float = 0.9, top_p: float = 0.9):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # make sure the first token is not removed
    sorted_indices_to_remove[..., 0] = False
    # Shift mask right by one (keep the first token that exceeds top_p)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()

    sorted_probs[sorted_indices_to_remove] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_token)
    return next_token


@torch.no_grad()
def decode_sequence(
    model,
    initial_tokens,
    eos_token_id: int,
    max_length: int = 256,
    temperature: float = 0.9,
    top_p: float = 0.9,
):

    assert temperature > 0, "Temperature must be greater than 0"
    assert 0 < top_p <= 1, "top_p must be in the range (0, 1]"
    assert max_length >= 0, "max_length must be non-negative"

    model.eval()
    generated_tokens = initial_tokens.clone()
    batch_size = generated_tokens.size(0)
    device = generated_tokens.device
    context_length = model.context_length

    finished_mask = generated_tokens[:, -1] == eos_token_id

    if finished_mask.all():
        return generated_tokens

    for _ in range(max_length):
        if finished_mask.all():
            break

        # indices of sequences still in progress
        unfinished_indices = (~finished_mask).nonzero(as_tuple=True)[0]

        # forward only for unfinished sequences
        input_seq = generated_tokens[unfinished_indices, -context_length:].clone()
        logits = model(input_seq)
        next_token_unfinished = decode_one_step(logits[:, -1, :], temperature, top_p)

        # Create next tokens for the whole batch, default to eos_token_id for finished sequences
        next_token = torch.full((batch_size, 1), eos_token_id, dtype=generated_tokens.dtype, device=device)
        next_token[unfinished_indices] = next_token_unfinished

        # Update finished mask(only for unfinished)
        new_finished = next_token_unfinished.squeeze(-1) == eos_token_id
        finished_mask[unfinished_indices] |= new_finished

        generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

    return generated_tokens


if __name__ == "__main__":
    from pathlib import Path

    from tokenize_datasets import load_tokenizer
    from trainer.builder import build_model
    from trainer.config import load_model_config

    run_dir = Path("experiments/artifacts/tinystories_lm")
    device = "cuda"

    m = build_model(load_model_config(run_dir / "config.json"))
    state_dict = torch.load(run_dir / "ckpt.pt", map_location="cpu")["model_state"]
    m.load_state_dict(state_dict)
    m.to(device)
    m.eval()

    tokenizer = load_tokenizer(dataset="tinystories")
    eos_token_id = tokenizer.special_to_id["<|endoftext|>"]

    prompt = "Once upon a time"
    encoded_prompt = tokenizer.encode(prompt)
    init_tokens = torch.tensor(encoded_prompt, device=device).unsqueeze(0)

    # compare different temperature values
    for temperature in [0.3, 0.9, 1.2]:
        print(f"temperature: {temperature}")
        # no eos token to ensure generation continues until max_length is reached
        decoded_tokens = decode_sequence(
            m,
            init_tokens,
            eos_token_id=512,
            temperature=temperature,
            max_length=256,
        )

        print(f"prompt length: {len(encoded_prompt)}")
        print(f"total length: {len(decoded_tokens[0].tolist())}")
        print(f"generated length: {len(decoded_tokens[0].tolist()) - len(encoded_prompt)}")

        print(tokenizer.decode(decoded_tokens[0].tolist()))
