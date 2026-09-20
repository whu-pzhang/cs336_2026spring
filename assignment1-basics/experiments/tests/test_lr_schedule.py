import math

from cs336_basics.optimizer import get_lr_wsd_schedule


def test_wsd_warmup_stable_decay_boundaries():
    lr_max = 1e-3
    lr_min = 1e-4
    warmup = 4
    total = 20
    decay = 5
    decay_start = total - decay

    assert get_lr_wsd_schedule(0, lr_max, lr_min, warmup, total, decay) == 0.0
    assert get_lr_wsd_schedule(warmup, lr_max, lr_min, warmup, total, decay) == lr_max
    assert get_lr_wsd_schedule(decay_start - 1, lr_max, lr_min, warmup, total, decay) == lr_max
    assert get_lr_wsd_schedule(decay_start, lr_max, lr_min, warmup, total, decay) == lr_max
    assert get_lr_wsd_schedule(total, lr_max, lr_min, warmup, total, decay) == lr_min

    mid = decay_start + decay // 2
    expected = lr_min + 0.5 * (1 + math.cos((mid - decay_start) / decay * math.pi)) * (lr_max - lr_min)
    assert get_lr_wsd_schedule(mid, lr_max, lr_min, warmup, total, decay) == expected
