from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def create_optimizer_and_scheduler(
    model,
    total_steps: int,
    weight_decay: float = 0.1,
    warmup_proportion: float = 0.1,
    learning_rate: float = 5e-5,
    adam_epsilon: float = 1e-8,
    warmup_steps: int = 0
):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    num_warmup_steps_by_ratio = int(total_steps * warmup_proportion)
    num_warmup_steps_absolute = warmup_steps
    if num_warmup_steps_absolute == 0 or num_warmup_steps_by_ratio == 0:
        num_warmup_steps = max(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    else:
        num_warmup_steps = min(num_warmup_steps_by_ratio, num_warmup_steps_absolute)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    return optimizer, scheduler
