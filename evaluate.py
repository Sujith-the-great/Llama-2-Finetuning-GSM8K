import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import default_data_collator
from torch.optim.lr_scheduler import StepLR
from llama_recipes.utils.memory_utils import MemoryTrace
from transformers.data import DataCollatorForSeq2Seq



def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_loss_f = 0.0
    eval_loss_m = 0.0
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                break
            for key in batch.keys():
                # if is_xpu_available():
                #     batch[key] = batch[key].to('xpu:0')
                # else:
                batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )
# Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_loss_f = eval_loss_f / len(eval_dataloader)
    eval_epoch_loss_m = eval_loss_m / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
# Print evaluation metrics
    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)
    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity