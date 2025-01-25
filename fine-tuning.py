"""
1. Libraries:

dataclasses: This library is used to create data classes, which are a way to define structured data with type hints.
wandb: This library is likely used for logging training metrics and visualizing them on the Weights & Biases platform.
typing: This library provides type annotations for functions and variables, improving code readability and maintainability.

2. Data-related Imports:

Dataset: This class from torch.utils.data is used to define custom datasets for training machine learning models.
tqdm: This library provides a progress bar for iterating through data, making training progress more visible.
default_data_collator: This function from the Transformers library is used to collate (combine and arrange) multiple data samples into a batch suitable for model training.
DataCollatorForSeq2Seq: This class (potentially from a custom library llama_recipes) is likely used for specific data collation related to sequence-to-sequence tasks.

3. Optimization and Scheduling:

optim: This module from torch provides various optimizers (like Adam, SGD) for updating model weights during training.
StepLR: This class from torch.optim.lr_scheduler implements a learning rate scheduler that reduces the learning rate at specific intervals (steps).

4. Context Management:

nullcontext: This function from contextlib acts as a context manager that does nothing. It's useful when you need a context manager but don't have any specific actions to perform within it.

5. Other Utilities:

datetime: This module provides functions for working with dates and times, potentially used for logging timestamps.
os: This module provides functions for interacting with the operating system, such as accessing files.
json: This module is used for working with JSON data format.
time: This module provides functions for handling time measurements.
MemoryTrace: This function (potentially from llama_recipes) is likely a custom utility for tracking memory usage during training.

This is a setup for training a large language model or a similar task that involves data loading, optimization, scheduling, and potentially memory monitoring.a setup for training a large language model or a similar task that involves data loading, optimization, scheduling, and potentially memory monitoring.
"""
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import wandb
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import default_data_collator
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from contextlib import nullcontext
from datetime import datetime
from llama_recipes.utils.memory_utils import MemoryTrace
import time
import os
import contextlib
import json
from transformers.data import DataCollatorForSeq2Seq
import numpy as np
import random
from evaluate import evaluation

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Process Using: ", device)

"""
AutoModelForLM is a class from the Hugging Face Transformers library that provides a convenient way to load pre-trained language models (LLMs) designed for causal
language modeling tasks. These models are typically trained on large amounts of text data and are capable of generating human-quality text, translating languages,
 writing different kinds of creative content, and answering your questions in an informative way.
 Automatic Model Selection: The class automatically selects the appropriate model architecture based on the specified model name.
 Pre-trained Weights: It loads pre-trained weights from the Hugging Face Model Hub, saving you the time and computational resources of training a model from scratch.
 Causal Language Modeling: These models are specifically designed for tasks that involve predicting the next token in a sequence, making them suitable for text generation, translation, and other sequential tasks.
 Flexibility: The class offers various configuration options to customize the model's behavior, such as setting the device (CPU or GPU), controlling caching, and specifying the attention implementation.

 Common Use Cases: Text Generation, Machine Translation, Summarization, Question Answering

model_name:   This specifies the name of the pre-trained model you want to load. For example, "gpt2-large" or "bert-base-uncased".
device_map="cuda":   This tells the model to use your GPU for computations if it's available. If you don't have a GPU, you can set it to "cpu".
use_cache=None:   This parameter is used to control whether the model should use cached activations to speed up inference. Setting it to None usually defaults to using caching.
attn_implementation=None:   This parameter specifies the attention implementation to use. If not set, the default implementation will be used. "flash","original", "einsum". Use "flash" for larger models in GPU
"""
model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    use_cache=None,
    attn_implementation=None,
)

"""
Loads a Tokenizer:

AutoTokenizer.from_pretrained(model_name) loads a tokenizer associated with the specified pre-trained language model (model_name).
This tokenizer is responsible for converting text into numerical representations (tokens) that the model can understand.
Sets Pad Token ID:

tokenizer.pad_token_id = tokenizer.eos_token_id sets the padding token ID of the tokenizer to be the same as the end-of-sequence (EOS) token ID.

Padding: When processing sequences of different lengths, it's often necessary to pad them to a uniform length. This is done by adding a special token (padding token) to the end of shorter sequences.
EOS Token: The EOS token is used to indicate the end of a sequence.
Setting Pad Token ID: By setting the padding token ID to the EOS token ID, the model can treat both padding tokens and EOS tokens similarly, which can simplify the training and inference process.
"""
tokenizer = AutoTokenizer.from_pretrained(model_name)

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)

"""
This defines a custom batch sampler class named LengthBasedBatchSampler that inherits from torch.utils.data.BatchSampler.
It creates batches for training a model based on the lengths of the data samples. Here's a breakdown of its functionality:

Initialization ( __init__ method):

data_source: This argument represents the data source, which can be a list of samples or a dictionary-like object containing samples.
batch_size: This specifies the desired number of samples per batch.
drop_last: This boolean flag determines whether to drop the last incomplete batch if the number of samples doesn't perfectly fit the batch size.
shuffle: This boolean flag controls whether to shuffle the order of the data samples before creating batches (default: True).

1. Determining Sample Lengths:

The first checks if the data source elements are dictionaries (e.g., each sample might have multiple keys).

If yes, it extracts the length of the first key's value (assuming all samples have the same keys) to represent the sample length.
If not (data source is a simple list), it directly uses the length of each element.
This approach ensures that batches are created with samples of similar lengths, which can improve training efficiency for certain models, particularly recurrent neural networks (RNNs) that process sequences.

2. Creating Batches ( __iter__ method):

Sort by Length: It sorts the indices of the data source based on the previously determined lengths using np.argsort with the mergesort kind for stability.
Handle Last Batch: If drop_last is True, it removes any leftover samples at the end that wouldn't form a complete batch of size batch_size.
Create Batches: It iterates through the sorted indices and creates batches by slicing the index list into chunks of size batch_size.
Shuffle Batches (Optional): If shuffle is True, it shuffles the order of the created batches to introduce randomness and potentially improve training performance.

3. Length of Sampler ( __len__ method):

It calculates the total number of batches based on the data source length, batch size, and the drop_last parameter.


This LengthBasedBatchSampler helps create batches with similar sample lengths, which can be beneficial for models that process sequences.
It also allows control over dropping incomplete batches and shuffling the batches during training.
"""
class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool=True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

    def __iter__(self):
        ids = np.argsort(self.lengths, kind='mergesort')
        if self.drop_last:
            ids = ids[:len(ids) // self.batch_size * self.batch_size]

        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        if self.shuffle:
            random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)

"""
This defines a function named get_dataloader_kwargs that returns a dictionary of keyword arguments for creating a PyTorch DataLoader.
The function is used to configure the data loading process based on different training configurations and batching strategies.


Setting Batch Size:

The batch_size is determined based on the mode parameter:
If mode is "train", the train_config.batch_size_training is used.
If mode is "val", the train_config.val_batch_size is used.
Batching Strategy:

Padding:

If train_config.batching_strategy is "padding", the function creates a LengthBasedBatchSampler to sort samples by length and create batches with similar lengths.
The DataCollatorForSeq2Seq is used as the collate function to handle padding sequences to the same length.

Packing:

If train_config.batching_strategy is "packing", the batch_size and drop_last parameters are directly set for the DataLoader.
The default_data_collator is used as the collate function, which typically pads sequences to the maximum length in the batch.

Returning Keyword Arguments:

The function returns the kwargs dictionary, which contains the necessary keyword arguments for creating a PyTorch DataLoader.

This function provides a flexible way to configure data loading for different training scenarios.
It allows you to choose between padding and packing strategies and customize the batch size and other parameters based on your specific needs.
"""
def get_dataloader_kwargs(train_config, dataset, tokenizer, mode):
        kwargs = {}
        batch_size = train_config.batch_size_training if mode=="train" else train_config.val_batch_size
        if train_config.batching_strategy == "padding":
            kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode=="train")
            kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
        elif train_config.batching_strategy == "packing":
            kwargs["batch_size"] = batch_size
            kwargs["drop_last"] = True
            kwargs["collate_fn"] = default_data_collator
        else:
            raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")

        return kwargs


"""
This defines a context manager profile that enables profiling or flop counting during the training process.

Context Manager Setup:

use_profiler and use_flop_counter: These flags determine whether to use the PyTorch profiler or a custom flop counter, respectively.
Error Handling: If both flags are set, an error is raised as they are mutually exclusive.

PyTorch Profiler:

Warm-up and Active Phases: The profiler requires a warm-up phase to gather accurate performance data. The wait_step, warmup_step, and active_step parameters control the duration of these phases.
Configuration: Various configuration options are set for the profiler, including:
activities: Specifies the types of activities to profile (CPU and CUDA).
schedule: Defines the timing of the profiling phases.
on_trace_ready: Specifies how to handle the collected traces (e.g., saving to TensorBoard).
profile_memory: Enables memory profiling.
with_stack: Controls whether to include stack traces in the profiling data.
with_flops: Enables flop counting.
record_shapes: Enables recording tensor shapes.
Yielding the Profiler: The yield torch_profiler statement allows the code within the with block to access the profiler object.

Flop Counter:

Minimum Steps: The flop counter requires a minimum number of training steps to provide accurate results.
Yielding the Flop Counter: The yield flop_counter statement allows the code within the with block to access the flop counter object.
Null Context:

If neither profiling nor flop counting is enabled, a null context is used, which essentially does nothing.

"""
@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


"""
The ConcatDataset class is a custom dataset class that concatenates multiple samples into larger chunks.
This is often useful for language models where processing longer sequences can improve performance.


Initialization:

dataset: This is the original dataset that will be concatenated.

chunk_size: This specifies the desired maximum length of each chunk.

A buffer dictionary is initialized to store the current sequence of tokens for each key (input_ids, attention_mask, labels).

Concatenation and Chunking:

The code iterates over the original dataset, appending each sample's tokens to the corresponding key in the buffer.
While the length of any key in the buffer exceeds the chunk_size, a chunk is created by slicing the buffer and appending it to the samples list.
The buffer is then updated to contain the remaining tokens.

Getting Items:

The __getitem__ method allows accessing individual samples from the concatenated dataset. It simply returns the idx-th chunk from the samples list.
Length:

The __len__ method returns the total number of chunks in the concatenated dataset.


The ConcatDataset class takes an original dataset and concatenates its samples into chunks of a specified maximum length.
This can be useful for training language models on longer sequences or for optimizing batching strategies.
"""
class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
            }

        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

def get_preprocessed_dataset(tokenizer):
    # Load dataset from Hugging Face
    dataset = datasets.load_dataset("openai/gsm8k", 'main')
    dataset = dataset["train"]
    dataset = dataset.select(range(100))

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = dataset.train_test_split(test_size=0.1, seed=42).values()  # Split 20% for validation

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + "###Input:\n" + sample["question"] + "\n", add_special_tokens=False)
        labels = tokenizer.encode("###Output:\n" + sample["answer"] + tokenizer.eos_token, add_special_tokens=False)
        sample = {
            "input_ids": prompt + labels,
            "attention_mask": [1] * (len(prompt) + len(labels)),
            "labels": prompt + labels
        }
        return sample

    # Apply the tokenize_add_label function to each dataset
    train_dataset = train_dataset.map(tokenize_add_label, remove_columns=list(train_dataset.features))
    val_dataset = val_dataset.map(tokenize_add_label, remove_columns=list(val_dataset.features))

    return train_dataset, val_dataset

# training function
"""
This defines a function named train that handles the training process for a machine learning model. Here's a breakdown of what it does:

Setup:

Model and Optimizers: The function takes the model, training data loader, evaluation data loader, tokenizer, optimizer, learning rate scheduler, gradient accumulation steps, and training configuration as arguments.
Gradients: A GradScaler is used to manage gradients during mixed precision training.
Logging: Various variables are initialized for tracking and logging metrics like training loss, perplexity, memory usage, and epoch times.

Training Loop:

Epoch Loop: The code iterates over the specified number of epochs (train_config.num_epochs).
Max Steps Check: It checks if the maximum training steps (train_config.max_train_step) have been reached. If so, it stops training.
Epoch Start Time: It records the start time of the current epoch for later timing analysis.
Memory Tracking: A MemoryTrace context is used to track memory usage during training.
Model in Train Mode: The model is set to train mode.
Training Progress Bar: A progress bar is displayed using tqdm to track training progress.
Profiling (Optional): If profiling or flop counting is enabled in the configuration, a profile context manager is used to profile the training process.

Training Step:

Data Batch: The code iterates over each batch in the training data loader.
Max Steps Check: It again checks if the maximum training steps have been reached. If so, it stops training within the loop.
Data to Device: The batch is transferred to the appropriate device (GPU or XPU, if available).
Automatic Mixed Precision (Optional): If mixed precision training is enabled, the code uses an autocast context manager to manage mixed precision calculations.
Forward Pass: The batch is passed through the model to get the model outputs and loss.
Loss Scaling (Optional): The loss is scaled by the gradient accumulation steps.

Backpropagation:

Record Loss (Optional): If saving metrics is enabled, the training step loss and perplexity are recorded.
Backward Pass: The calculated loss is backpropagated through the model to compute gradients.
Gradient Clipping (Optional): If gradient clipping is enabled in the configuration, the gradients are clipped to a specified threshold.
Gradient Accumulation: Gradients are accumulated over multiple steps before performing an optimizer update.
Optimizer Step: After enough gradients are accumulated or the end of the epoch is reached, the optimizer takes a step to update the model weights.
Zero Gradients: The optimizer's gradients are zeroed out for the next batch.
Progress Bar Update: The progress bar is updated to reflect training progress.

Evaluation (Optional):

Periodic Evaluation: After every specified number of training steps (usually every epoch), the model is evaluated on the validation data loader using the evaluation function.
Logging Results (Optional): If saving metrics is enabled, the evaluation loss and perplexity for each step within the epoch are recorded.

Saving Model:

Best Model Saving: After each epoch, the model with the best validation loss is saved as the "best_model".

Logging and Metrics:

Epoch Metrics: The training perplexity, loss, and epoch time are printed and averaged for the entire epoch.
Metrics Saving (Optional): If saving metrics is enabled, all the recorded training and validation metrics are saved to a JSON file.
Memory Cleanup: Finally, PyTorch's cache is cleared.


The function calculates and returns a dictionary containing average training loss, perplexity, validation loss, perplexity, epoch time, checkpointing time, and (optionally) the metrics filename.
"""
def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, wandb_run=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """

    local_rank=None
    rank=None

    autocast = nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        # stop when the maximum number of training steps is reached

        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_loss_f = 0.0
            total_loss_m = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    # print(f"The keys in the batch are:{batch.keys()}")
                    for key in batch.keys():
                          batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        outputs = model(**batch)
                        loss = outputs.loss

                    loss = loss / gradient_accumulation_steps

                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))

                    total_loss += loss.detach().float()
                    loss.backward()

                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                        optimizer.step()
                        optimizer.zero_grad()
                        pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        wandb_run.log({
                            'train/epoch': epoch + 1,
                            'train/step': epoch * len(train_dataloader) + step,
                            'train/loss': loss.detach().float(),
                        })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if step % 50 == 0:
                        if train_config.run_validation:
                            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                            if train_config.save_metrics:
                                val_step_loss.extend(temp_val_loss)
                                val_step_perplexity.extend(temp_step_perplexity)
                        model.train()

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))
        memtrace.print_stats()
        lr_scheduler.step()

        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()

            if train_config.save_model :
                epoch_dir = os.path.join(train_config.output_dir, f"epoch{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)
                # Save the model in the new directory
                model.save_pretrained(epoch_dir)
                print(f"Model is saved in {epoch_dir} directory")

            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    return results


@dataclass
class train_configy:
    model_name: str=model_name
    tokenizer_name: str=None
    run_validation: bool=True
    batch_size_training: int=1
    batching_strategy: str="packing" #alternative: padding
    context_length: int=128
    gradient_accumulation_steps: int=4
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=1 # set hoe many epoch to train
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    mixed_precision: bool=True
    val_batch_size: int=1
    output_dir: str = "test_model"
    save_model: bool = True
    save_metrics: bool = True # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "test_model" # will be used if using profiler


if __name__ == "__main__":

    train_config = train_configy()
    dataset_train, dataset_val = get_preprocessed_dataset(
    tokenizer
    )

    dataset_train, dataset_val = get_preprocessed_dataset(
    tokenizer
    )

    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    print(type(train_dataloader))
    print(train_dataloader)

    eval_dataloader = None
    
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
    
    # set optimizer to SGD
    optimizer = optim.SGD(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Initialize W&B run
    wandb_run = wandb.init(
        project="Fine-Tuning-LLama2-7B",   
        name="experiment_epoch_5",    
        config=train_config               
    )

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        wandb_run=wandb_run,
    )
    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

