# Llama-2 Fine-Tuning on GSM8K Dataset

This repository contains the code and resources for fine-tuning the **meta-llama/Llama-2-7b-hf** model on the **GSM8K** dataset. The project focuses on supervised fine-tuning of the Llama-2 model to solve math word problems, followed by evaluation on the GSM8K test set using exact match accuracy.

## Key Features

- **Supervised Fine-Tuning**: Fine-tuning the Llama-2 model on the GSM8K training set.
- **Evaluation**: Evaluating the model's performance on the GSM8K test set using exact match accuracy.
- **Weights & Biases Integration**: Logging and visualization of training and validation metrics using WandB.
- **Checkpointing**: Saving model checkpoints to ensure the best-performing model can be recovered.
- **Gradient Accumulation**: Efficient training with gradient accumulation to handle memory constraints.

## Script Structure

- **Training Script**: Contains the code for fine-tuning the Llama-2 model, including data loading, model training, and evaluation.
- **Evaluation Script**: Script for evaluating the fine-tuned model on the GSM8K test set.
- **Checkpoints**: Saved model checkpoints for recovery and further analysis.
- **WandB Logs**: Visualizations of training and validation loss curves, perplexity, and other metrics.

## Usage

### Clone the Repository
```
git clone https://github.com/your-username/Llama-2-Finetuning-GSM8K.git
cd Llama-2-Finetuning-GSM8K
```

## Key Components Explained
**1. Perplexity Calculation**
Perplexity is used to evaluate the model's performance. It measures how well the model predicts the next word in a sequence. Lower perplexity indicates better performance.

Perplexity
=
exp
⁡
(
−
1
N
∑
i
=
1
N
log
⁡
p
(
x
i
)
)
Perplexity=exp(− 
N
1
​
  
i=1
∑
N
​
 logp(x 
i
​
 ))
**2. Gradient Accumulation**
To handle memory constraints, gradient accumulation is used. This allows the model to process smaller batches and accumulate gradients over multiple steps before updating the model weights.

**3. Checkpointing**
Model checkpoints are saved after each epoch if the validation loss improves. This ensures that the best-performing model can be recovered even if training is interrupted.

**4. Weights & Biases Integration**
Training and validation metrics (loss, perplexity) are logged to WandB for real-time visualization and tracking.

## Results
**Training Loss**: The training loss curve shows the model's performance over epochs.

**Validation Loss**: The validation loss curve helps in understanding the model's generalization ability.

**Perplexity**: Perplexity metrics provide insights into the model's prediction certainty.

**Accuracy**: The model's accuracy on the GSM8K test set is reported based on exact match of the final answers.

## Team Contributions
**Chao-Shiang, Chen**: Perplexity calculation and WandB integration.

**Ankitha Dongerkerry Pai**: Data loading and hyperparameter tuning.

**Rakshita Madhavan**: Evaluation metrics and checkpointing.

**Bala Sujith Potineni**: Training loop implementation and gradient accumulation.

**Suraj Kumar Manylal**: Training configuration and loss curve analysis.

## References
**Llama-2 Model**

**GSM8K Dataset**

**PyTorch Documentation**

## License
This project is licensed under the MIT License. See the LICENSE file for details.
