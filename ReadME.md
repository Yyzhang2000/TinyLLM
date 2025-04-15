In this project, we are going to implement a TinyLLM Model based on the most popular LLM architecture (Mixture of Experts) and attention mechanism(MLA). The model is designed to be lightweight and efficient, making it suitable for deployment on edge devices with limited resources.

This will both include pre-training and post-training of the model. And the compressing of the model to improve the inference speed and reduce the model size.

- Pre-Training

  > - [x] Data Preparation
  > - [ ] Model Architecture
  > - [ ] Mixture of Experts
  > - [ ] MLA
  > - [ ] Training Loop
  > - [ ] Optimizer
  > - [ ] Loss Function
  > - [ ] Evaluation

- Post-Training
  > - [ ] Supervised Fine-Tuning
  > - [ ] RLHF
  > - [ ] PPO
  > - [ ] DPO
  > - [ ] GRPO

# Dataset

The Dataset we are using is [`Huggingface FineWeb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb) sample-10BT, which around 10B gpt2 tokens (**27.6GB**)
