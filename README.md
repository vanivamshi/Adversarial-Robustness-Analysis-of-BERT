# Adversarial Robustness Analysis of BERT

A comprehensive study of BERT/RoBERTa vulnerabilities to adversarial attacks using semantic-preserving text transformations.

## Overview

This project implements a complete pipeline for:
1. Training a BERT on SST-2 sentiment classification
2. Creating structured adversarial attacks using synonym substitution
3. Analyzing model behavior changes (attention patterns, confidence)
4. Proposing defense mechanisms

## Problem Statement

Transformer models perform well on classification tasks but can behave unpredictably when exposed to adversarial inputs. This project demonstrates:
- How easily BERT can be fooled by meaning-preserving text perturbations
- What changes occur in the model's attention mechanisms
- How confidence can remain high even when predictions are wrong

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Set Hugging Face token if needed (rarely required for public models)
# export HUGGINGFACE_HUB_TOKEN="your_token_here"
# OR
python download_models.py

# Run the analysis
python main.py
```
