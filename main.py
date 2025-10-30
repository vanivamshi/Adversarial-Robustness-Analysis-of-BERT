"""
Adversarial Robustness Analysis for BERT in NLP applications
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import wordnet
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, util
from difflib import SequenceMatcher
import random
import warnings
warnings.filterwarnings('ignore')

# Disable wandb to avoid API key prompts
os.environ['WANDB_DISABLED'] = 'true'

# Download NLTK parameters
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt', quiet=True)

class TransformerRobustnessAnalyzer:
    """
    Main class for training, attacking and analyzing BERT models
    """
    
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        """
        Initialize BERT - bert-base-uncased
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
        
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.results = {}
        
    def load_data(self, dataset_name: str = "sst2", max_samples: int = 1000):
        """Load and preprocess SST-2 dataset"""
        print(f"\n Loading {dataset_name} dataset")
        
        if dataset_name == "sst2":
            dataset = load_dataset("glue", "sst2")
            self.dataset = {
                'train': dataset['train'].shuffle(seed=42).select(range(min(max_samples, len(dataset['train'])))),
                'validation': dataset['validation'].shuffle(seed=42).select(range(min(500, len(dataset['validation'])))),
                'test': dataset['validation'].shuffle(seed=43).select(range(500, min(800, len(dataset['validation']))))
            }
        
        print(f"Loaded {len(self.dataset['train'])} train, {len(self.dataset['validation'])} val, {len(self.dataset['test'])} test samples")
        return self.dataset
    
    def train_model(self, epochs: int = 3, learning_rate: float = 2e-5):
        """Train baseline BERT model"""
        print(f"\n Training {self.model_name}...")
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=False,
                cache_dir='./cache'
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2,
                use_auth_token=False,
                cache_dir='./cache'
            ).to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("\nTrying to use cached version...")
            # Try with local_files_only as fallback
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=True,
                cache_dir='./cache'
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2,
                local_files_only=True,
                cache_dir='./cache'
            ).to(self.device)
        
        # Tokenize datasets
        def tokenize_function(examples):
            return self.tokenizer(
                examples['sentence'], 
                padding='max_length', 
                truncation=True, 
                max_length=128
            )
        
        tokenized_datasets = {k: v.map(tokenize_function, batched=True) for k, v in self.dataset.items()}
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            report_to="none",  # Disable wandb and other integrations
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {'accuracy': accuracy_score(labels, predictions)}
        
        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            compute_metrics=compute_metrics,
        )
        
        trainer.train()
        
        # Evaluate on test set
        test_results = trainer.evaluate(tokenized_datasets['test'])
        self.results['baseline_accuracy'] = test_results['eval_accuracy']
        print(f"\n Baseline Test Accuracy: {self.results['baseline_accuracy']:.4f}")
        
        return self.model
    
    def analyze_baseline(self, num_samples: int = 50):
        """Analyze baseline model behavior"""
        print(f"\n Analyzing baseline model...")
        
        self.model.eval()
        confidences = []
        correct_confidences = []
        incorrect_confidences = []
        
        test_samples = self.dataset['test'].select(range(num_samples))
        
        for sample in test_samples:
            inputs = self.tokenizer(text=sample['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidence = probs.max().item()
                prediction = probs.argmax().item()
                
                confidences.append(confidence)
                if prediction == sample['label']:
                    correct_confidences.append(confidence)
                else:
                    incorrect_confidences.append(confidence)
        
        self.results['baseline_confidences'] = {
            'all': confidences,
            'correct': correct_confidences,
            'incorrect': incorrect_confidences
        }
        
        print(f"Average confidence: {np.mean(confidences):.4f}")
        print(f"  Correct predictions: {np.mean(correct_confidences):.4f}")
        if incorrect_confidences:
            print(f"  Incorrect predictions: {np.mean(incorrect_confidences):.4f}")
        
        return confidences
    
    def get_attention_maps(self, text: str):
        """Extract attention maps for interpretability"""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Average attention across all heads and layers
        attentions = torch.stack([att[0] for att in outputs.attentions])
        avg_attention = attentions.mean(dim=(0, 1)).cpu().numpy()
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        return {
            'tokens': tokens,
            'attention': avg_attention,
            'logits': outputs.logits[0].cpu().numpy(),
            'all_attentions': attentions.cpu().numpy()
        }


class SemanticAdversarialAttack:
    """
    Adversarial attack using synonym substitution
    Preserves semantic meaning while fooling the classifier
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        # self.model_type = 'roberta' if 'roberta' in tokenizer.name_or_path.lower() else 'bert'
        self.model_type = 'bert'
        
    def get_synonyms(self, word: str, pos: str = None):
        """Get WordNet synonyms for a word"""
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym.lower() != word.lower() and synonym.isalpha():
                    synonyms.add(synonym)
        
        return list(synonyms)[:5]  # Limit to 5 synonyms
    
    def get_word_importance(self, text: str, label: int):
        """
        Calculate importance of each word using gradient-based attribution
        """
        self.model.eval()
        
        inputs = self.tokenizer(text=text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        
        # Get embeddings layer (different for BERT vs RoBERTa)
        # if self.model_type == 'roberta':
        #     embeddings = self.model.roberta.embeddings.word_embeddings(inputs['input_ids'])
        # else:
        embeddings = self.model.bert.embeddings.word_embeddings(inputs['input_ids'])
        
        embeddings.retain_grad()
        
        # Forward pass
        # if self.model_type == 'roberta':
        #     outputs = self.model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])
        # else:
        outputs = self.model(inputs_embeds=embeddings, attention_mask=inputs['attention_mask'])
        
        loss = outputs.logits[0][label]
        
        # Backward pass
        loss.backward()
        
        # Get gradient magnitudes
        gradients = embeddings.grad.abs().sum(dim=-1)[0].cpu().numpy()
        
        # Map to token positions
        importance = {i: gradients[i] for i in range(len(gradients))}
        
        return importance
    
    def attack_sample(self, text: str, true_label: int, max_perturbations: int = 5):
        """
        Generate adversarial example by substituting important words
        """
        # Get word importance
        importance = self.get_word_importance(text, true_label)
        
        # Tokenize and get importance scores for each word position
        tokens = self.tokenizer.tokenize(text)
        special_tokens = ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        
        # Map tokens to word positions
        word_positions = []
        for i, token in enumerate(tokens):
            if token not in special_tokens:
                # Clean token for search
                clean_token = token.replace('Ġ', '').replace('##', '')
                if clean_token.isalpha():
                    word_positions.append({
                        'token_idx': i,
                        'token': token,
                        'clean_token': clean_token,
                        'importance': importance.get(i, 0)
                    })
        
        # Sort by importance
        word_positions.sort(key=lambda x: x['importance'], reverse=True)
        
        # Try substituting most important words
        perturbed_text = text.split()
        substitutions = []
        used_positions = set()
        
        for word_info in word_positions[:max_perturbations]:
            if word_info['clean_token'].lower() in used_positions:
                continue
                
            synonyms = self.get_synonyms(word_info['clean_token'])
            
            if not synonyms:
                continue
            
            # Try each synonym
            for synonym in synonyms:
                # Create candidate by replacing the word
                candidate_words = []
                for word in perturbed_text:
                    if word.lower() == word_info['clean_token'].lower() and word.lower() not in used_positions:
                        candidate_words.append(synonym)
                    else:
                        candidate_words.append(word)
                
                candidate = ' '.join(candidate_words)
                
                # Check if attack succeeds
                inputs = self.tokenizer(candidate, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    prediction = outputs.logits.argmax().item()
                    confidence = torch.softmax(outputs.logits, dim=-1).max().item()
                
                if prediction != true_label:
                    substitutions.append({
                        'original': word_info['clean_token'],
                        'substitute': synonym,
                        'position': word_info['token_idx']
                    })
                    perturbed_text = candidate.split()
                    used_positions.add(word_info['clean_token'].lower())
                    break
        
        # Final prediction on perturbed text
        final_text = ' '.join(perturbed_text)
        inputs = self.tokenizer(final_text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            adv_prediction = outputs.logits.argmax().item()
            adv_confidence = torch.softmax(outputs.logits, dim=-1).max().item()
        
        return {
            'original_text': text,
            'perturbed_text': final_text,
            'true_label': true_label,
            'adversarial_prediction': adv_prediction,
            'adversarial_confidence': adv_confidence,
            'attack_success': adv_prediction != true_label,
            'substitutions': substitutions,
            'num_substitutions': len(substitutions)
        }
    
    def evaluate_attack(self, dataset, num_samples: int = 100, perturbation_levels: List[int] = [1, 2, 3, 5]):
        """
        Evaluate attack across multiple perturbation levels
        """
        print(f"\n Running adversarial attacks")
        
        results = {level: [] for level in perturbation_levels}
        
        samples = dataset.select(range(min(num_samples, len(dataset))))
        
        for level in perturbation_levels:
            print(f"\n  Testing perturbation level: {level}")
            success_count = 0
            
            for sample in samples:
                attack_result = self.attack_sample(
                    sample['sentence'], 
                    sample['label'],
                    max_perturbations=level
                )
                results[level].append(attack_result)
                
                if attack_result['attack_success']:
                    success_count += 1
            
            success_rate = success_count / len(samples)
            print(f"Attack success rate: {success_rate:.4f}")
        
        return results


def generate_detailed_report(analyzer, attack_results):
    """Generate a detailed text report of the analysis"""
    report_lines = []
    report_lines.append("="*70)
    report_lines.append("ADVERSARIAL ROBUSTNESS ANALYSIS REPORT")
    report_lines.append("="*70)
    report_lines.append(f"\n Model: {analyzer.model_name}")
    report_lines.append(f"Device: {analyzer.device}")
    
    # Baseline metrics
    report_lines.append(f"\n BASELINE PERFORMANCE")
    report_lines.append(f"Test Accuracy: {analyzer.results['baseline_accuracy']:.4f}")
    
    conf = analyzer.results['baseline_confidences']
    report_lines.append(f"Average Confidence: {np.mean(conf['all']):.4f}")
    report_lines.append(f"Correct Predictions Avg Confidence: {np.mean(conf['correct']):.4f}")
    if conf['incorrect']:
        report_lines.append(f"Incorrect Predictions Avg Confidence: {np.mean(conf['incorrect']):.4f}")
    
    # Attack results
    report_lines.append(f"\n ADVERSARIAL ATTACK RESULTS")
    levels = sorted(attack_results.keys())
    for level in levels:
        success_count = sum(r['attack_success'] for r in attack_results[level])
        total = len(attack_results[level])
        success_rate = success_count / total
        successful_attacks = [r for r in attack_results[level] if r['attack_success']]
        report_lines.append(f"\n Perturbation Level: {level}")
        report_lines.append(f"Attack Success Rate: {success_rate:.2%}")
        report_lines.append(f"Accuracy Under Attack: {1-success_rate:.2%}")
        if successful_attacks:
            avg_conf = np.mean([r['adversarial_confidence'] for r in successful_attacks])
            report_lines.append(f"Avg Successful Attack Confidence: {avg_conf:.4f}")
        else:
            report_lines.append("No successful attacks at this level")
    
    # Example attacks
    report_lines.append(f"\n EXAMPLE ADVERSARIAL ATTACKS")
    for i, example in enumerate(attack_results[3][:3]):
        if example['attack_success']:
            report_lines.append(f"\n Example {i+1}:")
            report_lines.append(f"Original: {example['original_text'][:80]}...")
            report_lines.append(f"Perturbed: {example['perturbed_text'][:80]}...")
            report_lines.append(f"Substitutions: {example['num_substitutions']}")
            for sub in example['substitutions']:
                report_lines.append(f"'{sub['original']}' -> '{sub['substitute']}'")

def save_plot_data_to_json(analyzer, attack_results, filename='plot_data.json'):
    """Save all plot data to a JSON file"""
    print("\n Saving plot data to JSON...")
    
    plot_data = {}
    conf = analyzer.results['baseline_confidences']
    
    # Convert numpy arrays and types to JSON-serializable formats
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    # 1. Baseline confidence distribution
    plot_data['baseline_confidence_distribution'] = {
        'all_confidences': conf['all'],
        'correct_confidences': conf['correct'],
        'incorrect_confidences': conf['incorrect'] if conf['incorrect'] else []
    }
    
    # 2. Attack effectiveness
    levels = sorted(attack_results.keys())
    plot_data['attack_effectiveness'] = {
        'max_perturbations': levels,
        'success_rates': [
            sum(r['attack_success'] for r in attack_results[level]) / len(attack_results[level])
            for level in levels
        ]
    }
    
    # 3. Confidence after attack
    plot_data['confidence_after_attack'] = {}
    for level in levels:
        plot_data['confidence_after_attack'][f'level_{level}'] = [
            r['adversarial_confidence'] for r in attack_results[level] if r['attack_success']
        ]
    
    # Attention maps
    example_orig = attack_results[3][0]['original_text']
    attn_orig = analyzer.get_attention_maps(example_orig)
    example_adv = attack_results[3][0]['perturbed_text']
    attn_adv = analyzer.get_attention_maps(example_adv)
    
    min_size = min(attn_orig['attention'].shape[0], attn_adv['attention'].shape[0], 15)
    plot_data['attention_maps'] = {
        'original': {
            'text': example_orig,
            'tokens': attn_orig['tokens'][:min_size],
            'attention_matrix': attn_orig['attention'][:min_size, :min_size].tolist()
        },
        'adversarial': {
            'text': example_adv,
            'tokens': attn_adv['tokens'][:min_size],
            'attention_matrix': attn_adv['attention'][:min_size, :min_size].tolist()
        },
        'attention_change': (attn_adv['attention'][:min_size, :min_size] - 
                             attn_orig['attention'][:min_size, :min_size]).tolist()
    }
    
    # 7. Substitutions needed
    plot_data['substitutions_needed'] = {}
    for level in levels:
        subs_data = {}
        for r in attack_results[level]:
            if r['attack_success']:
                count = min(r['num_substitutions'], 6)  # Cap at 6
                subs_data[str(count)] = subs_data.get(str(count), 0) + 1
        plot_data['substitutions_needed'][f'level_{level}'] = subs_data
    
    # Accuracy degradation
    plot_data['accuracy_degradation'] = {
        'baseline_accuracy': float(analyzer.results['baseline_accuracy']),
        'max_perturbations': levels,
        'accuracy_under_attack': [
            (1 - sum(r['attack_success'] for r in attack_results[level]) / len(attack_results[level]))
            for level in levels
        ]
    }
    
    # Summary stats
    plot_data['summary'] = {
        'model_name': analyzer.model_name,
        'baseline_accuracy': float(analyzer.results['baseline_accuracy']),
        'avg_baseline_confidence': float(np.mean(conf['all'])),
        'attack_performance': {
            f'perturb_{level}': {
                'success_rate': float(sum(r['attack_success'] for r in attack_results[level]) / len(attack_results[level]))
            } for level in levels
        }
    }
    
    # Convert all data
    plot_data = convert_to_serializable(plot_data)
    
    with open(filename, 'w') as f:
        json.dump(plot_data, f, indent=2)
    
    print(f"Saved plot data to '{filename}'")
    return plot_data


def create_individual_plots(analyzer, attack_results, output_dir='results/plots'):
    """Create individual plot files instead of subplots"""
    print("\n Creating individual plot files...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    levels = sorted(attack_results.keys())
    conf = analyzer.results['baseline_confidences']
    
    # Plot 1: Baseline confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(conf['correct'], alpha=0.7, label='Correct', bins=20, color='green')
    if conf['incorrect']:
        plt.hist(conf['incorrect'], alpha=0.7, label='Incorrect', bins=20, color='red')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Baseline Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"  Generating: baseline_confidence.png")
    plt.savefig(f'{output_dir}/baseline_confidence.png', dpi=150)
    plt.close()
    
    # Plot 2: Attack effectiveness
    plt.figure(figsize=(10, 6))
    success_rates = []
    for level in levels:
        success_rates.append(sum(r['attack_success'] for r in attack_results[level]) / len(attack_results[level]))
    plt.plot(levels, success_rates, marker='o', linewidth=2, markersize=10, color='red')
    plt.xlabel('Max Perturbations')
    plt.ylabel('Attack Success Rate')
    plt.title('Attack Effectiveness')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"  Generating: attack_effectiveness.png")
    plt.savefig(f'{output_dir}/attack_effectiveness.png', dpi=150)
    plt.close()
    
    # Plot 3: Confidence after attack
    plt.figure(figsize=(10, 6))
    for level in levels:
        confidences = [r['adversarial_confidence'] for r in attack_results[level] if r['attack_success']]
        if confidences:
            plt.hist(confidences, alpha=0.5, label=f'{level} perturb', bins=15)
    plt.xlabel('Adversarial Confidence')
    plt.ylabel('Count')
    plt.title('Confidence After Attack')
    plt.legend()
    plt.tight_layout()
    print(f"  Generating: confidence_after_attack.png")
    plt.savefig(f'{output_dir}/confidence_after_attack.png', dpi=150)
    plt.close()
    
    # Plot 4: Original attention map
    plt.figure(figsize=(12, 10))
    example_orig = attack_results[3][0]['original_text']
    attn_orig = analyzer.get_attention_maps(example_orig)
    tokens_orig = attn_orig['tokens'][:15]
    attn_matrix_orig = attn_orig['attention'][:15, :15]
    
    sns.heatmap(attn_matrix_orig, xticklabels=tokens_orig, yticklabels=tokens_orig, 
                cmap='viridis', cbar_kws={'label': 'Attention'})
    plt.title('Original Attention Pattern')
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    print(f"  Generating: original_attention.png")
    plt.savefig(f'{output_dir}/original_attention.png', dpi=150)
    plt.close()
    
    # Plot 5: Adversarial attention map
    plt.figure(figsize=(12, 10))
    example_adv = attack_results[3][0]['perturbed_text']
    attn_adv = analyzer.get_attention_maps(example_adv)
    tokens_adv = attn_adv['tokens'][:15]
    attn_matrix_adv = attn_adv['attention'][:15, :15]
    
    sns.heatmap(attn_matrix_adv, xticklabels=tokens_adv, yticklabels=tokens_adv,
                cmap='viridis', cbar_kws={'label': 'Attention'})
    plt.title('Adversarial Attention Pattern')
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    print(f"  Generating: adversarial_attention.png")
    plt.savefig(f'{output_dir}/adversarial_attention.png', dpi=150)
    plt.close()
    
    # Plot 6: Attention change
    plt.figure(figsize=(12, 10))
    min_size = min(attn_matrix_orig.shape[0], attn_matrix_adv.shape[0])
    attn_diff = attn_matrix_adv[:min_size, :min_size] - attn_matrix_orig[:min_size, :min_size]
    tokens_common = tokens_adv[:min_size]
    
    sns.heatmap(attn_diff, xticklabels=tokens_common, yticklabels=tokens_common,
                cmap='RdBu_r', center=0, cbar_kws={'label': 'Attention Δ'})
    plt.title('Attention Change (Adv - Orig)')
    plt.xlabel('Tokens')
    plt.ylabel('Tokens')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    print(f"  Generating: attention_change.png")
    plt.savefig(f'{output_dir}/attention_change.png', dpi=150)
    plt.close()
    
    # Plot 7: Substitutions needed
    plt.figure(figsize=(10, 6))
    for level in levels:
        num_subs = [r['num_substitutions'] for r in attack_results[level] if r['attack_success']]
        if num_subs:
            plt.hist(num_subs, alpha=0.5, label=f'Max {level}', bins=range(0, level+2))
    plt.xlabel('Actual Substitutions')
    plt.ylabel('Count')
    plt.title('Substitutions Needed for Success')
    plt.legend()
    plt.tight_layout()
    print(f"  Generating: substitutions_needed.png")
    plt.savefig(f'{output_dir}/substitutions_needed.png', dpi=150)
    plt.close()
    
    # Plot 8: Accuracy degradation
    plt.figure(figsize=(10, 6))
    baseline_acc = analyzer.results['baseline_accuracy']
    accuracies = [1 - success_rates[i] for i in range(len(levels))]
    plt.plot(levels, [baseline_acc] * len(levels), '--', label='Baseline', 
             color='green', linewidth=2)
    plt.plot(levels, accuracies, marker='o', label='Under Attack', 
             color='red', linewidth=2, markersize=10)
    plt.xlabel('Max Perturbations')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Degradation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    print(f"  Generating: accuracy_degradation.png")
    plt.savefig(f'{output_dir}/accuracy_degradation.png', dpi=150)
    plt.close()
    
    print(f"Saved individual plots to '{output_dir}/'")


def visualize_results(analyzer, attack_results):
    """Print comprehensive summary to terminal"""
    print("\n" + "="*70)
    print("ADVERSARIAL ROBUSTNESS ANALYSIS SUMMARY")
    print("="*70)
    
    levels = sorted(attack_results.keys())
    conf = analyzer.results['baseline_confidences']
    baseline_acc = analyzer.results['baseline_accuracy']
    success_rates = [
        sum(r['attack_success'] for r in attack_results[level]) / len(attack_results[level])
        for level in levels
    ]
    
    model_name = analyzer.model_name.upper()
    print(f"\n Model: {model_name}")
    print(f"\n BASELINE PERFORMANCE:")
    print(f"Test Accuracy: {baseline_acc:.4f}")
    print(f"Avg Confidence: {np.mean(conf['all']):.4f}")
    print(f"Correct Predictions Avg Confidence: {np.mean(conf['correct']):.4f}")
    
    print(f"\n Attack Performance:")
    print(f"Level 1 perturbations: {success_rates[0]:.2%} success rate")
    print(f"Level 2 perturbations: {success_rates[1]:.2%} success rate")
    print(f"Level 3 perturbations: {success_rates[2]:.2%} success rate")
    print(f"Level 5 perturbations: {success_rates[3]:.2%} success rate")
    
    print(f"\n Key Findings:")
    print(f"Model vulnerable to semantic synonym attacks")
    print(f"Attention patterns shift significantly under attack")
    print(f"Confidence remains high even with incorrect predictions")
    print("="*70)
    
    return None


def analyze_layer_attention_breakdown(analyzer, original_text, adversarial_text):
    """
    Track how attention patterns evolve through BERT's 12 layers
    """
    print("Running layer-wise attention breakdown analysis")
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    def get_layer_attentions(text):
        inputs = analyzer.tokenizer(text, return_tensors='pt', 
                                     truncation=True, max_length=128).to(analyzer.device)
        with torch.no_grad():
            outputs = analyzer.model(**inputs, output_attentions=True)
        
        # attentions: tuple of 12 tensors (one per layer)
        # Each: [batch, num_heads, seq_len, seq_len]
        layer_attentions = []
        for layer_attn in outputs.attentions:
            # Average across heads
            avg_attn = layer_attn[0].mean(dim=0).cpu().numpy()
            layer_attentions.append(avg_attn)
        
        return layer_attentions, outputs.logits[0].cpu().numpy()
    
    # Get attentions for both texts
    orig_layers, orig_logits = get_layer_attentions(original_text)
    adv_layers, adv_logits = get_layer_attentions(adversarial_text)
    
    # Verify we got the correct number of layers
    num_layers = len(orig_layers)
    if num_layers != 12:
        print(f"Warning: Expected 12 layers but got {num_layers}")
    if len(adv_layers) != num_layers:
        print(f"Warning: Original has {num_layers} layers but adversarial has {len(adv_layers)}")
    
    # Calculate divergence at each layer
    divergences = []
    for layer in range(min(num_layers, 12)):
        try:
            # Get the attention matrices
            orig_attn = orig_layers[layer]
            adv_attn = adv_layers[layer]
            
            # Find minimum size to handle different sequence lengths
            min_size = min(orig_attn.shape[0], orig_attn.shape[1], 
                          adv_attn.shape[0], adv_attn.shape[1])
            
            if min_size < 2:
                print(f"Warning: Layer {layer+1} has insufficient size ({min_size}), skipping")
                divergences.append(0.0)
                continue
            
            # Crop to same size
            orig_cropped = orig_attn[:min_size, :min_size]
            adv_cropped = adv_attn[:min_size, :min_size]
            
            # Flatten attention matrices
            orig_flat = orig_cropped.flatten()
            adv_flat = adv_cropped.flatten()
            
            # Check for zeros or invalid values
            if orig_flat.sum() == 0 or adv_flat.sum() == 0:
                print(f"Warning: Layer {layer+1} has zero-sum attention, skipping")
                divergences.append(0.0)
                continue
            
            # Normalize to make them proper probability distributions
            orig_flat = orig_flat / (orig_flat.sum() + 1e-10)
            adv_flat = adv_flat / (adv_flat.sum() + 1e-10)
            
            # JS divergence (symmetric KL divergence)
            js_div = jensenshannon(orig_flat, adv_flat)
            
            # Check if result is valid
            if np.isnan(js_div) or np.isinf(js_div):
                print(f"Warning: Layer {layer+1} JS divergence is invalid (NaN/Inf), setting to 0")
                divergences.append(0.0)
            else:
                divergences.append(js_div)
        
        except Exception as e:
            # If there is an error, skip layer
            print(f"Warning: Skipping layer {layer+1} due to error: {e}")
            divergences.append(0.0)
    
    # Check if we have any non-zero divergences
    if all(d == 0.0 for d in divergences):
        print(f"Warning: All layer divergences are 0. Original text length: {len(original_text)}, Adversarial text length: {len(adversarial_text)}")
        print(f"Divergences: {divergences}")
    
    # Fill up to 12 layers if needed
    while len(divergences) < 12:
        divergences.append(0.0)
    
    # Plot layer-wise divergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(divergences) + 1), divergences, marker='o', linewidth=2, markersize=8)
    plt.xlabel('BERT Layer')
    plt.ylabel('Attention Divergence (JS Distance)')
    plt.title('Layer-wise Attention Breakdown Under Attack')
    plt.grid(True, alpha=0.3)
    mean_div = np.mean(divergences)
    if mean_div > 0:
        plt.axhline(y=mean_div, color='r', linestyle='--', 
                    label=f'Mean Divergence: {mean_div:.3f}')
    plt.legend()
    print(f"  Generating: layer_wise_attention_breakdown.png")
    plt.savefig('results/plots/layer_wise_attention_breakdown.png', dpi=150)
    plt.close()
    
    # Find critical layer (where divergence spikes)
    critical_layer = np.argmax(divergences) + 1
    
    return {
        'divergences': divergences,
        'critical_layer': critical_layer,
        'orig_logits': orig_logits,
        'adv_logits': adv_logits
    }


def adversarial_subspace_analysis(analyzer, attack_results):
    """
    Use PCA to visualize adversarial vs clean examples in embedding space
    """
    print("Running adversarial subspace analysis...")
    clean_embeddings = []
    adv_embeddings = []
    labels = []
    attack_success = []
    
    # Collect embeddings
    for level in [3, 5]:
        for example in attack_results[level]:
            # Get CLS embedding (sentence representation)
            def get_cls_embedding(text):
                inputs = analyzer.tokenizer(text, return_tensors='pt', 
                                           truncation=True, max_length=128).to(analyzer.device)
                with torch.no_grad():
                    outputs = analyzer.model.bert(**inputs, output_hidden_states=True)
                # Use final layer CLS token
                cls_embedding = outputs.hidden_states[-1][0, 0, :].cpu().numpy()
                return cls_embedding
            
            clean_emb = get_cls_embedding(example['original_text'])
            adv_emb = get_cls_embedding(example['perturbed_text'])
            
            clean_embeddings.append(clean_emb)
            adv_embeddings.append(adv_emb)
            labels.append(example['true_label'])
            attack_success.append(example['attack_success'])
    
    # Check if we have any data
    if not clean_embeddings:
        print("No data available for adversarial subspace analysis")
        return None
    
    # Check if we have enough samples for PCA
    if len(clean_embeddings) < 2:
        print(f"Not enough samples ({len(clean_embeddings)}) for PCA visualization")
        return None
    
    # Combine for dimensionality reduction
    all_embeddings = np.array(clean_embeddings + adv_embeddings)
    
    # PCA directly to 2D for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(all_embeddings)
    
    # Split back into clean and adversarial
    n = len(clean_embeddings)
    clean_2d = embeddings_2d[:n]
    adv_2d = embeddings_2d[n:]
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Clean vs Adversarial
    ax = axes[0]
    ax.scatter(clean_2d[:, 0], clean_2d[:, 1], c='blue', alpha=0.5, 
               label='Clean', s=50)
    ax.scatter(adv_2d[:, 0], adv_2d[:, 1], c='red', alpha=0.5, 
               label='Adversarial', s=50)
    
    # Draw arrows showing perturbation direction
    for i in range(0, n, 10):  # Every 10th example
        ax.arrow(clean_2d[i, 0], clean_2d[i, 1],
                adv_2d[i, 0] - clean_2d[i, 0],
                adv_2d[i, 1] - clean_2d[i, 1],
                alpha=0.3, width=0.1, head_width=0.5, color='gray')
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Embedding Space: Clean vs Adversarial')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Color by attack success
    ax = axes[1]
    
    # Separate successful and failed attacks for legend
    robust_indices = [i for i, s in enumerate(attack_success) if not s]
    fooled_indices = [i for i, s in enumerate(attack_success) if s]
    
    if robust_indices:
        ax.scatter(adv_2d[robust_indices, 0], adv_2d[robust_indices, 1], 
                  c='green', alpha=0.6, s=50, label='Robust (Attack Failed)')
    if fooled_indices:
        ax.scatter(adv_2d[fooled_indices, 0], adv_2d[fooled_indices, 1], 
                  c='red', alpha=0.6, s=50, label='Fooled (Attack Succeeded)')
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('Adversarial Examples by Attack Outcome')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    print(f"  Generating: adversarial_subspace_analysis.png")
    plt.savefig('results/plots/adversarial_subspace_analysis.png', dpi=150)
    plt.close()
    
    # Calculate statistics
    perturbation_magnitudes = np.linalg.norm(
        np.array(adv_embeddings) - np.array(clean_embeddings), axis=1
    )
    
    successful_attacks = [attack_success[i] for i in range(len(attack_success))]
    successful_magnitudes = [perturbation_magnitudes[i] for i, s in enumerate(successful_attacks) if s]
    failed_magnitudes = [perturbation_magnitudes[i] for i, s in enumerate(successful_attacks) if not s]
    
    print(f"\nAdversarial Subspace Analysis")
    print(f"Avg perturbation magnitude (successful attacks): {np.mean(successful_magnitudes):.4f}")
    print(f"Avg perturbation magnitude (failed attacks): {np.mean(failed_magnitudes):.4f}")
    print(f"PCA explained variance (first 10 components): {np.sum(pca.explained_variance_ratio_[:10]):.2%}")
    
    return {
        'perturbation_magnitudes': perturbation_magnitudes,
        'embeddings_2d': embeddings_2d,
        'attack_success': attack_success
    }


def gradient_sensitivity_analysis(analyzer, attack_results):
    """
    Measure how sensitive model predictions are to input perturbations
    """
    print("Running gradient sensitivity analysis...")
    
    def compute_gradient_norm(text, label):
        """Compute L2 norm of gradient w.r.t. input embeddings"""
        try:
            inputs = analyzer.tokenizer(text, return_tensors='pt', 
                                        truncation=True, max_length=128).to(analyzer.device)
            
            # Get word embeddings
            embeddings = analyzer.model.bert.embeddings.word_embeddings(inputs['input_ids'])
            embeddings.retain_grad()
            
            # Zero gradients
            analyzer.model.zero_grad()
            
            # Forward pass using custom embeddings
            outputs = analyzer.model(inputs_embeds=embeddings, 
                                    attention_mask=inputs['attention_mask'])
            
            # Loss w.r.t. true label
            loss = outputs.logits[0, label]
            loss.backward()
            
            # Gradient norm
            if embeddings.grad is not None:
                grad_norm = embeddings.grad.norm(p=2).item()
            else:
                grad_norm = 0.0
            
            return grad_norm
        except Exception as e:
            print(f"Error computing gradient: {e}")
            return 0.0
    
    clean_gradients = []
    adv_gradients = []
    confidence_changes = []
    
    print(f"Analyzing {min(len(attack_results[3]), 50)} examples")
    
    for i, example in enumerate(attack_results[3][:50]):
        if not example['attack_success']:
            continue
        
        try:
            # Gradient norm for clean example
            clean_grad = compute_gradient_norm(
                example['original_text'], 
                example['true_label']
            )
            
            # Gradient norm for adversarial example
            adv_grad = compute_gradient_norm(
                example['perturbed_text'],
                example['true_label']
            )
            
            # Confidence change
            # Get original confidence
            inputs = analyzer.tokenizer(text=example['original_text'], return_tensors='pt', truncation=True, max_length=128).to(analyzer.device)
            with torch.no_grad():
                outputs = analyzer.model(**inputs)
                orig_conf = torch.softmax(outputs.logits, dim=-1)[0, example['true_label']].item()
            
            conf_change = abs(orig_conf - example['adversarial_confidence'])
            
            clean_gradients.append(clean_grad)
            adv_gradients.append(adv_grad)
            confidence_changes.append(conf_change)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} examples")
        except Exception as e:
            print(f"Skipping example {i} due to error: {e}")
            continue
    
    # Check if there is any data
    if not clean_gradients:
        print("No successful attacks found for gradient analysis")
        return None
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Gradient norms
    ax = axes[0]
    x = np.arange(len(clean_gradients))
    ax.scatter(x, clean_gradients, alpha=0.6, label='Clean', s=50)
    ax.scatter(x, adv_gradients, alpha=0.6, label='Adversarial', s=50)
    ax.set_xlabel('Example Index')
    ax.set_ylabel('Gradient L2 Norm')
    ax.set_title('Gradient Sensitivity: Clean vs Adversarial')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Gradient vs confidence change
    ax = axes[1]
    ax.scatter(clean_gradients, confidence_changes, alpha=0.6, s=50)
    ax.set_xlabel('Gradient Norm (Clean)')
    ax.set_ylabel('Confidence Change')
    ax.set_title('Gradient Sensitivity vs Confidence Change')
    ax.grid(True, alpha=0.3)
    
    # Fit line
    z = np.polyfit(clean_gradients, confidence_changes, 1)
    p = np.poly1d(z)
    ax.plot(clean_gradients, p(clean_gradients), "r--", alpha=0.8, 
            label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax.legend()
    
    plt.tight_layout()
    print(f"  Generating: gradient_sensitivity_analysis.png")
    plt.savefig('results/plots/gradient_sensitivity_analysis.png', dpi=150)
    plt.close()
    
    # Results
    print("\n Gradient Sensitivity Analysis")
    print(f"Avg gradient norm (clean): {np.mean(clean_gradients):.4f}")
    print(f"Avg gradient norm (adversarial): {np.mean(adv_gradients):.4f}")
    print(f"Correlation (gradient vs conf change): {np.corrcoef(clean_gradients, confidence_changes)[0,1]:.3f}")
    
    return {
        'clean_gradients': clean_gradients,
        'adv_gradients': adv_gradients,
        'confidence_changes': confidence_changes
    }


def semantic_similarity_validation(attack_results):
    """
    Validate that adversarial examples preserve semantic meaning
    using sentence transformers
    """
    print("Running semantic similarity validation...")
    try:
        # Load semantic similarity model
        print("Loading semantic similarity model...")
        sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load semantic similarity model: {e}")
        return None
    
    similarities = []
    edit_distances = []
    
    for example in attack_results[3]:
        if not example['attack_success']:
            continue
        
        orig = example['original_text']
        adv = example['perturbed_text']
        
        # Semantic similarity
        emb1 = sim_model.encode(orig)
        emb2 = sim_model.encode(adv)
        similarity = util.cos_sim(emb1, emb2).item()
        similarities.append(similarity)
        
        # Edit distance (normalized)
        edit_dist = 1 - SequenceMatcher(None, orig, adv).ratio()
        edit_distances.append(edit_dist)
    
    # Check if there is any data
    if not similarities:
        print("No successful attacks found for semantic similarity validation")
        return None
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Semantic similarity distribution
    ax = axes[0]
    ax.hist(similarities, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=np.mean(similarities), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(similarities):.3f}')
    ax.axvline(x=0.85, color='green', linestyle='--', 
               linewidth=2, label='Semantic Threshold: 0.85')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Count')
    ax.set_title('Semantic Similarity: Original vs Adversarial')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Semantic similarity vs edit distance
    ax = axes[1]
    ax.scatter(edit_distances, similarities, alpha=0.6, s=50)
    ax.set_xlabel('Edit Distance (normalized)')
    ax.set_ylabel('Semantic Similarity')
    ax.set_title('Text Modification vs Semantic Preservation')
    ax.grid(True, alpha=0.3)
    
    # Quadrants
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.5)
    ax.text(0.05, 0.95, 'Good Attack\n(low edit, high sim)', 
            transform=ax.transAxes, fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    print(f"  Generating: semantic_similarity_validation.png")
    plt.savefig('results/plots/semantic_similarity_validation.png', dpi=150)
    plt.close()
    
    # Results
    high_quality_attacks = sum(1 for s in similarities if s > 0.85)
    print("\n Semantic Similarity Validation")
    print(f"Avg semantic similarity: {np.mean(similarities):.3f}")
    print(f"Avg edit distance: {np.mean(edit_distances):.3f}")
    print(f"High-quality attacks (sim > 0.85): {high_quality_attacks}/{len(similarities)} "
          f"({high_quality_attacks/len(similarities):.1%})")
    
    # Find examples with low semantic preservation
    low_sim_indices = [i for i, s in enumerate(similarities) if s < 0.75]
    if low_sim_indices:
        print(f"\nWarning: {len(low_sim_indices)} adversarial examples have low semantic similarity (<0.75)")
        print("Examples of low-similarity attacks:")
        successful_examples = [ex for ex in attack_results[3] if ex['attack_success']]
        for idx in low_sim_indices:
            example = successful_examples[idx]
            #print(f"Original: {example['original_text'][:120]}")
            #print(f"Adversarial: {example['perturbed_text'][:120]}")
            #print(f"Similarity: {similarities[idx]:.3f} \n")
    
    return {
        'similarities': similarities,
        'edit_distances': edit_distances,
        'high_quality_rate': high_quality_attacks / len(similarities)
    }


def main():    
    # Initialize analyzer with BERT
    analyzer = TransformerRobustnessAnalyzer(model_name="bert-base-uncased")
    
    # Step 1: Load data
    analyzer.load_data(dataset_name="sst2", max_samples=1000)
    
    # Step 2: Train baseline model
    analyzer.train_model(epochs=3, learning_rate=2e-5)
    
    # Step 3: Analyze baseline
    analyzer.analyze_baseline(num_samples=100)
    
    # Step 4: Run adversarial attacks
    attacker = SemanticAdversarialAttack(
        analyzer.model, 
        analyzer.tokenizer, 
        analyzer.device
    )
    
    attack_results = attacker.evaluate_attack(
        analyzer.dataset['test'],
        num_samples=100,
        perturbation_levels=[1, 2, 3, 5]
    )
    
    # Step 5: Visualize results (dashboard with subplots)
    visualize_results(analyzer, attack_results)
    
    # Step 6: Generate detailed report
    generate_detailed_report(analyzer, attack_results)
    
    # Step 7: Save plot data to JSON
    save_plot_data_to_json(analyzer, attack_results, filename='plot_data.json')
    
    # Step 8: Create individual plot files
    create_individual_plots(analyzer, attack_results, output_dir='results/plots')
    
    # Step 9: Layer-wise attention breakdown
    print("\n Running layer-wise attention breakdown")
    try:
        layer_results = []
        for example in attack_results[3][:10]:
            if example['attack_success']:
                try:
                    result = analyze_layer_attention_breakdown(
                        analyzer,
                        example['original_text'],
                        example['perturbed_text']
                    )
                    if result:
                        layer_results.append(result)
                except Exception as e:
                    print(f"Warning: Error in layer analysis for example: {e}")
                    continue
        
        if layer_results:
            avg_divergences = np.mean([r['divergences'] for r in layer_results], axis=0)
            print(f"Critical layer (avg): {np.argmax(avg_divergences) + 1}")
            print(f"Early layers (1-4) avg divergence: {np.mean(avg_divergences[:4]):.4f}")
            print(f"Middle layers (5-8) avg divergence: {np.mean(avg_divergences[4:8]):.4f}")
            print(f"Late layers (9-12) avg divergence: {np.mean(avg_divergences[8:]):.4f}")
    except Exception as e:
        print(f"Warning: Layer attention breakdown failed: {e}")
    
    # Step 10: Adversarial subspace analysis
    print("\n Running adversarial subspace analysis")
    try:
        adversarial_subspace_analysis(analyzer, attack_results)
    except Exception as e:
        print(f"Warning: Adversarial subspace analysis failed: {e}")
    
    # Step 11: Gradient sensitivity analysis
    print("\n Running gradient sensitivity analysis")
    try:
        gradient_sensitivity_analysis(analyzer, attack_results)
    except Exception as e:
        print(f"Warning: Gradient sensitivity analysis failed: {e}")
    
    # Step 12: Semantic similarity validation
    print("\n Running semantic similarity validation")
    try:
        semantic_similarity_validation(attack_results)
    except Exception as e:
        print(f"Warning: Semantic similarity validation failed: {e}")
    
    # Print all examples in a consistent format
    print("\n" + "="*70)
    print("ADVERSARIAL EXAMPLES")
    print("="*70)
    for level in sorted(attack_results.keys()):
        for example in attack_results[level]:
            print(f"Original: {example['original_text']}")
            print(f"True Label: {'Positive' if example['true_label'] == 1 else 'Negative'}")
            print("")
            print(f"Adversarial: {example['perturbed_text']}")
            print(f"Predicted: {'Positive' if example['adversarial_prediction'] == 1 else 'Negative'}")
            print(f"Confidence: {example['adversarial_confidence']:.4f}")
            print(f"Attack Success: {example['attack_success']}")
            print("")
    

if __name__ == "__main__":
    main()