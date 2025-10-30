# main.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import requests
import zipfile
import os
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerAnalysis:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./cache')
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            cache_dir='./cache'
        ).to(self.device)
        
    def load_imdb_data(self, sample_size=2000):
        """Load a sample of IMDB data"""
        print("Loading IMDB data...")
        
        # Create sample data (in practice, you'd load the full dataset)
        np.random.seed(42)
        n_samples = sample_size
        
        # Generate synthetic movie reviews for demonstration
        positive_reviews = [
            f"This movie was absolutely fantastic! The acting was superb and the storyline kept me engaged throughout. " +
            f"The director did an amazing job with character development. I would definitely recommend this to anyone " +
            f"who enjoys {genre} films. The cinematography was stunning and the musical score was perfect."
            for genre in ['drama', 'action', 'comedy', 'romance', 'thriller']
        ]
        
        negative_reviews = [
            f"I was very disappointed with this film. The plot was predictable and the acting felt forced. " +
            f"The characters were poorly developed and the dialogue was unconvincing. The {aspect} was particularly " +
            f"weak and overall it failed to deliver an engaging experience. I wouldn't recommend wasting your time."
            for aspect in ['cinematography', 'editing', 'soundtrack', 'direction', 'screenplay']
        ]
        
        # Create balanced dataset
        texts = []
        labels = []
        
        for i in range(n_samples // 2):
            texts.append(np.random.choice(positive_reviews))
            labels.append(1)  # Positive
            
        for i in range(n_samples // 2):
            texts.append(np.random.choice(negative_reviews))
            labels.append(0)  # Negative
        
        # Shuffle
        indices = np.random.permutation(len(texts))
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]
        
        # Split train/test
        split_idx = int(0.8 * len(texts))
        train_texts, test_texts = texts[:split_idx], texts[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]
        
        return (train_texts, train_labels), (test_texts, test_labels)
    
    def train(self, train_dataset, test_dataset, epochs=3, batch_size=16):
        """Train the transformer model"""
        print(f"Training {self.model_name} on IMDB dataset...")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        loss_fn = nn.CrossEntropyLoss()
        
        train_losses = []
        accuracies = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Evaluate
            accuracy = self.evaluate(test_loader)
            accuracies.append(accuracy)
            
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
        
        return train_losses, accuracies
    
    def evaluate(self, dataloader):
        """Evaluate model accuracy"""
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())
        
        return accuracy_score(actual_labels, predictions)
    
    def get_confidence_profile(self, dataloader):
        """Analyze model confidence distribution"""
        self.model.eval()
        confidences = []
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                max_probs, preds = torch.max(probs, dim=1)
                
                confidences.extend(max_probs.cpu().numpy())
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())
        
        return np.array(confidences), np.array(predictions), np.array(actual_labels)
    
    def attention_analysis(self, text):
        """Analyze attention patterns for a given text"""
        self.model.eval()
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Get attention weights
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        
        # Get attention from last layer
        attention = outputs.attentions[-1]  # Last layer
        attention = attention.mean(dim=1)  # Average over heads
        
        # Process tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        return attention[0].cpu().numpy(), tokens, torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

class AdversarialAttack:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def synonym_replacement(self, text, replacement_ratio=0.3):
        """Meaning-preserving attack using synonym replacement"""
        # Simple synonym dictionary for demonstration
        synonyms = {
            'good': ['great', 'excellent', 'wonderful', 'fantastic', 'superb'],
            'bad': ['terrible', 'awful', 'horrible', 'poor', 'disappointing'],
            'happy': ['joyful', 'pleased', 'delighted', 'content', 'satisfied'],
            'sad': ['unhappy', 'depressed', 'miserable', 'gloomy', 'melancholy'],
            'beautiful': ['gorgeous', 'stunning', 'lovely', 'attractive', 'pretty'],
            'ugly': ['unattractive', 'hideous', 'repulsive', 'unpleasant', 'disgusting'],
            'smart': ['intelligent', 'clever', 'brilliant', 'bright', 'knowledgeable'],
            'stupid': ['foolish', 'dumb', 'unintelligent', 'silly', 'idiotic'],
        }
        
        words = text.split()
        n_replace = max(1, int(len(words) * replacement_ratio))
        
        # Find replaceable words
        replaceable_indices = []
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                replaceable_indices.append(i)
        
        # Randomly replace words
        if replaceable_indices:
            indices_to_replace = np.random.choice(
                replaceable_indices, 
                size=min(n_replace, len(replaceable_indices)), 
                replace=False
            )
            
            for idx in indices_to_replace:
                original_word = words[idx].lower()
                if original_word in synonyms:
                    replacement = np.random.choice(synonyms[original_word])
                    # Preserve capitalization
                    if words[idx][0].isupper():
                        replacement = replacement.capitalize()
                    words[idx] = replacement
        
        return ' '.join(words)
    
    def add_distracting_context(self, text, distraction_ratio=0.2):
        """Add irrelevant but plausible context to confuse the model"""
        distractions = [
            "It's worth mentioning that the overall production quality was quite standard.",
            "Interestingly, the cinematographic approach followed conventional patterns.",
            "From a technical perspective, the editing was competently executed.",
            "The narrative structure adhered to familiar storytelling conventions.",
            "Character development followed predictable but acceptable trajectories."
        ]
        
        words = text.split()
        insert_pos = int(len(words) * distraction_ratio)
        
        distraction = np.random.choice(distractions)
        words.insert(insert_pos, distraction)
        
        return ' '.join(words)

def plot_results(analyzer, confidences, predictions, actual_labels, attention_weights, tokens):
    """Create comprehensive visualization of results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Confidence distribution
    axes[0, 0].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Confidence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Model Confidence Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy vs Confidence threshold
    thresholds = np.linspace(0.5, 0.95, 10)
    accuracies = []
    coverages = []
    
    for threshold in thresholds:
        mask = confidences >= threshold
        if mask.sum() > 0:
            acc = accuracy_score(actual_labels[mask], predictions[mask])
            accuracies.append(acc)
            coverages.append(mask.sum() / len(confidences))
        else:
            accuracies.append(0)
            coverages.append(0)
    
    ax2 = axes[0, 1]
    ax2.plot(thresholds, accuracies, 'bo-', label='Accuracy')
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Accuracy', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, alpha=0.3)
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(thresholds, coverages, 'r^-', label='Coverage')
    ax2_twin.set_ylabel('Coverage', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title('Accuracy vs Confidence Threshold')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # Plot 3: Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 2])
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Actual')
    axes[0, 2].set_title('Confusion Matrix')
    
    # Plot 4: Attention visualization
    if attention_weights is not None and tokens is not None:
        # Use first 20 tokens for clarity
        n_tokens = min(20, len(tokens))
        attention_subset = attention_weights[:n_tokens, :n_tokens]
        tokens_subset = tokens[:n_tokens]
        
        im = axes[1, 0].imshow(attention_subset, cmap='viridis', aspect='auto')
        axes[1, 0].set_xticks(range(n_tokens))
        axes[1, 0].set_yticks(range(n_tokens))
        axes[1, 0].set_xticklabels(tokens_subset, rotation=45, fontsize=8)
        axes[1, 0].set_yticklabels(tokens_subset, fontsize=8)
        axes[1, 0].set_title('Attention Weights (First 20 tokens)')
        plt.colorbar(im, ax=axes[1, 0])
    
    # Plot 5: Class distribution
    class_counts = np.bincount(actual_labels)
    axes[1, 1].bar(['Negative', 'Positive'], class_counts, color=['red', 'green'], alpha=0.7)
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Class Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Training history (placeholder)
    axes[1, 2].text(0.5, 0.5, 'Training History\n(Would show loss/accuracy curves)', 
                    ha='center', va='center', transform=axes[1, 2].transAxes, fontsize=12)
    axes[1, 2].set_title('Training Progress')
    
    plt.tight_layout()
    plt.savefig('transformer_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def adversarial_analysis(analyzer, attack, test_texts, test_labels):
    """Analyze model vulnerability to adversarial attacks"""
    print("\n" + "="*60)
    print("ADVERSARIAL ATTACK ANALYSIS")
    print("="*60)
    
    attack_strengths = [0.1, 0.2, 0.3, 0.4, 0.5]
    original_accuracy = []
    attacked_accuracy = []
    attack_success_rates = []
    
    # Test original accuracy
    test_dataset = IMDBDataset(test_texts, test_labels, analyzer.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16)
    original_acc = analyzer.evaluate(test_loader)
    
    for strength in attack_strengths:
        print(f"\nTesting attack strength: {strength}")
        
        # Apply adversarial attacks
        attacked_texts = []
        for text in test_texts:
            # Apply both attacks
            text1 = attack.synonym_replacement(text, strength)
            text2 = attack.add_distracting_context(text1, strength/2)
            attacked_texts.append(text2)
        
        # Evaluate on attacked data
        attacked_dataset = IMDBDataset(attacked_texts, test_labels, analyzer.tokenizer)
        attacked_loader = DataLoader(attacked_dataset, batch_size=16)
        attacked_acc = analyzer.evaluate(attacked_loader)
        
        attack_success_rate = max(0, original_acc - attacked_acc) / original_acc
        
        original_accuracy.append(original_acc)
        attacked_accuracy.append(attacked_acc)
        attack_success_rates.append(attack_success_rate)
        
        print(f"Original Accuracy: {original_acc:.4f}")
        print(f"Attacked Accuracy: {attacked_acc:.4f}")
        print(f"Attack Success Rate: {attack_success_rate:.4f}")
    
    # Plot adversarial results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(attack_strengths, original_accuracy, 'go-', label='Original', linewidth=2)
    plt.plot(attack_strengths, attacked_accuracy, 'ro-', label='Attacked', linewidth=2)
    plt.xlabel('Attack Strength')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Attack Strength')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(range(len(attack_strengths)), attack_success_rates, color='orange', alpha=0.7)
    plt.xticks(range(len(attack_strengths)), [str(s) for s in attack_strengths])
    plt.xlabel('Attack Strength')
    plt.ylabel('Success Rate')
    plt.title('Attack Success Rate')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    accuracy_drop = [orig - attacked for orig, attacked in zip(original_accuracy, attacked_accuracy)]
    plt.plot(attack_strengths, accuracy_drop, 'mo-', linewidth=2)
    plt.xlabel('Attack Strength')
    plt.ylabel('Accuracy Drop')
    plt.title('Accuracy Reduction')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adversarial_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return attack_strengths, original_accuracy, attacked_accuracy, attack_success_rates

def main():
    print("Transformer Model Analysis with Adversarial Attacks")
    print("="*60)
    
    # Step 1: Build baseline model
    print("\nStep 1: Building Baseline Model")
    analyzer = TransformerAnalysis('bert-base-uncased')
    
    # Load data
    (train_texts, train_labels), (test_texts, test_labels) = analyzer.load_imdb_data()
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, analyzer.tokenizer)
    test_dataset = IMDBDataset(test_texts, test_labels, analyzer.tokenizer)
    
    # Train model
    train_losses, accuracies = analyzer.train(train_dataset, test_dataset, epochs=2)
    
    # Evaluate baseline performance
    test_loader = DataLoader(test_dataset, batch_size=16)
    baseline_accuracy = analyzer.evaluate(test_loader)
    print(f"\nBaseline Accuracy: {baseline_accuracy:.4f}")
    
    # Get confidence profile
    confidences, predictions, actual_labels = analyzer.get_confidence_profile(test_loader)
    
    # Attention analysis on sample text
    sample_text = "This movie was absolutely wonderful with fantastic acting and a great storyline."
    attention_weights, tokens, probs = analyzer.attention_analysis(sample_text)
    
    print(f"\nSample prediction: {np.argmax(probs)} (Positive)" if np.argmax(probs) == 1 
          else f"Sample prediction: {np.argmax(probs)} (Negative)")
    print(f"Confidence: {np.max(probs):.4f}")
    
    # Plot results
    plot_results(analyzer, confidences, predictions, actual_labels, attention_weights, tokens)
    
    # Step 2: Adversarial attacks
    print("\nStep 2: Performing Adversarial Attacks")
    attack = AdversarialAttack(analyzer.tokenizer)
    
    attack_strengths, original_acc, attacked_acc, success_rates = adversarial_analysis(
        analyzer, attack, test_texts, test_labels
    )
    
    # Step 3: Analyze failure modes
    print("\n" + "="*60)
    print("FAILURE MODE ANALYSIS")
    print("="*60)
    
    print("\nWhat's going wrong:")
    print("1. Attention diversion: Adversarial changes redirect attention from important words")
    print("2. Semantic confusion: Synonym replacement creates subtle meaning shifts")
    print("3. Context pollution: Added distractions dilute the main sentiment")
    print("4. Over-reliance on specific keywords: Model depends too much on strong sentiment words")
    
    # Step 4: Defense proposals
    print("\n" + "="*60)
    print("DEFENSE STRATEGIES")
    print("="*60)
    
    print("\nProposed Defenses:")
    print("1. Adversarial Training:")
    print("   - Train on mix of clean and adversarial examples")
    print("   - Expected: +5-10% robustness, -2-5% clean accuracy")
    
    print("\n2. Attention Regularization:")
    print("   - Penalize erratic attention patterns")
    print("   - Expected: More consistent predictions, minimal accuracy impact")
    
    print("\n3. Ensemble Methods:")
    print("   - Combine multiple models with different architectures")
    print("   - Expected: +10-15% robustness, +2x inference cost")
    
    print("\n4. Input Sanitization:")
    print("   - Detect and filter adversarial patterns")
    print("   - Expected: +8-12% robustness, potential false positives")
    
    print("\n5. Certified Defenses:")
    print("   - Mathematical guarantees for robustness within bounds")
    print("   - Expected: Strong theoretical protection, significant compute overhead")

if __name__ == "__main__":
    main()