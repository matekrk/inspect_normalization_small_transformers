import math
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


torch.manual_seed(8)
np.random.seed(8)

# ======== NORMALIZATION LAYERS ========

class BatchNorm1D(nn.Module):
    """Batch Normalization for transformers (applied to the last dimension)"""
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # BatchNorm1d expects: (batch_size, hidden_size, seq_len)
        return self.batch_norm(x.transpose(1, 2)).transpose(1, 2)


class LayerNorm(nn.LayerNorm):
    """Standard Layer Normalization"""
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__(hidden_size, eps=eps)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization
    
    Paper: https://arxiv.org/abs/1910.07467
    """
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normalized = x / rms
        # Scale with learned parameters
        return self.weight * x_normalized


class NoNorm(nn.Module):
    """No normalization, just a pass-through layer for comparison"""
    def __init__(self, hidden_size, eps=1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        
    def forward(self, x):
        return x * self.weight

class DyT(nn.Module):
    def __init__(self, hidden_size, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
    
    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias

# ======== TRANSFORMER BLOCKS ========

class FeedForward(nn.Module):
    """Standard feed-forward network in transformer"""
    def __init__(self, hidden_size, ff_size, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, ff_size)
        self.linear2 = nn.Linear(ff_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_size ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.shape
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_size]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, seq_len, hidden_size)
        
        output = self.out_proj(context)
        return output


class TransformerBlock(nn.Module):
    """Generic Transformer block that can use different normalization strategies"""
    def __init__(self, hidden_size, num_heads, ff_size, dropout=0.1, norm_type='layer'):
        super().__init__()
        
        # Define different normalization types - for now all are the same
        norm_classes = {
            'dyt': DyT,
            'layer': LayerNorm,
            'batch': BatchNorm1D,
            'rms': RMSNorm,
            'none': NoNorm
        }
        
        if norm_type not in norm_classes:
            raise ValueError(f"Unknown normalization type: {norm_type}")
        
        norm_class = norm_classes[norm_type]
        
        self.norm1 = norm_class(hidden_size)
        self.norm2 = norm_class(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.ff = FeedForward(hidden_size, ff_size, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        attn_output = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_output)
        
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x


class Transformer(nn.Module):
    """Simple transformer model for sequence classification"""
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, ff_size,
                 max_seq_len, num_classes, dropout=0.1, norm_type='layer'):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_size))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, ff_size, dropout, norm_type)
            for _ in range(num_layers)
        ])
        
        self.final_norm = LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        
        position = torch.arange(max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(1, max_seq_len, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.pos_encoding.data = pe
        
    def forward(self, x, mask=None):
        seq_len = x.size(1)
        
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x, mask)
            
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
            
        x = self.final_norm(x)
        x = self.classifier(x)
        
        return x


# ======== SYNTHETIC DATASET ========

class SyntheticSequenceDataset(Dataset):
    """Generate a synthetic sequence classification dataset"""
    def __init__(self, num_samples=10000, seq_len=128, vocab_size=10000, num_classes=2):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        # Generate synthetic data
        # For simplicity, we'll create data where the class is determined by the 
        # presence of specific tokens in the first half vs second half of the sequence
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # For class 0, place important tokens in first half
            # For class 1, place important tokens in second half
            label = np.random.randint(0, num_classes)
            
            # Generate random sequence
            seq = np.random.randint(1, vocab_size, size=seq_len)
            
            # Place "signal" tokens
            if label == 0:
                signal_pos = np.random.randint(0, seq_len // 2, size=5)
            else:
                signal_pos = np.random.randint(seq_len // 2, seq_len, size=5)
                
            seq[signal_pos] = np.random.randint(vocab_size - 100, vocab_size, size=5)
            
            self.data.append(seq)
            self.labels.append(label)
            
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# ======== TRAINING AND EVALUATION ========

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for data, labels in progress_bar:
        data, labels = data.to(device), labels.to(device)
        
        mask = (data != 0).float()
        
        logits = model(data, mask)
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        total_loss += loss.item()
        
        progress_bar.set_postfix({"loss": f"{total_loss/len(progress_bar):.4f}", 
                                  "acc": f"{correct/total:.4f}"})
    
    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            
            mask = (data != 0).float()
            
            logits = model(data, mask)
            loss = F.cross_entropy(logits, labels)
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader), correct / total


def train_and_evaluate(norm_type, train_loader, val_loader, device, 
                       vocab_size=10000, hidden_size=128, num_layers=2, num_heads=4, 
                       ff_size=512, max_seq_len=128, num_classes=2, 
                       lr=1e-4, epochs=10):
    """Train and evaluate a transformer model with the specified normalization type"""
    
    model = Transformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_size=ff_size,
        max_seq_len=max_seq_len,
        num_classes=num_classes,
        dropout=0.1,
        norm_type=norm_type
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model with {norm_type} normalization has {num_params:,} trainable parameters")
    
    # Be careful on optimizer choice.
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    print(f"\nTraining model with {norm_type} normalization...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        val_loss, val_acc = evaluate(model, val_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }


def run_normalization_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Creating synthetic dataset...")
    train_dataset = SyntheticSequenceDataset(num_samples=10000, seq_len=128)
    val_dataset = SyntheticSequenceDataset(num_samples=2000, seq_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    norm_types = ['dyt', 'layer', 'batch', 'rms', 'none']
    results = {}
    
    for norm_type in norm_types:
        results[norm_type] = train_and_evaluate(
            norm_type=norm_type,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=10  # Reduced for demonstration, use more for a real study
        )
    
    plot_results(results)

    torch.save(results, 'normalization_results_seed8_adamw_wd-4.pt')
    
    return results


def plot_results(results):
    """Plot training curves for all normalization methods"""
    norm_types = list(results.keys())
    epochs = range(1, len(results[norm_types[0]]['train_losses']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    ax = axes[0, 0]
    for norm_type in norm_types:
        ax.plot(epochs, results[norm_type]['train_losses'], label=norm_type)
    ax.set_title('Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    ax = axes[0, 1]
    for norm_type in norm_types:
        ax.plot(epochs, results[norm_type]['val_losses'], label=norm_type)
    ax.set_title('Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1, 0]
    for norm_type in norm_types:
        ax.plot(epochs, results[norm_type]['train_accs'], label=norm_type)
    ax.set_title('Training Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    ax = axes[1, 1]
    for norm_type in norm_types:
        ax.plot(epochs, results[norm_type]['val_accs'], label=norm_type)
    ax.set_title('Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('normalization_results_seed8_adamw_wd-4.png')
    plt.show()


def demonstrate_forward_stability(hidden = 128):
    """
    Demonstrates how different normalization techniques handle
    vanishing/exploding activations during forward pass
    """
    np.random.seed(8)
    torch.manual_seed(8)
    
    # Create input with regular and extreme values
    regular_values = torch.randn(32, 16, hidden)  # batch, seq, hidden
    extreme_values = torch.ones(32, 4, hidden) * 50.0  # Very large values
    
    # Replace some positions with extreme values
    x = regular_values.clone()
    x[:, -4:, :] = extreme_values
    
    norms = {
        'Dynamic Tanh': DyT(hidden),
        'Layer Norm': LayerNorm(hidden),
        'Batch Norm': BatchNorm1D(hidden),
        'RMS Norm': RMSNorm(hidden),
        'No Norm': NoNorm(hidden)
    }
    
    results = {}
    
    for name, norm in norms.items():
        output = norm(x)
        results[name] = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item(),
            'norm_ratio': output.norm().item() / x.norm().item(),
            'sample': output[0, -1, :10].tolist()  # First 10 values from an extreme row
        }
    
    print("\nForward Stability Analysis:")
    print("-" * 50)
    print(f"Input tensor shape: {x.shape}")
    print(f"Input statistics - Mean: {x.mean().item():.4f}, Std: {x.std().item():.4f}")
    print(f"Input range: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print("-" * 50)
    
    for name, stats in results.items():
        print(f"{name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  Norm ratio: {stats['norm_ratio']:.4f}")
        print(f"  Sample values (extreme row): {[f'{v:.4f}' for v in stats['sample'][:5]]}...")
        print("-" * 50)
        
    return results


if __name__ == "__main__":

    start_time = time.time()
    
    stability_results = demonstrate_forward_stability() # Demonstrate forward stability differences
    
    results = run_normalization_experiment() # Note: This will take some time to run

    print("\nFinal Results:")
    print("-" * 50)
    
    for norm_type, result in results.items():
        final_train_acc = result['train_accs'][-1]
        final_val_acc = result['val_accs'][-1]
        
        print(f"{norm_type.upper()} Normalization:")
        print(f"  Final Training Accuracy: {final_train_acc:.4f}")
        print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
        print("-" * 50)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
