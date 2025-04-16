# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention block with residual connections and layer normalization"""
    
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=emb_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),  # GELU generally performs better than ReLU
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection and layer norm
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # FFN with residual connection and layer norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


class AgriRecommender(nn.Module):
    def __init__(
        self, 
        input_dim, 
        emb_dim=128,          
        num_heads=4,          
        hidden_dim=256,       
        crop_classes=10, 
        fert_classes=5, 
        vocab_size=100, 
        task_embedding_dim=32,  
        num_layers=2,         
        dropout=0.1           
    ):
        super(AgriRecommender, self).__init__()
        
        # Feature embedding with BatchNorm for numerical stability
        self.feature_embed = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(dropout)
        )
        
        # Task embedding with larger embedding dimension
        self.task_embed = nn.Embedding(num_embeddings=10, embedding_dim=task_embedding_dim)
        self.task_proj = nn.Linear(task_embedding_dim, emb_dim)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AttentionBlock(emb_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Task-specific classifier heads with dropout
        self.crop_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, crop_classes)
        )
        
        self.fertilizer_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fert_classes)
        )
        
        # Optional text generation components with improved architecture
        self.gru_decoder = nn.GRU(
            input_size=emb_dim, 
            hidden_size=emb_dim, 
            num_layers=2,  # More layers for better sequence modeling
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        self.text_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, vocab_size)
        )
        
        # Initialize weights for better training
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for p in self.parameters():
            if p.dim() > 1:  # Weight matrices
                nn.init.xavier_uniform_(p)
                
    def forward(self, x, task_id, text_seq=None):
        # Handle single samples vs batches
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            
        # Apply feature embedding (with proper reshaping for BatchNorm1d)
        batch_size = x.shape[0]
        x = self.feature_embed(x)  # Result: [batch_size, emb_dim]
        x = x.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, emb_dim]
        
        # Get task embedding and add to input
        task_emb = self.task_proj(self.task_embed(task_id)).unsqueeze(1)  # [batch_size, 1, emb_dim]
        x = x + task_emb  # Feature + task conditioning
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Get output embeddings - take the sequence output
        embedding = x.squeeze(1)  # [batch_size, emb_dim]
        
        # Compute logits for each head
        crop_logits = self.crop_head(embedding)  # [batch_size, crop_classes]
        fert_logits = self.fertilizer_head(embedding)  # [batch_size, fert_classes]
        
        outputs = {
            "crop": crop_logits,
            "fertilizer": fert_logits,
            "embedding": embedding  # Include embeddings for potential other uses
        }
        
        # Optional text generation component
        if text_seq is not None:
            # Use the sequence embedding as initial hidden state
            hidden = embedding.unsqueeze(0).repeat(self.gru_decoder.num_layers, 1, 1)
            # Pass through GRU decoder
            dec_out, _ = self.gru_decoder(text_seq, hidden)
            # Project to vocabulary space
            text_logits = self.text_proj(dec_out)
            outputs["text"] = text_logits
            
        return outputs


# Helper function for inference with confidence scores
def predict_with_confidence(model, features, task_id, device='cpu'):
    """Get predictions with confidence scores"""
    model.eval()
    with torch.no_grad():
        features = features.to(device)
        task_id = task_id.to(device)
        outputs = model(features, task_id)
        
        # Get predicted class and confidence for crops
        crop_probs = F.softmax(outputs['crop'], dim=1)
        crop_conf, crop_pred = torch.max(crop_probs, dim=1)
        
        # Get predicted class and confidence for fertilizer
        fert_probs = F.softmax(outputs['fertilizer'], dim=1)
        fert_conf, fert_pred = torch.max(fert_probs, dim=1)
        
        return {
            'crop_pred': crop_pred.item(),
            'crop_confidence': crop_conf.item(),
            'fertilizer_pred': fert_pred.item(),
            'fertilizer_confidence': fert_conf.item(),
            'embeddings': outputs['embedding'].cpu()  # Feature representation
        }