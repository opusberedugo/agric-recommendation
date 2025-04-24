# enhanced_model.py
"""
Enhanced agricultural recommendation model that leverages 
both transformer attention and domain-specific agricultural relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SoilFeatureEncoder(nn.Module):
    """
    Specialized encoder for soil and environmental features that
    captures the relationships between soil nutrients, pH, and environmental factors.
    """
    
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(SoilFeatureEncoder, self).__init__()
        
        # Track soil features
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Main feature extraction pathway
        self.main_path = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )
        
        # Special NPK interaction module
        self.npk_module = nn.Sequential(
            nn.Linear(3, 32),  # Assuming first 3 features are N, P, K
            nn.ReLU(),
            nn.Linear(32, output_dim // 4),
            nn.ReLU(),
        )
        
        # Final projection
        self.output_projection = nn.Linear(output_dim + output_dim // 4, output_dim)
        
    def forward(self, x):
        # Main feature extraction
        main_features = self.main_path(x)
        
        # Extract and process NPK interactions
        # Assuming first 3 features are N, P, K
        npk_features = x[:, :3]
        npk_encoding = self.npk_module(npk_features)
        
        # Combine features
        combined = torch.cat([main_features, npk_encoding], dim=1)
        output = self.output_projection(combined)
        
        return output

class MultiheadSelfAttention(nn.Module):
    """Enhanced multi-head attention module with agriculture-specific optimizations"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiheadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # QKV projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to V
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        return output

class AgriTransformerBlock(nn.Module):
    """Enhanced transformer block for agricultural data"""
    
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super(AgriTransformerBlock, self).__init__()
        
        # Self-attention layer
        self.self_attn = MultiheadSelfAttention(emb_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        # Feed-forward network with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),  # GELU outperforms ReLU for transformer models
            nn.Dropout(dropout),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x

class TaskEmbedding(nn.Module):
    """Enhanced task embedding module"""
    
    def __init__(self, num_tasks, embedding_dim):
        super(TaskEmbedding, self).__init__()
        self.task_embed = nn.Embedding(num_tasks, embedding_dim)
        self.task_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
    def forward(self, task_id):
        task_embedding = self.task_embed(task_id)
        return self.task_projection(task_embedding)

class EnhancedAgriRecommender(nn.Module):
    """
    Enhanced agricultural recommendation model with specialized components
    for soil data, crop attributes, and fertilizer properties.
    """
    
    def __init__(
        self, 
        input_dim, 
        emb_dim=128,
        hidden_dim=256,
        num_heads=4,
        crop_classes=10, 
        fert_classes=5, 
        vocab_size=100, 
        task_embedding_dim=32,
        num_layers=2,
        dropout=0.1
    ):
        super(EnhancedAgriRecommender, self).__init__()
        
        # Feature embedding with specialized soil encoder
        self.soil_encoder = SoilFeatureEncoder(input_dim, emb_dim, dropout)
        
        # Task embedding
        self.task_embedding = TaskEmbedding(10, task_embedding_dim)  # 10 possible task types
        self.task_projection = nn.Linear(task_embedding_dim, emb_dim)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            AgriTransformerBlock(emb_dim, num_heads, dropout) 
            for _ in range(num_layers)
        ])
        
        # Crop recommendation head
        self.crop_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, crop_classes)
        )
        
        # Fertilizer recommendation head with nutrient-specific layers
        self.fert_common = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
        )
        
        # Specialized fertilizer outputs for different nutrient needs
        self.fert_n_layer = nn.Linear(hidden_dim, fert_classes // 2)
        self.fert_pk_layer = nn.Linear(hidden_dim, fert_classes // 2)
        self.fert_combine = nn.Linear(fert_classes, fert_classes)
        
        # Optional text generation for explanations (e.g., "This crop is recommended because...")
        self.explanation_generator = nn.GRU(
            input_size=emb_dim, 
            hidden_size=emb_dim, 
            num_layers=2,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        self.text_projection = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, vocab_size)
        )
        
        # Initialize weights for better convergence
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using modern techniques for better training stability"""
        for name, p in self.named_parameters():
            if 'weight' in name and p.dim() > 1:
                # Use He initialization for ReLU/GELU layers
                if 'ffn' in name or 'head' in name:
                    nn.init.kaiming_normal_(p, nonlinearity='relu')
                # Use Xavier for others
                else:
                    nn.init.xavier_uniform_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
                
    def forward(self, x, task_id, text_seq=None):
        # Handle single samples vs batches
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Apply soil feature encoding 
        batch_size = x.shape[0]
        x = self.soil_encoder(x)  # Result: [batch_size, emb_dim]
        x = x.unsqueeze(1)  # Add sequence dimension: [batch_size, 1, emb_dim]
        
        # Get task embedding and add to input
        task_emb = self.task_projection(self.task_embedding(task_id)).unsqueeze(1)
        x = x + task_emb  # Feature + task conditioning
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Get output embeddings
        embedding = x.squeeze(1)  # [batch_size, emb_dim]
        
        # Compute crop recommendation logits
        crop_logits = self.crop_head(embedding)
        
        # Compute fertilizer recommendation logits with nutrient specialization
        fert_features = self.fert_common(embedding)
        fert_n_logits = self.fert_n_layer(fert_features)
        fert_pk_logits = self.fert_pk_layer(fert_features)
        fert_combined = torch.cat([fert_n_logits, fert_pk_logits], dim=1)
        fert_logits = self.fert_combine(fert_combined)
        
        outputs = {
            "crop": crop_logits,
            "fertilizer": fert_logits,
            "embedding": embedding
        }
        
        # Optional explanation generation
        if text_seq is not None:
            hidden = embedding.unsqueeze(0).repeat(self.explanation_generator.num_layers, 1, 1)
            dec_out, _ = self.explanation_generator(text_seq, hidden)
            text_logits = self.text_projection(dec_out)
            outputs["text"] = text_logits
            
        return outputs


class AgriMultiTaskModel(nn.Module):
    """
    A multi-task model that can handle multiple agricultural tasks including:
    - Crop recommendation
    - Fertilizer recommendation
    - Yield prediction
    - Climate impact analysis
    """
    
    def __init__(
        self,
        input_dim,
        emb_dim=128,
        hidden_dim=256,
        num_heads=4,
        crop_classes=10,
        fert_classes=5,
        climate_effect_classes=3,
        yield_bins=5,
        num_layers=3,
        dropout=0.1
    ):
        super(AgriMultiTaskModel, self).__init__()
        
        # Feature encoder
        self.soil_encoder = SoilFeatureEncoder(input_dim, emb_dim, dropout)
        
        # Task embedding
        self.task_embedding = nn.Embedding(10, 32)  # Supporting up to 10 tasks
        self.task_projection = nn.Linear(32, emb_dim)
        
        # Transformer backbone
        self.transformer_blocks = nn.ModuleList([
            AgriTransformerBlock(emb_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Task-specific heads
        self.crop_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, crop_classes)
        )
        
        self.fert_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fert_classes)
        )
        
        self.climate_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, climate_effect_classes)
        )
        
        self.yield_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Regression output
        )
        
    def forward(self, x, task_id):
        # Handle single samples
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Encode features
        batch_size = x.shape[0]
        x = self.soil_encoder(x)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        # Add task embedding
        task_emb = self.task_projection(self.task_embedding(task_id)).unsqueeze(1)
        x = x + task_emb
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Get output embeddings
        embedding = x.squeeze(1)
        
        # Apply task-specific heads
        crop_logits = self.crop_head(embedding)
        fert_logits = self.fert_head(embedding)
        climate_logits = self.climate_head(embedding)
        yield_pred = self.yield_head(embedding)
        
        return {
            "crop": crop_logits,
            "fertilizer": fert_logits,
            "climate": climate_logits,
            "yield": yield_pred,
            "embedding": embedding
        }


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
        
        # Get top 3 predictions for each task
        _, top3_crop_indices = torch.topk(crop_probs, 3, dim=1)
        _, top3_fert_indices = torch.topk(fert_probs, 3, dim=1)
        
        top3_crop_probs = torch.gather(crop_probs, 1, top3_crop_indices)
        top3_fert_probs = torch.gather(fert_probs, 1, top3_fert_indices)
        
        return {
            'crop_pred': crop_pred.item(),
            'crop_confidence': crop_conf.item(),
            'top3_crops': [(idx.item(), prob.item()) for idx, prob in zip(top3_crop_indices[0], top3_crop_probs[0])],
            'fertilizer_pred': fert_pred.item(),
            'fertilizer_confidence': fert_conf.item(),
            'top3_fertilizers': [(idx.item(), prob.item()) for idx, prob in zip(top3_fert_indices[0], top3_fert_probs[0])],
            'embeddings': outputs['embedding'].cpu()
        }