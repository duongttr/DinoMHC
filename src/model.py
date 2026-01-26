import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


# START==========================================================================
# Additional Helper Modules
# ===============================================================================
class LayerNorm(nn.Module):
    """Layer normalization with optional bias."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / (std + self.eps) + self.bias
    
# END============================================================================
# Additional Helper Modules
# ===============================================================================




# START==========================================================================
# Hierarchical Groove Extractor
# ===============================================================================
class HierarchicalGrooveExtractor(nn.Module):
    """
    Hierarchical Groove Extraction Module.
    
    Level 1 (Coarse): Identify relevant MHC domains (α1, α2)
    Level 2 (Fine): Extract specific groove residues using peptide guidance
    """
    def __init__(
        self,
        dim: int,
        num_groove_tokens: int = 60,
        num_regions: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_groove_tokens = num_groove_tokens
        self.num_regions = num_regions
        
        # Level 1: Region identification
        self.region_queries = nn.Parameter(torch.randn(num_regions, dim))
        self.region_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.region_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        self.region_norm = LayerNorm(dim)
        
        # Level 2: Fine-grained groove extraction
        self.groove_queries = nn.Parameter(torch.randn(num_groove_tokens, dim))
        self.peptide_modulation = nn.Linear(dim, dim)
        self.groove_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.groove_norm = LayerNorm(dim)
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
        self.output_norm = LayerNorm(dim)
    
    def forward(
        self,
        mhc_emb: torch.Tensor,              # [batch, mhc_len, dim]
        peptide_emb: Optional[torch.Tensor] = None,  # [batch, pep_len, dim]
        mhc_mask: Optional[torch.Tensor] = None,     # [batch, mhc_len] True=valid
        peptide_mask: Optional[torch.Tensor] = None, # [batch, pep_len] True=valid
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Extract groove features from MHC using hierarchical attention.

        Args:
            mhc_emb: Full MHC embeddings
            peptide_emb: Optional peptide embeddings for guided extraction
            mhc_mask: Boolean mask where True indicates valid (non-padding) tokens
            peptide_mask: Boolean mask where True indicates valid (non-padding) tokens
            return_attention: Whether to return attention weights

        Returns:
            groove_features: [batch, num_groove_tokens, dim]
            attention_info: Optional dict with attention weights
        """
        batch_size = mhc_emb.shape[0]
        attention_info = {} if return_attention else None

        # Convert boolean masks to key_padding_mask format (True = padding)
        mhc_key_padding_mask = ~mhc_mask if mhc_mask is not None else None

        # === Level 1: Region Identification ===
        # Expand region queries for batch
        region_queries = self.region_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Attend to MHC to identify regions        
        region_features, region_attn = self.region_attention(
            region_queries, mhc_emb, mhc_emb,
            key_padding_mask=mhc_key_padding_mask,
            average_attn_weights=False  # Get per-head attention
        )
        region_features = self.region_norm(region_features)

        # Compute region importance (which regions to focus on)
        # region_attn shape: [batch, num_heads, num_regions, mhc_len]
        region_importance = region_attn.mean(dim=1)  # [batch, num_regions, mhc_len]

        # Weight MHC by region importance
        weighted_mhc = torch.einsum(
            'brl,bld->brd', region_importance, mhc_emb
        )  # [batch, num_regions, dim]

        # Gate and combine with original
        gate = self.region_gate(weighted_mhc)
        mhc_region_enhanced = mhc_emb + torch.einsum(
            'brd,brl->bld', gate * weighted_mhc, region_importance
        )

        # === Level 2: Fine-grained Groove Extraction ===
        groove_queries = self.groove_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Modulate queries with peptide information if available
        if peptide_emb is not None:
            # Proper masked mean pooling
            if peptide_mask is not None:
                masked_peptide = peptide_emb * peptide_mask.unsqueeze(-1).to(peptide_emb.dtype)
                pep_lengths = peptide_mask.sum(dim=1, keepdim=True).to(peptide_emb.dtype).clamp(min=1)
                pep_context = masked_peptide.sum(dim=1, keepdim=True) / pep_lengths.unsqueeze(-1)
            else:
                pep_context = peptide_emb.mean(dim=1, keepdim=True)  # [batch, 1, dim]

            pep_modulation = self.peptide_modulation(pep_context)
            groove_queries = groove_queries + pep_modulation

        # Attend to region-enhanced MHC
        groove_features, groove_attn = self.groove_attention(
            groove_queries, mhc_region_enhanced, mhc_region_enhanced,
            key_padding_mask=mhc_key_padding_mask,
            average_attn_weights=False  # Get per-head attention
        )
        groove_features = self.groove_norm(groove_features)

        # Output projection
        groove_features = self.output_proj(groove_features)
        groove_features = self.output_norm(groove_features)

        if return_attention:
            attention_info = {
                'region_attention': region_attn,      # [batch, num_heads, num_regions, mhc_len]
                'groove_attention': groove_attn,      # [batch, num_heads, num_groove_tokens, mhc_len]
                'region_importance': region_importance # [batch, num_regions, mhc_len]
            }

        return groove_features, attention_info
# END============================================================================
# Hierarchical Groove Extractor
# ===============================================================================






# START==========================================================================
# Cross-Attention with Geometric Bias
# ===============================================================================
class GeometricBiasModule(nn.Module):
    """
    Learns geometric/distance biases between peptide and groove positions.
    
    Inspired by AlphaFold's pair representation, this module creates
    position-dependent attention biases that capture spatial relationships
    at the binding interface.
    
    The key insight: certain peptide positions (e.g., anchors P2, P9) 
    preferentially interact with specific groove pockets. This module
    learns these patterns from data.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_query_len: int = 15,  # Max peptide length
        max_key_len: int = 60     # Max groove tokens
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.max_query_len = max_query_len
        self.max_key_len = max_key_len
        hidden_dim = hidden_dim or dim // 2

        # Project concatenated pair features to bias per head
        self.pair_encoder = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads)
        )

        # Position-specific learned biases
        # query_pos_bias[h, i, 0] = bias for head h at query position i
        # key_pos_bias[h, 0, j] = bias for head h at key position j
        self.query_pos_bias = nn.Parameter(torch.zeros(1, num_heads, max_query_len, 1))
        self.key_pos_bias = nn.Parameter(torch.zeros(1, num_heads, 1, max_key_len))
    
    def forward(
        self, 
        query_emb: torch.Tensor,  # [batch, query_len, dim]
        key_emb: torch.Tensor     # [batch, key_len, dim]
    ) -> torch.Tensor:
        """
        Compute geometric attention bias.
        
        Returns:
            bias: [batch, num_heads, query_len, key_len]
        """
        batch_size, query_len, _ = query_emb.shape
        _, key_len, _ = key_emb.shape
        
        # Create all pairs by broadcasting
        # query_expanded: [batch, query_len, 1, dim]
        # key_expanded: [batch, 1, key_len, dim]
        query_expanded = query_emb.unsqueeze(2)
        key_expanded = key_emb.unsqueeze(1)
        
        # Expand to create pair matrix
        query_expanded = query_expanded.expand(-1, -1, key_len, -1)
        key_expanded = key_expanded.expand(-1, query_len, -1, -1)
        
        # Concatenate: [batch, query_len, key_len, dim*2]
        pairs = torch.cat([query_expanded, key_expanded], dim=-1)
        
        # Encode pairs to get bias: [batch, query_len, key_len, num_heads]
        bias = self.pair_encoder(pairs)

        # Reshape to attention format: [batch, num_heads, query_len, key_len]
        bias = bias.permute(0, 3, 1, 2)

        # Add position-specific learned biases
        # Slice to actual sequence lengths and broadcast appropriately
        # Handle cases where sequence is longer than max_query_len/max_key_len
        # by using min to avoid index errors
        query_bias_len = min(query_len, self.max_query_len)
        key_bias_len = min(key_len, self.max_key_len)

        # query_pos_bias: [1, num_heads, query_bias_len, 1] broadcasts over key_len dimension
        qpb = self.query_pos_bias[..., :query_bias_len, :]
        # If query is longer than max, pad the bias with zeros
        if query_len > self.max_query_len:
            pad_size = query_len - self.max_query_len
            qpb = torch.cat([qpb, torch.zeros(1, self.num_heads, pad_size, 1, device=qpb.device, dtype=qpb.dtype)], dim=2)
        bias = bias + qpb

        # key_pos_bias: [1, num_heads, 1, key_bias_len] broadcasts over query_len dimension
        kpb = self.key_pos_bias[..., :, :key_bias_len]
        # If key is longer than max, pad the bias with zeros
        if key_len > self.max_key_len:
            pad_size = key_len - self.max_key_len
            kpb = torch.cat([kpb, torch.zeros(1, self.num_heads, 1, pad_size, device=kpb.device, dtype=kpb.dtype)], dim=3)
        bias = bias + kpb

        return bias

class CrossAttentionWithGeometricBias(nn.Module):
    """
    Cross-attention module with geometric bias.
    
    This is the core attention mechanism used in Groove-Peptide Fusion.
    It allows one sequence (query) to attend to another (key/value)
    with learned geometric biases that capture binding patterns.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_geometric_bias: bool = True,
        max_query_len: int = 15,  # Max peptide length
        max_key_len: int = 60     # Max groove tokens
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_geometric_bias = use_geometric_bias

        # Linear projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, dim)

        # Geometric bias module with position-specific biases
        if use_geometric_bias:
            self.geometric_bias = GeometricBiasModule(
                dim, num_heads, dropout=dropout,
                max_query_len=max_query_len,
                max_key_len=max_key_len
            )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(dim)
    
    def forward(
        self,
        query: torch.Tensor,           # [batch, query_len, dim]
        key_value: torch.Tensor,       # [batch, kv_len, dim]
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Perform cross-attention with geometric bias.
        
        Args:
            query: Query sequence embeddings
            key_value: Key/Value sequence embeddings
            attention_mask: Optional mask [batch, 1, query_len, kv_len]
            return_attention: Whether to return attention weights
        
        Returns:
            output: Attended output [batch, query_len, dim]
            attention_weights: Optional attention weights
        """
        batch_size, query_len, _ = query.shape
        _, kv_len, _ = key_value.shape
        
        # Project to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key_value)
        V = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        # [batch, len, num_heads * head_dim] -> [batch, num_heads, len, head_dim]
        Q = Q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [batch, num_heads, query_len, kv_len]
        
        # Add geometric bias
        if self.use_geometric_bias:
            geo_bias = self.geometric_bias(query, key_value)
            attn_scores = attn_scores + geo_bias
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # [batch, num_heads, query_len, head_dim]
        
        # Reshape back
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, query_len, self.num_heads * self.head_dim)
        
        # Output projection
        output = self.out_proj(output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None

class GroovePeptideFusionLayer(nn.Module):
    """
    Single layer of Groove-Peptide Fusion.
    
    Implements bidirectional cross-attention:
    1. Peptide attends to Groove (peptide queries, groove provides context)
    2. Groove attends to Peptide (groove queries, peptide provides context)
    
    Plus self-attention for each stream and feed-forward networks.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_peptide_length: int = 15,
        num_groove_tokens: int = 60
    ):
        super().__init__()
        self.dim = dim
        ff_dim = ff_dim or dim * 4

        # Self-attention for peptide
        self.peptide_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.peptide_self_norm = LayerNorm(dim)

        # Self-attention for groove
        self.groove_self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.groove_self_norm = LayerNorm(dim)

        # Cross-attention: Peptide -> Groove (peptide queries groove)
        self.peptide_to_groove_attn = CrossAttentionWithGeometricBias(
            dim, num_heads, dropout=dropout, use_geometric_bias=True,
            max_query_len=max_peptide_length,
            max_key_len=num_groove_tokens
        )
        self.peptide_cross_norm = LayerNorm(dim)

        # Cross-attention: Groove -> Peptide (groove queries peptide)
        self.groove_to_peptide_attn = CrossAttentionWithGeometricBias(
            dim, num_heads, dropout=dropout, use_geometric_bias=True,
            max_query_len=num_groove_tokens,
            max_key_len=max_peptide_length
        )
        self.groove_cross_norm = LayerNorm(dim)
        
        # Feed-forward networks
        self.peptide_ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.peptide_ffn_norm = LayerNorm(dim)
        
        self.groove_ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout)
        )
        self.groove_ffn_norm = LayerNorm(dim)
    
    def forward(
        self,
        peptide_emb: torch.Tensor,  # [batch, pep_len, dim]
        groove_emb: torch.Tensor,   # [batch, groove_len, dim]
        peptide_mask: Optional[torch.Tensor] = None,  # [batch, pep_len] True=valid
        groove_mask: Optional[torch.Tensor] = None,   # [batch, groove_len] True=valid
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through fusion layer.
        
        Args:
            peptide_emb: Peptide embeddings [batch, pep_len, dim]
            groove_emb: Groove embeddings [batch, groove_len, dim]
            peptide_mask: Boolean mask where True = valid, False = padding
            groove_mask: Boolean mask where True = valid, False = padding
            return_attention: Whether to return attention weights
        
        Returns:
            peptide_out: Enhanced peptide embeddings
            groove_out: Enhanced groove embeddings
            attention_info: Optional dict with attention weights
        """
        attention_info = {} if return_attention else None
        
        # Convert to key_padding_mask format (True = padding) for nn.MultiheadAttention
        pep_key_padding_mask = ~peptide_mask if peptide_mask is not None else None
        groove_key_padding_mask = ~groove_mask if groove_mask is not None else None
        
        # Create cross-attention masks [batch, 1, query_len, kv_len]
        # For peptide->groove: peptide queries, groove is key/value
        pep_to_groove_mask = None
        if groove_mask is not None:
            pep_to_groove_mask = groove_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, groove_len]
        
        # For groove->peptide: groove queries, peptide is key/value  
        groove_to_pep_mask = None
        if peptide_mask is not None:
            groove_to_pep_mask = peptide_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, pep_len]
        
        # === Peptide Stream ===
        # Self-attention
        pep_self, _ = self.peptide_self_attn(
            peptide_emb, peptide_emb, peptide_emb,
            key_padding_mask=pep_key_padding_mask
        )
        peptide_emb = self.peptide_self_norm(peptide_emb + pep_self)
        
        # Cross-attention (peptide queries groove)
        pep_cross, pep_to_groove_attn = self.peptide_to_groove_attn(
            peptide_emb, groove_emb, 
            attention_mask=pep_to_groove_mask,
            return_attention=return_attention
        )
        peptide_emb = self.peptide_cross_norm(peptide_emb + pep_cross)
        
        # Feed-forward
        peptide_emb = self.peptide_ffn_norm(peptide_emb + self.peptide_ffn(peptide_emb))
        
        # === Groove Stream ===
        # Self-attention
        groove_self, _ = self.groove_self_attn(
            groove_emb, groove_emb, groove_emb,
            key_padding_mask=groove_key_padding_mask
        )
        groove_emb = self.groove_self_norm(groove_emb + groove_self)
        
        # Cross-attention (groove queries peptide)
        groove_cross, groove_to_pep_attn = self.groove_to_peptide_attn(
            groove_emb, peptide_emb, 
            attention_mask=groove_to_pep_mask,
            return_attention=return_attention
        )
        groove_emb = self.groove_cross_norm(groove_emb + groove_cross)
        
        # Feed-forward
        groove_emb = self.groove_ffn_norm(groove_emb + self.groove_ffn(groove_emb))
        
        # Zero out padded positions in output
        if peptide_mask is not None:
            peptide_emb = peptide_emb * peptide_mask.unsqueeze(-1).to(peptide_emb.dtype)
        if groove_mask is not None:
            groove_emb = groove_emb * groove_mask.unsqueeze(-1).to(groove_emb.dtype)
        
        if return_attention:
            attention_info = {
                'peptide_to_groove': pep_to_groove_attn,
                'groove_to_peptide': groove_to_pep_attn
            }
        
        return peptide_emb, groove_emb, attention_info

class GroovePeptideFusion(nn.Module):
    """
    Complete Groove-Peptide Fusion Module.
    
    Stacks multiple fusion layers to enable deep interaction between
    peptide and groove representations. This is the core innovation
    that replaces simple concatenation with structured cross-attention.
    
    Key features:
    - Bidirectional cross-attention (both directions)
    - Geometric bias for position-aware attention
    - Multi-layer refinement
    - Interpretable attention weights
    """
    def __init__(
        self,
        dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        max_peptide_length: int = 15,
        num_groove_tokens: int = 60
    ):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers

        # Stack of fusion layers
        self.layers = nn.ModuleList([
            GroovePeptideFusionLayer(
                dim, num_heads, ff_dim, dropout,
                max_peptide_length=max_peptide_length,
                num_groove_tokens=num_groove_tokens
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norms
        self.peptide_final_norm = LayerNorm(dim)
        self.groove_final_norm = LayerNorm(dim)
    
    def forward(
        self,
        peptide_emb: torch.Tensor,  # [batch, pep_len, dim]
        groove_emb: torch.Tensor,   # [batch, groove_len, dim]
        peptide_mask: Optional[torch.Tensor] = None,  # [batch, pep_len] True=valid
        groove_mask: Optional[torch.Tensor] = None,   # [batch, groove_len] True=valid
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Forward pass through all fusion layers.
        
        Args:
            peptide_emb: Peptide embeddings [batch, pep_len, dim]
            groove_emb: Groove embeddings [batch, groove_len, dim]
            peptide_mask: Boolean mask where True = valid, False = padding
            groove_mask: Boolean mask where True = valid, False = padding
            return_attention: Whether to return attention weights from all layers
        
        Returns:
            peptide_fused: Fused peptide representation
            groove_fused: Fused groove representation
            all_attention: Dict with attention from each layer
        """
        all_attention = {} if return_attention else None
        
        for i, layer in enumerate(self.layers):
            peptide_emb, groove_emb, attn_info = layer(
                peptide_emb, groove_emb, 
                peptide_mask=peptide_mask,
                groove_mask=groove_mask,
                return_attention=return_attention
            )
            
            if return_attention:
                all_attention[f'layer_{i}'] = attn_info
        
        # Final normalization
        peptide_fused = self.peptide_final_norm(peptide_emb)
        groove_fused = self.groove_final_norm(groove_emb)
        
        # Ensure padded positions stay zeroed after final norm
        if peptide_mask is not None:
            peptide_fused = peptide_fused * peptide_mask.unsqueeze(-1).to(peptide_fused.dtype)
        if groove_mask is not None:
            groove_fused = groove_fused * groove_mask.unsqueeze(-1).to(groove_fused.dtype)
        
        return peptide_fused, groove_fused, all_attention


# END============================================================================
# Cross-Attention with Geometric Bias
# ===============================================================================



# START==========================================================================
# Interface Geometry Reasoning
# ===============================================================================
# class TriangleAttention(nn.Module):
#     """
#     Triangle Attention from AlphaFold.
    
#     Models geometric constraints by considering triplets of positions.
#     For pair representation z[i,j], attends over the third index k
#     to capture constraints like "if residue i contacts j, and j contacts k,
#     what does that imply about i-k relationship?"
    
#     Two modes:
#     - 'starting': Attention over starting node (row-wise)
#     - 'ending': Attention over ending node (column-wise)
    
#     Supports non-square pair representations with optional masking.
#     """
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 4,
#         mode: str = 'starting',
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         assert mode in ['starting', 'ending']
#         self.mode = mode
#         self.dim = dim
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
        
#         # Projections
#         self.q_proj = nn.Linear(dim, dim, bias=False)
#         self.k_proj = nn.Linear(dim, dim, bias=False)
#         self.v_proj = nn.Linear(dim, dim, bias=False)
#         self.out_proj = nn.Linear(dim, dim)
        
#         # Gate for residual
#         self.gate = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Sigmoid()
#         )
        
#         self.layer_norm = LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(
#         self, 
#         pair_rep: torch.Tensor,
#         pair_mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Apply triangle attention.
        
#         Args:
#             pair_rep: Pair representation [batch, seq_i, seq_j, dim]
#             pair_mask: Optional mask [batch, seq_i, seq_j] where True = valid, False = padding
        
#         Returns:
#             Updated pair representation
#         """
#         batch_size, seq_i, seq_j, _ = pair_rep.shape
#         original_seq_i, original_seq_j = seq_i, seq_j
        
#         # Normalize
#         z = self.layer_norm(pair_rep)
        
#         # Handle mask for attention - need to prepare before potential transpose
#         attn_mask = None
#         if pair_mask is not None:
#             if self.mode == 'starting':
#                 # For starting mode, attention is along seq_j for each (batch, seq_i)
#                 attn_mask = pair_mask.clone()  # [batch, seq_i, seq_j]
#             else:
#                 # For ending mode, we transpose first, so transpose the mask too
#                 attn_mask = pair_mask.transpose(1, 2).contiguous()  # [batch, seq_j, seq_i]
        
#         # Transpose if ending mode (attention over rows instead of columns)
#         if self.mode == 'ending':
#             z = z.transpose(1, 2).contiguous()
#             seq_i, seq_j = seq_j, seq_i
        
#         # Project to Q, K, V
#         Q = self.q_proj(z)  # [batch, seq_i, seq_j, dim]
#         K = self.k_proj(z)
#         V = self.v_proj(z)
        
#         # Reshape for multi-head attention
#         # [batch, seq_i, seq_j, num_heads, head_dim]
#         Q = Q.view(batch_size, seq_i, seq_j, self.num_heads, self.head_dim)
#         K = K.view(batch_size, seq_i, seq_j, self.num_heads, self.head_dim)
#         V = V.view(batch_size, seq_i, seq_j, self.num_heads, self.head_dim)
        
#         # Attention is computed along seq_j dimension for each seq_i
#         # Reshape: [batch * seq_i, seq_j, num_heads, head_dim]
#         Q = Q.reshape(batch_size * seq_i, seq_j, self.num_heads, self.head_dim)
#         K = K.reshape(batch_size * seq_i, seq_j, self.num_heads, self.head_dim)
#         V = V.reshape(batch_size * seq_i, seq_j, self.num_heads, self.head_dim)
        
#         # [batch * seq_i, num_heads, seq_j, head_dim]
#         Q = Q.transpose(1, 2)
#         K = K.transpose(1, 2)
#         V = V.transpose(1, 2)
        
#         # Compute attention: [batch * seq_i, num_heads, seq_j, seq_j]
#         attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
#         # Apply attention mask if provided
#         if attn_mask is not None:
#             # Reshape mask: [batch, seq_i, seq_j] -> [batch * seq_i, 1, 1, seq_j]
#             mask_reshaped = attn_mask.reshape(batch_size * seq_i, 1, 1, seq_j)
#             attn_scores = attn_scores.masked_fill(~mask_reshaped, float('-inf'))
        
#         attn_weights = F.softmax(attn_scores, dim=-1)
        
#         # Handle case where all positions are masked (avoid NaN)
#         attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
#         attn_weights = self.dropout(attn_weights)
        
#         # Apply to values: [batch * seq_i, num_heads, seq_j, head_dim]
#         out = torch.matmul(attn_weights, V)
        
#         # Reshape back: [batch, seq_i, seq_j, dim]
#         out = out.transpose(1, 2)  # [batch * seq_i, seq_j, num_heads, head_dim]
#         out = out.contiguous().reshape(batch_size, seq_i, seq_j, self.dim)
        
#         # Output projection
#         out = self.out_proj(out)
        
#         # Transpose back if ending mode
#         if self.mode == 'ending':
#             out = out.transpose(1, 2).contiguous()
        
#         # Gated residual
#         gate = self.gate(pair_rep)
#         out = pair_rep + gate * self.dropout(out)
        
#         return out

# class TriangleMultiplication(nn.Module):
#     """
#     Triangle Multiplication from AlphaFold.
    
#     Updates pair representation by combining information from two edges
#     that share a common node, modeling transitive relationships:
#     z[i,j] is updated based on z[i,k] and z[k,j] (or z[j,k]) for all k.
    
#     Two modes:
#     - 'outgoing': Uses z[i,k] * z[j,k] (both edges going out from k)
#     - 'incoming': Uses z[k,i] * z[k,j] (both edges coming into k)
    
#     Supports non-square pair representations with optional masking.
#     The mask is used to zero out contributions from padded positions.
#     """
#     def __init__(
#         self,
#         dim: int,
#         hidden_dim: Optional[int] = None,
#         mode: str = 'outgoing',
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         assert mode in ['outgoing', 'incoming']
#         self.mode = mode
#         self.dim = dim
#         hidden_dim = hidden_dim or dim
#         self.hidden_dim = hidden_dim
        
#         # Projections for the two edges
#         self.left_proj = nn.Linear(dim, hidden_dim)
#         self.right_proj = nn.Linear(dim, hidden_dim)
        
#         # Gates
#         self.left_gate = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.Sigmoid()
#         )
#         self.right_gate = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.Sigmoid()
#         )
        
#         # Output projection
#         self.out_proj = nn.Linear(hidden_dim, dim)
        
#         # Final gate
#         self.output_gate = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.Sigmoid()
#         )
        
#         self.layer_norm = LayerNorm(dim)
#         self.output_norm = LayerNorm(hidden_dim)
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(
#         self, 
#         pair_rep: torch.Tensor,
#         pair_mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Apply triangle multiplication.
        
#         Args:
#             pair_rep: Pair representation [batch, seq_i, seq_j, dim]
#                       For non-square matrices, this should be padded to square.
#             pair_mask: Optional mask [batch, seq_i, seq_j] where True = valid, False = padding
        
#         Returns:
#             Updated pair representation
#         """
#         batch_size, seq_i, seq_j, _ = pair_rep.shape
        
#         # Normalize
#         z = self.layer_norm(pair_rep)
        
#         # Project and gate
#         left = self.left_proj(z) * self.left_gate(z)
#         right = self.right_proj(z) * self.right_gate(z)
        
#         # Apply mask to zero out padded positions before multiplication
#         if pair_mask is not None:
#             # Expand mask to match hidden dim: [batch, seq_i, seq_j, 1]
#             mask_expanded = pair_mask.unsqueeze(-1).float()
#             left = left * mask_expanded
#             right = right * mask_expanded
        
#         if self.mode == 'outgoing':
#             # z[i,k] * z[j,k] -> update z[i,j]
#             # left: [batch, seq_i, seq_j, hidden] where we use as z[i,k]
#             # right: [batch, seq_i, seq_j, hidden] where we use as z[j,k]
            
#             # Einsum: sum over k
#             # left[i,k] * right[j,k] summed over k
#             out = torch.einsum('bikd,bjkd->bijd', left, right)
#         else:  # incoming
#             # z[k,i] * z[k,j] -> update z[i,j]
#             out = torch.einsum('bkid,bkjd->bijd', left, right)
        
#         # Normalize and project
#         out = self.output_norm(out)
#         out = self.out_proj(out)
        
#         # Apply mask to output (zero out padded positions)
#         if pair_mask is not None:
#             out = out * mask_expanded
        
#         # Gated residual
#         gate = self.output_gate(pair_rep)
#         out = pair_rep + gate * self.dropout(out)
        
#         return out


# class PairTransition(nn.Module):
#     """
#     Feed-forward transition for pair representation.
#     Applied after triangle operations for additional refinement.
#     """
#     def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0.1):
#         super().__init__()
#         hidden_dim = dim * expansion_factor
        
#         self.layer_norm = LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
    
#     def forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
#         return pair_rep + self.ffn(self.layer_norm(pair_rep))


# class InterfaceGeometryBlock(nn.Module):
#     """
#     Single block of Interface Geometry Reasoning.
    
#     Combines all triangle operations:
#     1. Triangle Attention (starting)
#     2. Triangle Attention (ending)
#     3. Triangle Multiplication (outgoing)
#     4. Triangle Multiplication (incoming)
#     5. Pair Transition (FFN)
    
#     Supports masked pair representations for padded non-square matrices.
#     """
#     def __init__(
#         self,
#         dim: int,
#         num_heads: int = 4,
#         dropout: float = 0.1
#     ):
#         super().__init__()
        
#         self.triangle_attn_starting = TriangleAttention(
#             dim, num_heads, mode='starting', dropout=dropout
#         )
#         self.triangle_attn_ending = TriangleAttention(
#             dim, num_heads, mode='ending', dropout=dropout
#         )
#         self.triangle_mult_outgoing = TriangleMultiplication(
#             dim, mode='outgoing', dropout=dropout
#         )
#         self.triangle_mult_incoming = TriangleMultiplication(
#             dim, mode='incoming', dropout=dropout
#         )
#         self.pair_transition = PairTransition(dim, dropout=dropout)
    
#     def forward(
#         self, 
#         pair_rep: torch.Tensor,
#         pair_mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Apply full block of geometry reasoning.
        
#         Args:
#             pair_rep: [batch, seq_i, seq_j, dim] (should be square if using triangle multiplication)
#             pair_mask: Optional mask [batch, seq_i, seq_j] where True = valid, False = padding
        
#         Returns:
#             Updated pair representation
#         """
#         pair_rep = self.triangle_attn_starting(pair_rep, pair_mask)
#         pair_rep = self.triangle_attn_ending(pair_rep, pair_mask)
#         pair_rep = self.triangle_mult_outgoing(pair_rep, pair_mask)
#         pair_rep = self.triangle_mult_incoming(pair_rep, pair_mask)
#         pair_rep = self.pair_transition(pair_rep)
        
#         # Apply mask after transition to ensure padded positions stay zeroed
#         if pair_mask is not None:
#             pair_rep = pair_rep * pair_mask.unsqueeze(-1).float()
        
#         return pair_rep


# class InterfaceGeometryModule(nn.Module):
#     """
#     Complete Interface Geometry Reasoning Module.
    
#     This module operates on the pair representation (peptide x groove)
#     to model spatial/geometric constraints at the binding interface.
    
#     Inspired by AlphaFold's Evoformer, it uses triangle attention and
#     triangle multiplication to reason about relationships between
#     triplets of residues, capturing constraints like:
#     - "If P2 contacts groove residue A, and A is near residue B,
#        then P2 might also interact with B"
    
#     Key insight: The binding interface has geometric constraints that
#     go beyond pairwise interactions. Triangle operations capture these.
    
#     Supports non-square pair representations by padding to square and
#     using masks to handle the padded positions correctly.
#     """
#     def __init__(
#         self,
#         dim: int,
#         num_blocks: int = 3,
#         num_heads: int = 4,
#         dropout: float = 0.1
#     ):
#         super().__init__()
#         self.dim = dim
        
#         # Initial pair representation projection
#         self.pair_proj = nn.Linear(dim * 2, dim)
        
#         # Stack of geometry blocks
#         self.blocks = nn.ModuleList([
#             InterfaceGeometryBlock(dim, num_heads, dropout)
#             for _ in range(num_blocks)
#         ])
        
#         # Final normalization
#         self.final_norm = LayerNorm(dim)
    
#     def create_pair_representation(
#         self,
#         peptide_emb: torch.Tensor,  # [batch, pep_len, dim]
#         groove_emb: torch.Tensor    # [batch, groove_len, dim]
#     ) -> torch.Tensor:
#         """
#         Create pair representation from peptide and groove embeddings.
        
#         Each element pair_rep[i,j] represents the relationship between
#         peptide residue i and groove residue j.
#         """
#         batch_size, pep_len, dim = peptide_emb.shape
#         _, groove_len, _ = groove_emb.shape
        
#         # Expand and concatenate
#         # peptide: [batch, pep_len, 1, dim] -> [batch, pep_len, groove_len, dim]
#         pep_expanded = peptide_emb.unsqueeze(2).expand(-1, -1, groove_len, -1)
        
#         # groove: [batch, 1, groove_len, dim] -> [batch, pep_len, groove_len, dim]
#         groove_expanded = groove_emb.unsqueeze(1).expand(-1, pep_len, -1, -1)
        
#         # Concatenate: [batch, pep_len, groove_len, dim*2]
#         pair_concat = torch.cat([pep_expanded, groove_expanded], dim=-1)
        
#         # Project to dim: [batch, pep_len, groove_len, dim]
#         pair_rep = self.pair_proj(pair_concat)
        
#         return pair_rep
    
#     def _pad_to_square(
#         self,
#         pair_rep: torch.Tensor,  # [batch, pep_len, groove_len, dim]
#         pep_len: int,
#         groove_len: int
#     ) -> Tuple[torch.Tensor, torch.Tensor, int]:
#         """
#         Pad pair representation to square matrix for triangle operations.
        
#         Args:
#             pair_rep: Non-square pair representation [batch, pep_len, groove_len, dim]
#             pep_len: Original peptide length
#             groove_len: Original groove length
        
#         Returns:
#             padded_pair: Square pair representation [batch, max_len, max_len, dim]
#             pair_mask: Boolean mask [batch, max_len, max_len] where True = valid
#             max_len: The size of the square matrix
#         """
#         batch_size = pair_rep.shape[0]
#         max_len = max(pep_len, groove_len)
#         device = pair_rep.device
        
#         if pep_len == groove_len:
#             # Already square, create full valid mask
#             pair_mask = torch.ones(batch_size, max_len, max_len, dtype=torch.bool, device=device)
#             return pair_rep, pair_mask, max_len
        
#         # Create padded tensor
#         padded_pair = torch.zeros(
#             batch_size, max_len, max_len, self.dim,
#             dtype=pair_rep.dtype, device=device
#         )
        
#         # Copy original data to top-left corner
#         padded_pair[:, :pep_len, :groove_len, :] = pair_rep
        
#         # Create mask: True for valid positions, False for padding
#         pair_mask = torch.zeros(batch_size, max_len, max_len, dtype=torch.bool, device=device)
#         pair_mask[:, :pep_len, :groove_len] = True
        
#         # For triangle operations to work properly, we also need to handle
#         # the "virtual" positions. We fill diagonal of padding region with
#         # a copy of the mean representation to provide reasonable values
#         # for the einsum operations (they will be masked out anyway).
#         if pep_len < groove_len:
#             # Pad rows (peptide dimension)
#             # Fill padded rows with mean of existing rows for numerical stability
#             mean_row = pair_rep.mean(dim=1, keepdim=True)  # [batch, 1, groove_len, dim]
#             for i in range(pep_len, max_len):
#                 if i < groove_len:
#                     padded_pair[:, i, :groove_len, :] = mean_row.squeeze(1)
#         else:
#             # Pad columns (groove dimension)
#             # Fill padded columns with mean of existing columns for numerical stability
#             mean_col = pair_rep.mean(dim=2, keepdim=True)  # [batch, pep_len, 1, dim]
#             for j in range(groove_len, max_len):
#                 if j < pep_len:
#                     padded_pair[:, :pep_len, j, :] = mean_col.squeeze(2)
        
#         return padded_pair, pair_mask, max_len
    
#     def _unpad_from_square(
#         self,
#         padded_pair: torch.Tensor,  # [batch, max_len, max_len, dim]
#         pep_len: int,
#         groove_len: int
#     ) -> torch.Tensor:
#         """
#         Extract original non-square region from padded square matrix.
        
#         Args:
#             padded_pair: Square pair representation [batch, max_len, max_len, dim]
#             pep_len: Original peptide length
#             groove_len: Original groove length
        
#         Returns:
#             pair_rep: Original shape [batch, pep_len, groove_len, dim]
#         """
#         return padded_pair[:, :pep_len, :groove_len, :]
    
#     def forward(
#         self,
#         peptide_emb: torch.Tensor,
#         groove_emb: torch.Tensor,
#         peptide_mask: Optional[torch.Tensor] = None,  # [batch, pep_len] True=valid
#         groove_mask: Optional[torch.Tensor] = None,   # [batch, groove_len] True=valid
#         return_pair_rep: bool = False
#     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """
#         Apply interface geometry reasoning.
        
#         Handles non-square pair representations by:
#         1. Creating initial pair representation [batch, pep_len, groove_len, dim]
#         2. Masking out padding positions in the pair representation
#         3. Padding to square [batch, max_len, max_len, dim] with mask
#         4. Applying triangle operations with masking
#         5. Extracting original region for output
        
#         Args:
#             peptide_emb: [batch, pep_len, dim]
#             groove_emb: [batch, groove_len, dim]
#             peptide_mask: Boolean mask where True = valid, False = padding
#             groove_mask: Boolean mask where True = valid, False = padding
#             return_pair_rep: Whether to return the full pair representation
        
#         Returns:
#             interface_rep: Pooled interface representation [batch, dim]
#             pair_rep: Optional full pair representation [batch, pep_len, groove_len, dim]
#         """
#         batch_size, pep_len, _ = peptide_emb.shape
#         _, groove_len, _ = groove_emb.shape
        
#         # Create pair representation (non-square)
#         pair_rep = self.create_pair_representation(peptide_emb, groove_emb)
#         # pair_rep: [batch, pep_len, groove_len, dim]
        
#         # Determine valid lengths from masks
#         if peptide_mask is not None:
#             # Get number of valid peptide positions (assume same for all batch items)
#             valid_pep_len = peptide_mask[0].sum().item()
#         else:
#             valid_pep_len = pep_len
            
#         if groove_mask is not None:
#             valid_groove_len = groove_mask[0].sum().item()
#         else:
#             valid_groove_len = groove_len
        
#         # Create combined pair mask from individual masks
#         # pair_mask[b, i, j] = peptide_mask[b, i] AND groove_mask[b, j]
#         if peptide_mask is not None or groove_mask is not None:
#             if peptide_mask is None:
#                 peptide_mask = torch.ones(batch_size, pep_len, dtype=torch.bool, device=peptide_emb.device)
#             if groove_mask is None:
#                 groove_mask = torch.ones(batch_size, groove_len, dtype=torch.bool, device=groove_emb.device)
            
#             # Create pair mask: [batch, pep_len, groove_len]
#             input_pair_mask = peptide_mask.unsqueeze(2) & groove_mask.unsqueeze(1)
            
#             # Zero out padding positions in pair_rep
#             pair_rep = pair_rep * input_pair_mask.unsqueeze(-1)
#         else:
#             input_pair_mask = None
        
#         # Slice to valid length to ensure consistent behavior with sliced inputs
#         pair_rep = pair_rep[:, :valid_pep_len, :valid_groove_len, :]
#         pep_len = valid_pep_len
#         groove_len = valid_groove_len
        
#         # Pad to square for triangle operations
#         padded_pair, pair_mask, max_len = self._pad_to_square(
#             pair_rep, pep_len, groove_len
#         )
#         # padded_pair: [batch, max_len, max_len, dim]
#         # pair_mask: [batch, max_len, max_len]
        
#         # Apply geometry blocks with masking
#         for block in self.blocks:
#             padded_pair = block(padded_pair, pair_mask)
        
#         # Final normalization
#         padded_pair = self.final_norm(padded_pair)
        
#         # Extract original (non-padded) region
#         pair_rep_out = self._unpad_from_square(padded_pair, pep_len, groove_len)
#         # pair_rep_out: [batch, pep_len, groove_len, dim]
        
#         # Pool to get interface representation (only over valid positions)
#         interface_rep = pair_rep_out.mean(dim=[1, 2])  # [batch, dim]
        
#         if return_pair_rep:
#             return interface_rep, pair_rep_out
#         return interface_rep, None

def apply_pair_mask(pair_rep: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    """
    Zero out padding positions in pair representation.
    
    CRITICAL: Must be called after every operation to prevent
    information leakage from padding positions.
    
    Args:
        pair_rep: [batch, seq, seq, dim]
        pair_mask: [batch, seq, seq] boolean, True = valid
    
    Returns:
        Masked pair representation
    """
    return pair_rep * pair_mask.unsqueeze(-1).to(pair_rep.dtype)

class TriangleAttention(nn.Module):
    """
    Triangle Attention from AlphaFold.
    
    FIXED: Properly masks both query positions and key positions.
    Padding positions:
    - As queries: output is zeroed (via post-masking)
    - As keys: attention score is -inf (via attention mask)
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mode: str = 'starting',
        dropout: float = 0.1
    ):
        super().__init__()
        assert mode in ['starting', 'ending']
        self.mode = mode
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections (bias=False following AlphaFold)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Third-edge bias projection
        self.bias_proj = nn.Linear(dim, num_heads, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Gate for output
        self.gate_proj = nn.Linear(dim, dim)
        
        self.layer_norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        pair_rep: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply triangle attention with proper masking.
        
        Args:
            pair_rep: [batch, seq_len, seq_len, dim]
            pair_mask: [batch, seq_len, seq_len] where True = valid
        
        Returns:
            Updated pair representation (padding positions zeroed)
        """
        batch_size, seq_len, _, _ = pair_rep.shape
        
        # Pre-LayerNorm
        z = self.layer_norm(pair_rep)
        
        # For ending mode, transpose to reuse same attention logic
        if self.mode == 'ending':
            z = z.transpose(1, 2).contiguous()
            if pair_mask is not None:
                pair_mask_transposed = pair_mask.transpose(1, 2).contiguous()
            else:
                pair_mask_transposed = None
        else:
            pair_mask_transposed = pair_mask
        
        # Project to Q, K, V
        Q = self.q_proj(z)
        K = self.k_proj(z)
        V = self.v_proj(z)
        
        # Third-edge bias
        bias = self.bias_proj(z)  # [batch, seq, seq, num_heads]
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, seq_len, self.num_heads, self.head_dim)
        
        # [batch * seq_i, seq_j, num_heads, head_dim]
        Q = Q.reshape(batch_size * seq_len, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size * seq_len, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size * seq_len, seq_len, self.num_heads, self.head_dim)
        
        # [batch * seq_i, num_heads, seq_j, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Attention scores: [batch * seq_i, num_heads, seq_j, seq_k]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Add third-edge bias
        bias_expanded = bias.unsqueeze(1).expand(-1, seq_len, -1, -1, -1)
        bias_expanded = bias_expanded.reshape(batch_size * seq_len, seq_len, seq_len, self.num_heads)
        bias_expanded = bias_expanded.permute(0, 3, 1, 2)
        attn_scores = attn_scores + bias_expanded
        
        # Apply KEY mask (prevent attending TO padding positions)
        if pair_mask_transposed is not None:
            # For starting mode: pair_mask[i, k] tells if position (i, k) is valid
            # We want to mask keys where pair_mask[:, :, k] is False
            # mask shape: [batch, seq_i, seq_k] -> [batch * seq_i, 1, 1, seq_k]
            key_mask = pair_mask_transposed.reshape(batch_size * seq_len, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(~key_mask, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        out = torch.matmul(attn_weights, V)
        
        # Reshape back
        out = out.transpose(1, 2)
        out = out.contiguous().reshape(batch_size, seq_len, seq_len, self.dim)
        
        # Output projection
        out = self.out_proj(out)
        
        # Transpose back for ending mode
        if self.mode == 'ending':
            out = out.transpose(1, 2).contiguous()
        
        # Gated residual connection
        gate = torch.sigmoid(self.gate_proj(pair_rep))
        out = pair_rep + gate * self.dropout(out)
        
        # CRITICAL: Apply mask after residual to zero out padding positions
        # This prevents information leakage through the residual connection
        if pair_mask is not None:
            out = apply_pair_mask(out, pair_mask)
        
        return out

class TriangleMultiplication(nn.Module):
    """
    Triangle Multiplication from AlphaFold.
    
    FIXED: Applies mask after residual connection to ensure
    padding positions remain zeroed.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        mode: str = 'outgoing',
        dropout: float = 0.1
    ):
        super().__init__()
        assert mode in ['outgoing', 'incoming']
        self.mode = mode
        self.dim = dim
        hidden_dim = hidden_dim or dim
        self.hidden_dim = hidden_dim
        
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, dim)
        self.output_gate = nn.Linear(dim, dim)
        
        self.layer_norm = LayerNorm(dim)
        self.output_norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        pair_rep: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply triangle multiplication with proper masking.
        """
        z = self.layer_norm(pair_rep)
        
        # Gated projections
        left = self.left_proj(z) * torch.sigmoid(self.left_gate(z))
        right = self.right_proj(z) * torch.sigmoid(self.right_gate(z))
        
        # Zero out padding before einsum to prevent them from contributing
        if pair_mask is not None:
            mask_expanded = pair_mask.unsqueeze(-1).to(left.dtype)
            left = left * mask_expanded
            right = right * mask_expanded
        
        # Triangle aggregation
        if self.mode == 'outgoing':
            out = torch.einsum('bikd,bjkd->bijd', left, right)
        else:
            out = torch.einsum('bkid,bkjd->bijd', left, right)
        
        out = self.output_norm(out)
        out = self.out_proj(out)
        
        # Gated residual
        gate = torch.sigmoid(self.output_gate(pair_rep))
        out = pair_rep + gate * self.dropout(out)
        
        # CRITICAL: Apply mask after residual
        if pair_mask is not None:
            out = apply_pair_mask(out, pair_mask)
        
        return out

class PairTransition(nn.Module):
    """
    Feed-forward transition for pair representation.
    FIXED: Applies mask after residual.
    """
    def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = dim * expansion_factor
        
        self.layer_norm = LayerNorm(dim)
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        pair_rep: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        z = self.layer_norm(pair_rep)
        z = self.linear1(z)
        z = F.relu(z)
        z = self.dropout(z)
        z = self.linear2(z)
        z = self.dropout(z)
        out = pair_rep + z
        
        # CRITICAL: Apply mask after residual
        if pair_mask is not None:
            out = apply_pair_mask(out, pair_mask)
        
        return out
    

class InterfaceGeometryBlock(nn.Module):
    """
    Single block of Interface Geometry Reasoning.
    
    Each sub-module now properly handles masking internally.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.triangle_mult_outgoing = TriangleMultiplication(
            dim, mode='outgoing', dropout=dropout
        )
        self.triangle_mult_incoming = TriangleMultiplication(
            dim, mode='incoming', dropout=dropout
        )
        self.triangle_attn_starting = TriangleAttention(
            dim, num_heads, mode='starting', dropout=dropout
        )
        self.triangle_attn_ending = TriangleAttention(
            dim, num_heads, mode='ending', dropout=dropout
        )
        self.pair_transition = PairTransition(dim, dropout=dropout)
    
    def forward(
        self, 
        pair_rep: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Each module now handles masking internally and returns masked output
        pair_rep = self.triangle_mult_outgoing(pair_rep, pair_mask)
        pair_rep = self.triangle_mult_incoming(pair_rep, pair_mask)
        pair_rep = self.triangle_attn_starting(pair_rep, pair_mask)
        pair_rep = self.triangle_attn_ending(pair_rep, pair_mask)
        pair_rep = self.pair_transition(pair_rep, pair_mask)
        
        return pair_rep
    
class SegmentPairEmbedding(nn.Module):
    """
    Embedding to distinguish peptide-peptide, peptide-MHC, MHC-peptide, MHC-MHC pairs.
    
    FIXED: Padding positions get segment_id from input, but embeddings will be
    masked out anyway. Added explicit handling for clarity.
    """
    def __init__(self, dim: int, num_segments: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(num_segments * num_segments, dim)
        self.num_segments = num_segments
    
    def forward(
        self, 
        segment_ids: torch.Tensor,  # [batch, seq_len]
        pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create pair-wise segment embeddings.
        
        Args:
            segment_ids: [batch, seq_len] where 0=peptide, 1=MHC
                         (padding positions should have valid segment_ids but will be masked)
            pair_mask: [batch, seq_len, seq_len] optional mask
        
        Returns:
            [batch, seq_len, seq_len, dim] (masked if pair_mask provided)
        """
        
        seg_i = segment_ids.unsqueeze(2)
        seg_j = segment_ids.unsqueeze(1)
        pair_segment_ids = seg_i * self.num_segments + seg_j
        
        emb = self.embedding(pair_segment_ids)
        
        # Optionally mask the embeddings
        if pair_mask is not None:
            emb = apply_pair_mask(emb, pair_mask)
        
        return emb

class RelativePositionEmbedding(nn.Module):
    """
    Relative position embedding for pair representation.
    
    FIXED: 
    - Uses sequence mask to identify padding positions
    - Inter-chain embedding for positions across peptide-MHC boundary
    - Padding positions get zeroed out via mask
    """
    def __init__(self, dim: int, max_relative_pos: int = 32):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        
        # Intra-chain relative position
        self.rel_pos_embedding = nn.Embedding(2 * max_relative_pos + 1, dim)
        
        # Inter-chain embedding
        self.inter_chain_embedding = nn.Embedding(1, dim)
    
    def forward(
        self,
        seq_len: int,
        segment_ids: torch.Tensor,  # [batch, seq_len]
        pair_mask: Optional[torch.Tensor] = None,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        Create relative position embeddings.
        
        Note: Padding positions will have some embedding values, but these
        will be zeroed out when pair_mask is applied.
        """
        if device is None:
            device = segment_ids.device
        batch_size = segment_ids.shape[0]
        
        # Compute relative positions
        pos = torch.arange(seq_len, device=device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        rel_pos = rel_pos.clamp(-self.max_relative_pos, self.max_relative_pos)
        rel_pos = rel_pos + self.max_relative_pos
        
        rel_pos_emb = self.rel_pos_embedding(rel_pos)
        rel_pos_emb = rel_pos_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Inter-chain mask
        seg_i = segment_ids.unsqueeze(2)
        seg_j = segment_ids.unsqueeze(1)
        is_inter_chain = (seg_i != seg_j)
        
        inter_emb = self.inter_chain_embedding.weight[0]
        inter_emb = inter_emb.view(1, 1, 1, -1).expand(batch_size, seq_len, seq_len, -1)
        
        rel_pos_emb = torch.where(
            is_inter_chain.unsqueeze(-1),
            inter_emb,
            rel_pos_emb
        )
        
        # Apply mask if provided
        if pair_mask is not None:
            rel_pos_emb = apply_pair_mask(rel_pos_emb, pair_mask)
        
        return rel_pos_emb  

class InterfaceGeometryModule(nn.Module):
    """
    Complete Interface Geometry Reasoning Module using Concatenation Approach.
    
    PROPERLY HANDLES VARIABLE-LENGTH PEPTIDES:
    
    When peptide has padding (e.g., max_len=15, actual=9), the concatenated
    sequence looks like:
    
        [pep_0, pep_1, ..., pep_8, PAD, PAD, ..., PAD, MHC_0, MHC_1, ..., MHC_33]
        └────── real peptide ─────┘ └─── padding ───┘ └─────── MHC ───────────┘
    
    The pair mask ensures:
    - Padding positions don't attend to anything (keys masked as -inf)
    - Padding positions don't contribute to triangle operations (zeroed inputs)
    - Padding positions stay zeroed after every operation (post-masking)
    - Interface extraction correctly handles the mask
    
    Example:
        module = InterfaceGeometryModule(dim=128)
        
        # Variable length peptides (padded to max_len=15)
        peptide_emb = torch.randn(batch, 15, 128)
        peptide_mask = torch.tensor([[True]*9 + [False]*6, [True]*11 + [False]*4, ...])
        
        groove_emb = torch.randn(batch, 34, 128)
        groove_mask = torch.ones(batch, 34, dtype=torch.bool)
        
        interface_rep, pair_rep = module(peptide_emb, groove_emb, peptide_mask, groove_mask)
    """
    def __init__(
        self,
        dim: int,
        num_blocks: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_relative_pos: int = 32
    ):
        super().__init__()
        self.dim = dim
        self.num_blocks = num_blocks
        
        # Initial pair representation projection
        self.pair_proj = nn.Linear(dim * 2, dim)
        
        # Segment embedding
        self.segment_pair_embedding = SegmentPairEmbedding(dim, num_segments=2)
        
        # Relative position embedding
        self.rel_pos_embedding = RelativePositionEmbedding(dim, max_relative_pos)
        
        # Geometry blocks
        self.blocks = nn.ModuleList([
            InterfaceGeometryBlock(dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        # Final normalization
        self.final_norm = LayerNorm(dim)
    
    def create_pair_representation(
        self,
        combined_emb: torch.Tensor,
        pair_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Create initial pair representation."""
        batch_size, total_len, dim = combined_emb.shape
        
        emb_i = combined_emb.unsqueeze(2).expand(-1, -1, total_len, -1)
        emb_j = combined_emb.unsqueeze(1).expand(-1, total_len, -1, -1)
        pair_concat = torch.cat([emb_i, emb_j], dim=-1)
        pair_rep = self.pair_proj(pair_concat)
        
        # Apply initial mask
        if pair_mask is not None:
            pair_rep = apply_pair_mask(pair_rep, pair_mask)
        
        return pair_rep
    
    def create_segment_ids(
        self,
        batch_size: int,
        pep_len: int,
        groove_len: int,
        peptide_mask: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create segment IDs: 0 for peptide positions, 1 for MHC positions.
        
        Note: Padding positions within peptide region get segment_id=0,
        but this doesn't matter as they'll be masked anyway.
        """
        segment_ids = torch.cat([
            torch.zeros(batch_size, pep_len, dtype=torch.long, device=device),
            torch.ones(batch_size, groove_len, dtype=torch.long, device=device)
        ], dim=1)
        return segment_ids
    
    def forward(
        self,
        peptide_emb: torch.Tensor,      # [batch, pep_len, dim]
        groove_emb: torch.Tensor,       # [batch, groove_len, dim]
        peptide_mask: Optional[torch.Tensor] = None,  # [batch, pep_len]
        groove_mask: Optional[torch.Tensor] = None,   # [batch, groove_len]
        return_pair_rep: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply interface geometry reasoning.
        
        Args:
            peptide_emb: [batch, pep_len, dim] - may contain padding!
            groove_emb: [batch, groove_len, dim]
            peptide_mask: [batch, pep_len] boolean, True = real position
            groove_mask: [batch, groove_len] boolean, True = real position
            return_pair_rep: Whether to return interface pair representation
        
        Returns:
            interface_rep: [batch, dim] - pooled interface representation
            pair_rep: [batch, pep_len, groove_len, dim] - interface pairs (masked)
        """
        batch_size, pep_len, dim = peptide_emb.shape
        _, groove_len, _ = groove_emb.shape
        device = peptide_emb.device
        total_len = pep_len + groove_len
        
        # =====================================================================
        # Step 1: Create masks (handle None cases)
        # =====================================================================
        if peptide_mask is None:
            peptide_mask = torch.ones(batch_size, pep_len, dtype=torch.bool, device=device)
        if groove_mask is None:
            groove_mask = torch.ones(batch_size, groove_len, dtype=torch.bool, device=device)
        
        # Combined sequence mask: [batch, total_len]
        # This will have False (padding) in the MIDDLE if peptide has padding
        combined_mask = torch.cat([peptide_mask, groove_mask], dim=1)
        
        # Pair mask: [batch, total_len, total_len]
        # pair_mask[b, i, j] = True only if BOTH positions i and j are valid
        pair_mask = combined_mask.unsqueeze(2) & combined_mask.unsqueeze(1)
        
        # =====================================================================
        # Step 2: Zero out padding in input embeddings BEFORE concatenation
        # =====================================================================
        peptide_emb_masked = peptide_emb * peptide_mask.unsqueeze(-1).to(peptide_emb.dtype)
        groove_emb_masked = groove_emb * groove_mask.unsqueeze(-1).to(groove_emb.dtype)
        
        # =====================================================================
        # Step 3: Concatenate sequences
        # =====================================================================
        combined_emb = torch.cat([peptide_emb_masked, groove_emb_masked], dim=1)
        
        # =====================================================================
        # Step 4: Create segment IDs
        # =====================================================================
        segment_ids = self.create_segment_ids(
            batch_size, pep_len, groove_len, peptide_mask, device
        )
        
        # =====================================================================
        # Step 5: Create initial pair representation (with masking)
        # =====================================================================
        pair_rep = self.create_pair_representation(combined_emb, pair_mask)
        
        # =====================================================================
        # Step 6: Add embeddings (segment + relative position)
        # =====================================================================
        # These are also masked to zero out padding positions
        segment_emb = self.segment_pair_embedding(segment_ids, pair_mask)
        rel_pos_emb = self.rel_pos_embedding(total_len, segment_ids, pair_mask, device)
        
        pair_rep = pair_rep + segment_emb + rel_pos_emb
        
        # Re-apply mask after addition (ensures zeros stay zero)
        pair_rep = apply_pair_mask(pair_rep, pair_mask)
        
        # =====================================================================
        # Step 7: Apply geometry blocks
        # =====================================================================
        for block in self.blocks:
            pair_rep = block(pair_rep, pair_mask)
        
        # Final normalization
        pair_rep = self.final_norm(pair_rep)
        pair_rep = apply_pair_mask(pair_rep, pair_mask)  # Final mask application
        
        # =====================================================================
        # Step 8: Extract interface region
        # =====================================================================
        # Interface is the peptide-MHC block (top-right of the pair matrix)
        # Shape: [batch, pep_len, groove_len, dim]
        interface_pair_rep = pair_rep[:, :pep_len, pep_len:, :]
        
        # Interface mask: which (peptide, MHC) pairs are valid
        interface_mask = peptide_mask.unsqueeze(2) & groove_mask.unsqueeze(1)
        # [batch, pep_len, groove_len]
        
        # =====================================================================
        # Step 9: Pool interface representation
        # =====================================================================
        # Masked mean pooling - only over valid positions
        interface_mask_expanded = interface_mask.unsqueeze(-1).to(interface_pair_rep.dtype)

        interface_sum = (interface_pair_rep * interface_mask_expanded).sum(dim=[1, 2])
        # Convert count to match interface_sum dtype to preserve bfloat16/float16
        interface_count = interface_mask.sum(dim=[1, 2], keepdim=False).to(interface_sum.dtype).clamp(min=1)
        interface_rep = interface_sum / interface_count.unsqueeze(-1)
        # [batch, dim]
        
        if return_pair_rep:
            # Return masked interface pair representation
            # Padding rows/columns are already zeroed
            return interface_rep, interface_pair_rep
        
        return interface_rep, None
    
    def get_valid_interface_pairs(
        self,
        pair_rep: torch.Tensor,           # [batch, pep_len, groove_len, dim]
        peptide_mask: torch.Tensor,       # [batch, pep_len]
        groove_mask: torch.Tensor,        # [batch, groove_len]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract only valid (non-padding) interface pairs.
        
        Useful when you need to process only real positions without padding.
        
        Returns:
            valid_pairs: [total_valid_pairs, dim]
            batch_indices: [total_valid_pairs] - which batch each pair belongs to
        """
        interface_mask = peptide_mask.unsqueeze(2) & groove_mask.unsqueeze(1)
        # [batch, pep_len, groove_len]
        
        valid_pairs = pair_rep[interface_mask]  # [num_valid, dim]
        
        # Get batch indices for each valid pair
        batch_indices = torch.arange(pair_rep.shape[0], device=pair_rep.device)
        batch_indices = batch_indices.view(-1, 1, 1).expand_as(interface_mask)
        batch_indices = batch_indices[interface_mask]  # [num_valid]
        
        return valid_pairs, batch_indices
      
# END============================================================================
# Interface Geometry Reasoning
# ===============================================================================
    
    
# START==========================================================================
# Position-Aware Peptide Encoding
# ===============================================================================  
class PositionAwarePeptideEncoder(nn.Module):
    """
    Position-aware encoding for peptides.
    
    Key insight: Anchor positions (P2, P9 for 9-mers) are critical
    for binding. This module explicitly models position importance.
    """
    def __init__(
        self,
        dim: int,
        max_peptide_length: int = 15,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_peptide_length, dim)
        
        # Anchor position learner
        self.anchor_attention = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.anchor_norm = LayerNorm(dim)
        
        # Position importance predictor
        self.importance_predictor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(dim, dim)
    
    def forward(
        self,
        peptide_emb: torch.Tensor,  # [batch, pep_len, dim]
        peptide_mask: Optional[torch.Tensor] = None,  # [batch, pep_len] True=valid
        return_importance: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply position-aware encoding.
        
        Args:
            peptide_emb: Peptide embeddings [batch, pep_len, dim]
            peptide_mask: Boolean mask where True = valid, False = padding
            return_importance: Whether to return position importance scores
        
        Returns:
            enhanced_emb: Position-enhanced peptide embeddings
            position_importance: Optional per-position importance scores
        """
        batch_size, pep_len, _ = peptide_emb.shape

        # Convert to key_padding_mask format (True = padding) for nn.MultiheadAttention
        key_padding_mask = ~peptide_mask if peptide_mask is not None else None

        # Add position embeddings
        # Clamp positions to max_peptide_length to handle sequences with BOS/EOS tokens
        positions = torch.arange(pep_len, device=peptide_emb.device)
        positions = torch.clamp(positions, max=self.position_embeddings.num_embeddings - 1)
        pos_emb = self.position_embeddings(positions)
        peptide_emb = peptide_emb + pos_emb.unsqueeze(0)
        
        # Self-attention to learn anchor positions
        attended, _ = self.anchor_attention(
            peptide_emb, peptide_emb, peptide_emb,
            key_padding_mask=key_padding_mask
        )
        peptide_emb = self.anchor_norm(peptide_emb + attended)
        
        # Predict position importance
        importance = self.importance_predictor(peptide_emb)  # [batch, pep_len, 1]
        
        # Weight embeddings by importance
        enhanced_emb = peptide_emb * (1 + importance)
        enhanced_emb = self.output_proj(enhanced_emb)
        
        # Zero out padded positions
        if peptide_mask is not None:
            enhanced_emb = enhanced_emb * peptide_mask.unsqueeze(-1).to(enhanced_emb.dtype)
            importance = importance * peptide_mask.unsqueeze(-1).to(importance.dtype)
        
        if return_importance:
            return enhanced_emb, importance.squeeze(-1)
        return enhanced_emb, None
# END============================================================================
# Position-Aware Peptide Encoding
# ===============================================================================





# START==========================================================================
# Prediction Heads
# ===============================================================================  
class BindingAffinityHead(nn.Module):
    """
    Prediction head for binding affinity (regression).
    Outputs IC50/KD prediction.
    """
    def __init__(self, dim: int, dropout: float = 0.1, use_flanks: bool = False):
        super().__init__()
        self.use_flanks = use_flanks
        input_dim = dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 4, 1)
        )
        
        # self.flank_proj = nn.Linear(dim * 2, dim) if use_flanks else None

    def forward(self, x: torch.Tensor, flank_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Interface representation [batch, dim]
            flank_features: Optional [batch, 2, dim] with [nflank_pool, cflank_pool]
        """
        # if self.use_flanks and flank_features is not None:
        #     x = x + self.flank_proj(flank_features.flatten(-2, -1))
        return self.head(x)


class PresentationHead(nn.Module):
    """
    Prediction head for binding classification.
    Outputs binder/non-binder probability.
    """
    def __init__(self, dim: int, dropout: float = 0.1, use_flanks: bool = False):
        super().__init__()
        self.use_flanks = use_flanks
        input_dim = dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        
        # self.flank_proj = nn.Linear(dim * 2, dim) if use_flanks else None

    def forward(self, x: torch.Tensor, flank_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Interface representation [batch, dim]
            flank_features: Optional [batch, 2, dim] with [nflank_pool, cflank_pool]
        """
        # if self.use_flanks and flank_features is not None:
        #     x = x + self.flank_proj(flank_features.flatten(-2, -1))
        return self.head(x)


class ContactPredictionHead(nn.Module):
    """
    Auxiliary head for contact prediction.
    Operates on pair representation to predict contacts.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair_rep: [batch, pep_len, groove_len, dim]
        Returns:
            contact_prob: [batch, pep_len, groove_len]
        """
        return self.head(pair_rep).squeeze(-1)
# END============================================================================
# Prediction Heads
# ===============================================================================





# START==========================================================================
# Sequence Encoders
# ===============================================================================  
class EmbeddingEncoder(nn.Module):
    """
    Simple embedding-based encoder for sequences.
    Lightweight alternative to ESM-2 for fast prototyping.
    """
    def __init__(self, dim: int, vocab_size: int = 21):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)  # 20 AA + padding
        self.proj = nn.Linear(dim, dim)
        self.norm = LayerNorm(dim)
    
    def forward(
        self, 
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len] token indices
            mask: [batch, seq_len] boolean mask where True = valid, False = padding
        Returns:
            embeddings: [batch, seq_len, dim]
        """
        x = self.embedding(tokens)
        x = self.proj(x)
        x = self.norm(x)
        
        # Zero out padded positions
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        
        return x


# class ESM2Encoder(nn.Module):
#     """
#     ESM-2 based encoder with configurable layer unfreezing.
    
#     Supports:
#     - Loading pretrained ESM-2 8M model
#     - Freezing/unfreezing specific layers for fine-tuning
#     - Projection to match model dimension
    
#     ESM-2 8M has 6 transformer layers (0-5).
#     Common strategies:
#     - Freeze all: unfreeze_layers=0 (feature extraction)
#     - Unfreeze last N: unfreeze_layers=N (fine-tuning)
#     - Unfreeze all: unfreeze_layers=-1 (full fine-tuning)
    
#     Note on masking:
#     ESM-2 uses its own internal attention mechanism that produces different
#     representations based on sequence length/context. The mask parameter
#     here is used to zero out padded positions in the OUTPUT, which is the
#     correct behavior for downstream modules. However, for proper batching
#     with variable-length sequences, ensure tokens are properly padded using
#     ESM's padding token (alphabet.padding_idx).
#     """
#     def __init__(
#         self,
#         output_dim: int,
#         model_name: str = "facebook/esm2_t33_650M_UR50D",
#         unfreeze_layers: int = 3,  # Number of top layers to unfreeze (0=freeze all, -1=unfreeze all)
#         unfreeze_embeddings: bool = False
#     ):
#         super().__init__()
#         self.output_dim = output_dim
#         self.unfreeze_layers = unfreeze_layers
#         self.unfreeze_embeddings = unfreeze_embeddings
        
#         # Load ESM-2 model
#         try:
#             import esm
#             self.esm_model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
#             self.esm_dim = self.esm_model.embed_dim  # 320 for 8M model
#             self.num_layers = self.esm_model.num_layers  # 6 for 8M model
#             self.padding_idx = self.alphabet.padding_idx  # ESM's padding token
#         except ImportError:
#             raise ImportError(
#                 "ESM package not found. Install with: pip install fair-esm"
#             )
#         except Exception as e:
#             raise RuntimeError(f"Failed to load ESM model '{model_name}': {e}")
        
#         # Projection layer if dimensions don't match
#         if self.esm_dim != output_dim:
#             self.proj = nn.Linear(self.esm_dim, output_dim)
#         else:
#             self.proj = nn.Identity()
        
#         self.norm = LayerNorm(output_dim)
        
#         # Apply freezing strategy
#         self._apply_freeze_strategy()
    
#     def _apply_freeze_strategy(self):
#         """Apply layer freezing based on configuration."""
#         # First, freeze everything
#         for param in self.esm_model.parameters():
#             param.requires_grad = False
        
#         if self.unfreeze_layers == -1:
#             # Unfreeze all layers
#             for param in self.esm_model.parameters():
#                 param.requires_grad = True
#         elif self.unfreeze_layers > 0:
#             # Unfreeze top N transformer layers
#             # ESM-2 layers are indexed 0 to num_layers-1
#             layers_to_unfreeze = list(range(
#                 self.num_layers - self.unfreeze_layers, 
#                 self.num_layers
#             ))
            
#             for layer_idx in layers_to_unfreeze:
#                 if hasattr(self.esm_model, 'layers'):
#                     # ESM-2 structure
#                     for param in self.esm_model.layers[layer_idx].parameters():
#                         param.requires_grad = True
            
#             # Also unfreeze the final layer norm if it exists
#             if hasattr(self.esm_model, 'emb_layer_norm_after'):
#                 for param in self.esm_model.emb_layer_norm_after.parameters():
#                     param.requires_grad = True
        
#         # Optionally unfreeze embeddings
#         if self.unfreeze_embeddings:
#             if hasattr(self.esm_model, 'embed_tokens'):
#                 for param in self.esm_model.embed_tokens.parameters():
#                     param.requires_grad = True
        
#         # Projection layer is always trainable
#         if hasattr(self.proj, 'parameters'):
#             for param in self.proj.parameters():
#                 param.requires_grad = True
    
#     def get_trainable_params_info(self) -> Dict[str, int]:
#         """Get information about trainable parameters."""
#         total = sum(p.numel() for p in self.parameters())
#         trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         frozen = total - trainable
        
#         return {
#             'total': total,
#             'trainable': trainable,
#             'frozen': frozen,
#             'trainable_pct': 100 * trainable / total if total > 0 else 0
#         }
    
#     def forward(
#         self, 
#         tokens: torch.Tensor,
#         mask: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         """
#         Args:
#             tokens: [batch, seq_len] ESM token indices (includes BOS/EOS/PAD)
#             mask: [batch, seq_len] boolean mask where True = valid, False = padding
#                   Note: ESM's mask marks BOS and EOS as valid (non-padding)
#         Returns:
#             embeddings: [batch, seq_len-1, output_dim] - BOS removed, EOS zeroed
        
#         Note:
#             ESM tokenization: [BOS, seq..., EOS, PAD...]
#             - BOS is always at position 0
#             - EOS is at variable position (right after the sequence)
#             - PAD fills the rest
            
#             This method:
#             1. Removes BOS (position 0) from output
#             2. Marks EOS as invalid so it gets zeroed
#             3. Applies mask to zero out EOS and PAD positions
#         """
#         # ESM forward pass
#         results = self.esm_model(tokens, repr_layers=[self.num_layers])
        
#         # Get representations from last layer
#         # Shape: [batch, seq_len, esm_dim]
#         representations = results["representations"][self.num_layers]
        
#         # Remove BOS (position 0) from representations
#         # Shape: [batch, seq_len-1, esm_dim]
#         representations = representations[:, 1:, :]
        
#         # Project to output dimension
#         x = self.proj(representations)
#         x = self.norm(x)
        
#         # Handle masking: remove BOS from mask, mark EOS as invalid
#         if mask is not None:
#             # Find EOS positions and mark them as invalid
#             eos_idx = self.alphabet.eos_idx
#             is_eos = (tokens == eos_idx)
#             mask_no_eos = mask.clone()
#             mask_no_eos[is_eos] = False
            
#             # Remove BOS position (0) from mask
#             mask_stripped = mask_no_eos[:, 1:]
            
#             # Apply mask to zero out EOS and PAD positions
#             x = x * mask_stripped.unsqueeze(-1).float()
        
#         return x


class ESM2Encoder(nn.Module):
    """
    ESM-2 based encoder using Hugging Face Transformers.
    
    Supports:
    - Loading pretrained ESM-2 models from Hugging Face
    - Freezing/unfreezing specific layers for fine-tuning
    - Projection to match model dimension
    - Quantization (4-bit/8-bit) using bitsandbytes
    - LoRA (Low-Rank Adaptation) using PEFT
    
    Model options:
    - "facebook/esm2_t6_8M_UR50D" (6 layers, 8M params)
    - "facebook/esm2_t12_35M_UR50D" (12 layers, 35M params)
    - "facebook/esm2_t30_150M_UR50D" (30 layers, 150M params)
    - "facebook/esm2_t33_650M_UR50D" (33 layers, 650M params)
    - "facebook/esm2_t36_3B_UR50D" (36 layers, 3B params)
    
    Common strategies:
    - Freeze all: unfreeze_layers=0 (feature extraction)
    - Unfreeze last N: unfreeze_layers=N (fine-tuning)
    - Unfreeze all: unfreeze_layers=-1 (full fine-tuning)
    
    Quantization options:
    - quantization=None: No quantization (default)
    - quantization='4bit': 4-bit quantization (NF4)
    - quantization='8bit': 8-bit quantization
    
    LoRA options:
    - use_lora=False: No LoRA (default)
    - use_lora=True: Apply LoRA to attention layers
    - lora_r: LoRA rank (default: 8)
    - lora_alpha: LoRA alpha scaling (default: 16)
    - lora_dropout: LoRA dropout (default: 0.1)
    - lora_target_modules: Modules to apply LoRA (default: query, key, value)
    """
    def __init__(
        self,
        output_dim: int,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        unfreeze_layers: int = 2,
        unfreeze_embeddings: bool = False,
        # Quantization options
        quantization: Optional[str] = None,  # None, '4bit', '8bit'
        # LoRA options
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.unfreeze_layers = unfreeze_layers
        self.unfreeze_embeddings = unfreeze_embeddings
        self.quantization = quantization
        self.use_lora = use_lora
        
        # Load ESM-2 model from Hugging Face
        try:
            from transformers import EsmModel, EsmTokenizer
            
            # Setup quantization config if requested
            quantization_config = None
            if quantization is not None:
                # Check CUDA availability for quantization
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"Quantization '{quantization}' requires CUDA but no GPU is available. "
                        "Set quantization=None to use CPU."
                    )
                
                try:
                    from transformers import BitsAndBytesConfig
                    import bitsandbytes as bnb
                    
                    if quantization == '4bit':
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                        )
                    elif quantization == '8bit':
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_skip_modules=["lm_head"],  # Skip problematic modules
                        )
                    else:
                        raise ValueError(f"Unknown quantization type: {quantization}. Use '4bit' or '8bit'.")
                except ImportError as e:
                    raise ImportError(
                        f"bitsandbytes package not found or error importing: {e}. "
                        "Install with: pip install bitsandbytes>=0.41.0"
                    )
            
            # Load model with or without quantization
            if quantization_config is not None:
                self.esm_model = EsmModel.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.esm_model = EsmModel.from_pretrained(model_name)
            
            self.tokenizer = EsmTokenizer.from_pretrained(model_name)
            
            self.esm_dim = self.esm_model.config.hidden_size
            self.num_layers = self.esm_model.config.num_hidden_layers
            self.padding_idx = self.tokenizer.pad_token_id
            
            # Store special token IDs for mask handling
            self.bos_idx = self.tokenizer.cls_token_id  # ESM uses <cls> as BOS
            self.eos_idx = self.tokenizer.eos_token_id
            
        except ImportError:
            raise ImportError(
                "Transformers package not found. Install with: pip install transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ESM model '{model_name}': {e}")
        
        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
            )
        
        # Projection layer if dimensions don't match
        if self.esm_dim != output_dim:
            self.proj = nn.Linear(self.esm_dim, output_dim)
        else:
            self.proj = nn.Identity()
        
        self.norm = LayerNorm(output_dim)
        
        # Apply freezing strategy (skip if using LoRA - LoRA handles this)
        if not use_lora:
            self._apply_freeze_strategy()
        
        # Create a fake alphabet object for compatibility with existing code
        self._create_alphabet_compat()
    
    def _apply_lora(
        self,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[list] = None,
    ):
        """Apply LoRA to the ESM model using PEFT."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # Default target modules for ESM-2 attention layers
            if lora_target_modules is None:
                lora_target_modules = [
                    "query",
                    "key", 
                    "value"
                ]
            
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            
            # Wrap model with LoRA
            self.esm_model = get_peft_model(self.esm_model, lora_config)
            
            # Print trainable parameters info
            self.esm_model.print_trainable_parameters()
            
        except ImportError:
            raise ImportError(
                "PEFT package not found for LoRA. Install with: pip install peft"
            )
    
    def _create_alphabet_compat(self):
        """Create a compatibility layer for code expecting fair-esm alphabet."""
        class AlphabetCompat:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.padding_idx = tokenizer.pad_token_id
                self.cls_idx = tokenizer.cls_token_id
                self.eos_idx = tokenizer.eos_token_id
                self.mask_idx = tokenizer.mask_token_id
                
            def get_tok(self, idx):
                return self.tokenizer.convert_ids_to_tokens(idx)
            
            def get_batch_converter(self):
                """Return a batch converter function compatible with fair-esm."""
                def batch_converter(data):
                    # data is list of (label, sequence) tuples
                    labels = [d[0] for d in data]
                    sequences = [d[1] for d in data]
                    
                    # Tokenize with transformers
                    encoded = self.tokenizer(
                        sequences,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        add_special_tokens=True
                    )
                    
                    return labels, sequences, encoded['input_ids']
                
                return batch_converter
        
        self.alphabet = AlphabetCompat(self.tokenizer)
    
    def _apply_freeze_strategy(self):
        """Apply layer freezing based on configuration."""
        # First, freeze everything
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        if self.unfreeze_layers == -1:
            # Unfreeze all layers
            for param in self.esm_model.parameters():
                param.requires_grad = True
        elif self.unfreeze_layers > 0:
            # Unfreeze top N transformer layers
            layers_to_unfreeze = list(range(
                self.num_layers - self.unfreeze_layers, 
                self.num_layers
            ))
            
            # Hugging Face ESM-2 structure: esm_model.encoder.layer[i]
            for layer_idx in layers_to_unfreeze:
                if hasattr(self.esm_model, 'encoder') and hasattr(self.esm_model.encoder, 'layer'):
                    for param in self.esm_model.encoder.layer[layer_idx].parameters():
                        param.requires_grad = True
            
            # Also unfreeze the final layer norm
            if hasattr(self.esm_model, 'encoder') and hasattr(self.esm_model.encoder, 'emb_layer_norm_after'):
                for param in self.esm_model.encoder.emb_layer_norm_after.parameters():
                    param.requires_grad = True
        
        # Optionally unfreeze embeddings
        if self.unfreeze_embeddings:
            if hasattr(self.esm_model, 'embeddings'):
                for param in self.esm_model.embeddings.parameters():
                    param.requires_grad = True
        
        # Projection layer is always trainable
        if hasattr(self.proj, 'parameters'):
            for param in self.proj.parameters():
                param.requires_grad = True
    
    def get_trainable_params_info(self) -> Dict[str, int]:
        """Get information about trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        info = {
            'total': total,
            'trainable': trainable,
            'frozen': frozen,
            'trainable_pct': 100 * trainable / total if total > 0 else 0,
            'quantization': self.quantization,
            'use_lora': self.use_lora,
        }
        
        # Add LoRA-specific info if using LoRA
        if self.use_lora:
            try:
                lora_params = sum(
                    p.numel() for n, p in self.esm_model.named_parameters() 
                    if 'lora' in n.lower() and p.requires_grad
                )
                info['lora_params'] = lora_params
            except:
                pass
        
        return info
    
    def merge_and_unload_lora(self):
        """Merge LoRA weights into base model and unload LoRA (for inference)."""
        if self.use_lora:
            try:
                self.esm_model = self.esm_model.merge_and_unload()
                self.use_lora = False
                print("LoRA weights merged and unloaded successfully.")
            except Exception as e:
                print(f"Failed to merge LoRA: {e}")
    
    def save_lora_weights(self, path: str):
        """Save only the LoRA weights to a file."""
        if self.use_lora:
            self.esm_model.save_pretrained(path)
            print(f"LoRA weights saved to {path}")
        else:
            print("LoRA is not enabled, nothing to save.")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights from a file."""
        if self.use_lora:
            try:
                from peft import PeftModel
                self.esm_model = PeftModel.from_pretrained(
                    self.esm_model.base_model.model, 
                    path
                )
                print(f"LoRA weights loaded from {path}")
            except Exception as e:
                print(f"Failed to load LoRA weights: {e}")
        else:
            print("LoRA is not enabled. Initialize with use_lora=True first.")
    
    def forward(
        self, 
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len] ESM token indices (includes BOS/EOS/PAD)
            mask: [batch, seq_len] boolean mask where True = valid, False = padding
        Returns:
            embeddings: [batch, seq_len-1, output_dim] - BOS removed, EOS zeroed
        """
        # Create attention mask for transformers (1 = attend, 0 = ignore)
        if mask is not None:
            attention_mask = mask.long()
        else:
            attention_mask = (tokens != self.padding_idx).long()
        
        # ESM forward pass using transformers
        outputs = self.esm_model(
            input_ids=tokens,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get last hidden state
        # Shape: [batch, seq_len, esm_dim]
        representations = outputs.last_hidden_state
        
        # Remove BOS (position 0) from representations
        # Shape: [batch, seq_len-1, esm_dim]
        representations = representations[:, 1:, :]
        
        # Project to output dimension
        x = self.proj(representations)
        x = self.norm(x)
        
        # Handle masking: remove BOS from mask, mark EOS as invalid
        if mask is not None:
            # Find EOS positions and mark them as invalid
            is_eos = (tokens == self.eos_idx)
            mask_no_eos = mask.clone()
            mask_no_eos[is_eos] = False
            
            # Remove BOS position (0) from mask
            mask_stripped = mask_no_eos[:, 1:]
            
            # Apply mask to zero out EOS and PAD positions
            # Use x.dtype to preserve the dtype (bfloat16/float16/float32)
            x = x * mask_stripped.unsqueeze(-1).to(x.dtype)
        
        return x
            
# END============================================================================
# Sequence Encoders
# ===============================================================================






# START==========================================================================
# Main Net
# ===============================================================================
class DinoMHC(nn.Module):
    """
    Complete DinoMHC Architecture.
    
    Combines all modules:
    1. Protein encoders (Embedding or ESM-2)
    2. Hierarchical Groove Extractor
    3. Position-Aware Peptide Encoder
    4. Groove-Peptide Fusion
    5. Interface Geometry Reasoning
    6. Multi-task prediction heads
    
    Encoder Options:
    - 'embedding': Simple nn.Embedding (fast, for prototyping)
    - 'esm2': ESM-2 8M pretrained model (better performance)
    - 'esm2_shared': Single ESM-2 for both peptide and MHC (parameter efficient)
    
    This architecture explicitly models the biological binding process:
    - Extracts the relevant groove region from MHC
    - Enhances peptide with position-aware encoding
    - Fuses peptide and groove through bidirectional cross-attention
    - Reasons about geometric constraints at the interface
    - Predicts binding affinity/classification
    """
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Default configuration
        default_config = {
            'dim': 320,
            'num_groove_tokens': 60,
            'num_fusion_layers': 4,
            'num_geometry_blocks': 3,
            'num_heads': 8,
            'dropout': 0.1,
            'max_peptide_length': 15,
            'task_head': 'presentation',  # 'affinity', 'presentation', 'contact'
            # Encoder configuration
            'encoder_type': 'esm2',  # 'embedding', 'esm2', 'esm2_shared'
            'esm_model_name': 'facebook/esm2_t33_650M_UR50D',
            'esm_unfreeze_layers': 2,  # Number of top layers to unfreeze
            'esm_unfreeze_embeddings': False,
            # Flank configuration
            'use_flanks': True,  # Whether to use flanking regions
            'flank_pooling': 'mean',  # 'mean' or 'max' pooling for flanks
            'esm_quantization': None,
            'esm_use_lora': True,
            'esm_lora_r': 8,
            'esm_lora_alpha': 16,
            'esm_lora_dropout': 0.1,
            'esm_lora_target_modules': ['query', 'key', 'value'],
        }
        
        if config is not None:
            default_config.update(config)
        config = default_config
        
        self.config = config
        dim = config['dim']
        
        # === Build Encoders ===
        self._build_encoders(config)
        
        # === Core Modules ===
        self.groove_extractor = HierarchicalGrooveExtractor(
            dim=dim,
            num_groove_tokens=config['num_groove_tokens'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        )
        
        self.position_encoder = PositionAwarePeptideEncoder(
            dim=dim,
            max_peptide_length=config['max_peptide_length'],
            dropout=config['dropout']
        )
        
        self.groove_peptide_fusion = GroovePeptideFusion(
            dim=dim,
            num_layers=config['num_fusion_layers'],
            num_heads=config['num_heads'],
            dropout=config['dropout'],
            max_peptide_length=config['max_peptide_length'],
            num_groove_tokens=config['num_groove_tokens']
        )
        
        self.interface_geometry = InterfaceGeometryModule(
            dim=dim,
            num_blocks=config['num_geometry_blocks'],
            num_heads=config['num_heads'] // 2,  # Fewer heads for efficiency
            dropout=config['dropout'],
            max_relative_pos=config['max_peptide_length'] + config['num_groove_tokens']
        )
        
        # === Prediction Heads ===
        use_flanks = config.get('use_flanks', False)
        if config['task_head'] == 'affinity':
            self.task_head = BindingAffinityHead(dim, config['dropout'], use_flanks=use_flanks)
        elif config['task_head'] == 'presentation':
            self.task_head = PresentationHead(dim, config['dropout'], use_flanks=use_flanks)
        elif config['task_head'] == 'contact':
            self.task_head = ContactPredictionHead(dim, config['dropout'])
        else:
            raise ValueError(
                f"Unknown task_head: {config['task_head']}, "
                "must be one of 'affinity', 'presentation', 'contact'"
            )
    
    def _build_encoders(self, config: Dict):
        """Build sequence encoders based on configuration."""
        dim = config['dim']
        encoder_type = config['encoder_type']
        
        # Store ESM special token indices (will be None for embedding encoder)
        self._esm_eos_idx = None
        
        # Extract ESM-specific config options
        esm_kwargs = {
            'output_dim': dim,
            'model_name': config.get('esm_model_name', 'facebook/esm2_t6_8M_UR50D'),
            'unfreeze_layers': config.get('esm_unfreeze_layers', 2),
            'unfreeze_embeddings': config.get('esm_unfreeze_embeddings', False),
            # Quantization and LoRA options
            'quantization': config.get('esm_quantization', None),
            'use_lora': config.get('esm_use_lora', False),
            'lora_r': config.get('esm_lora_r', 8),
            'lora_alpha': config.get('esm_lora_alpha', 16),
            'lora_dropout': config.get('esm_lora_dropout', 0.1),
            'lora_target_modules': config.get('esm_lora_target_modules', None),
        }
        
        if encoder_type == 'embedding':
            # Simple embedding encoders (separate for peptide and MHC)
            self.peptide_encoder = EmbeddingEncoder(dim)
            self.mhc_encoder = EmbeddingEncoder(dim)
            self.shared_encoder = False
            
        elif encoder_type == 'esm2':
            # Separate ESM-2 encoders for peptide and MHC
            # Different encoders allow specialized fine-tuning
            self.peptide_encoder = ESM2Encoder(**esm_kwargs)
            self.mhc_encoder = ESM2Encoder(**esm_kwargs)
            
            # Freeze mhc encoder
            for param in self.mhc_encoder.parameters():
                param.requires_grad = False
            self.mhc_encoder.eval()
            
            self.shared_encoder = False
            # Store EOS index for mask adjustment
            self._esm_eos_idx = self.peptide_encoder.alphabet.eos_idx
            
        elif encoder_type == 'esm2_shared':
            # Single shared ESM-2 encoder for both peptide and MHC
            # More parameter efficient, but may not capture sequence-specific features
            self.shared_esm_encoder = ESM2Encoder(**esm_kwargs)
            # Aliases for compatibility
            self.peptide_encoder = self.shared_esm_encoder
            self.mhc_encoder = self.shared_esm_encoder
            self.shared_encoder = True
            # Store EOS index for mask adjustment
            self._esm_eos_idx = self.shared_esm_encoder.alphabet.eos_idx
            
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type}, "
                "must be one of 'embedding', 'esm2', 'esm2_shared'"
            )
    
    def get_encoder_info(self) -> Dict:
        """Get information about encoder configuration and parameters."""
        info = {
            'encoder_type': self.config['encoder_type'],
            'shared_encoder': self.shared_encoder,
        }
        
        if self.config['encoder_type'] in ['esm2', 'esm2_shared']:
            if self.shared_encoder:
                info['encoder_params'] = self.shared_esm_encoder.get_trainable_params_info()
            else:
                info['peptide_encoder_params'] = self.peptide_encoder.get_trainable_params_info()
                info['mhc_encoder_params'] = self.mhc_encoder.get_trainable_params_info()
        
        return info
    
    def freeze_encoders(self):
        """Freeze all encoder parameters."""
        for param in self.peptide_encoder.parameters():
            param.requires_grad = False
        if not self.shared_encoder:
            for param in self.mhc_encoder.parameters():
                param.requires_grad = False
    
    def unfreeze_encoder_layers(self, num_layers: int = 2):
        """
        Unfreeze top N layers of ESM encoders.
        Only works for ESM-2 encoders.
        """
        if self.config['encoder_type'] == 'embedding':
            print("Warning: unfreeze_encoder_layers has no effect on embedding encoders")
            return
        
        if self.shared_encoder:
            self.shared_esm_encoder.unfreeze_layers = num_layers
            self.shared_esm_encoder._apply_freeze_strategy()
        else:
            self.peptide_encoder.unfreeze_layers = num_layers
            self.peptide_encoder._apply_freeze_strategy()
            self.mhc_encoder.unfreeze_layers = num_layers
            self.mhc_encoder._apply_freeze_strategy()
    
    def encode_sequences(
        self,
        peptide_tokens: torch.Tensor,  # [batch, pep_len]
        mhc_tokens: torch.Tensor,      # [batch, mhc_len]
        peptide_mask: Optional[torch.Tensor] = None,  # [batch, pep_len] True=valid
        mhc_mask: Optional[torch.Tensor] = None       # [batch, mhc_len] True=valid
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode peptide and MHC sequences.
        
        Args:
            peptide_tokens: Tokenized peptide [batch, pep_len]
            mhc_tokens: Tokenized MHC [batch, mhc_len]
            peptide_mask: Boolean mask where True = valid, False = padding
            mhc_mask: Boolean mask where True = valid, False = padding
        
        Returns:
            peptide_emb: [batch, pep_len', dim] (pep_len' = pep_len-1 for ESM, BOS removed)
            mhc_emb: [batch, mhc_len', dim] (mhc_len' = mhc_len-1 for ESM, BOS removed)
            peptide_mask_out: Adjusted mask matching peptide_emb length (EOS marked invalid)
            mhc_mask_out: Adjusted mask matching mhc_emb length (EOS marked invalid)
        
        Note:
            For ESM encoders:
            - Input: [BOS, seq..., EOS, PAD...]
            - Output: [seq..., (EOS zeroed), (PAD zeroed)]
            - BOS is removed (position 0), EOS is zeroed via mask
        """
        
        peptide_emb = self.peptide_encoder(peptide_tokens, mask=peptide_mask)
        mhc_emb = self.mhc_encoder(mhc_tokens, mask=mhc_mask)
        
        # Adjust masks for ESM encoders
        peptide_mask_out = peptide_mask
        mhc_mask_out = mhc_mask
        
        if self._esm_eos_idx is not None:
            # ESM encoders remove BOS and zero out EOS
            # The encoder handles this internally, but we need to provide
            # the adjusted mask for downstream modules
            eos_idx = self._esm_eos_idx
            
            if peptide_mask is not None:
                # Mark EOS as invalid, then remove BOS position
                is_eos = (peptide_tokens == eos_idx)
                pep_mask_no_eos = peptide_mask.clone()
                pep_mask_no_eos[is_eos] = False
                peptide_mask_out = pep_mask_no_eos[:, 1:]  # Remove BOS position
                
            if mhc_mask is not None:
                is_eos = (mhc_tokens == eos_idx)
                mhc_mask_no_eos = mhc_mask.clone()
                mhc_mask_no_eos[is_eos] = False
                mhc_mask_out = mhc_mask_no_eos[:, 1:]  # Remove BOS position
        
        return peptide_emb, mhc_emb, peptide_mask_out, mhc_mask_out

    def _extract_flank_features(
        self,
        peptide_enhanced: torch.Tensor,  # [batch, seq_len, dim]
        nflank_len: torch.Tensor,        # [batch] N-terminal flank lengths
        cflank_len: torch.Tensor,        # [batch] C-terminal flank lengths
        original_peptide_len: torch.Tensor,  # [batch] Original peptide lengths
        peptide_mask: Optional[torch.Tensor] = None,  # [batch, seq_len]
        pooling: str = 'mean'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract flank features and core peptide from the enhanced peptide representation.

        Strategy:
        1. For each sample in batch, extract N-flank, core peptide, and C-flank regions
        2. Pool N-flank and C-flank features separately (mean or max pooling)
        3. Return concatenated flank features [batch, 2*dim] and core peptide [batch, core_len, dim]

        Args:
            peptide_enhanced: Enhanced peptide representation [batch, seq_len, dim]
            nflank_len: Length of N-terminal flank for each sample [batch]
            cflank_len: Length of C-terminal flank for each sample [batch]
            peptide_mask: Optional mask for valid positions [batch, seq_len]
            pooling: Pooling strategy ('mean' or 'max')

        Returns:
            flank_features: Concatenated flank features [batch, 2*dim]
            peptide_core: Core peptide without flanks [batch, max_core_len, dim]

        Note:
            - Handles variable flank lengths (0-5 residues)
            - Handles 'X' residues and padding via mask
            - Empty flanks result in zero-filled features
        """
        batch_size, seq_len, dim = peptide_enhanced.shape
        device = peptide_enhanced.device

        # Calculate core peptide lengths for each sample
        # seq_len includes both flanks and core peptide
        core_lengths = original_peptide_len # [batch]
        max_core_len = core_lengths.max().item()

        # Initialize outputs
        nflank_pool = torch.zeros(batch_size, dim, device=device, dtype=peptide_enhanced.dtype)
        cflank_pool = torch.zeros(batch_size, dim, device=device, dtype=peptide_enhanced.dtype)
        peptide_core = torch.zeros(batch_size, max_core_len, dim, device=device, dtype=peptide_enhanced.dtype)

        # Process each sample in the batch
        for i in range(batch_size):
            n_len = nflank_len[i].item()
            c_len = cflank_len[i].item()
            core_len = core_lengths[i].item()

            # Extract regions (indices are 0-based)
            # Structure: [nflank (0:n_len) | core (n_len:n_len+core_len) | cflank (n_len+core_len:)]

            # N-terminal flank
            if n_len > 0:
                nflank_features = peptide_enhanced[i, :n_len, :]  # [n_len, dim]

                # Apply mask if available
                if peptide_mask is not None:
                    nflank_mask = peptide_mask[i, :n_len]  # [n_len]
                    if nflank_mask.any():  # Only pool if there are valid positions
                        # Mask out invalid positions (padding, 'X', etc.)
                        nflank_features_masked = nflank_features[nflank_mask]  # [valid_len, dim]

                        if pooling == 'mean':
                            nflank_pool[i] = nflank_features_masked.mean(dim=0)
                        else:  # max pooling
                            nflank_pool[i] = nflank_features_masked.max(dim=0)[0]
                else:
                    # No mask, pool all positions
                    if pooling == 'mean':
                        nflank_pool[i] = nflank_features.mean(dim=0)
                    else:  # max pooling
                        nflank_pool[i] = nflank_features.max(dim=0)[0]

            # Core peptide
            core_start = n_len
            core_end = n_len + core_len
            peptide_core[i, :core_len, :] = peptide_enhanced[i, core_start:core_end, :]

            # C-terminal flank
            if c_len > 0:
                cflank_start = n_len + core_len
                cflank_end = cflank_start + c_len
                cflank_features = peptide_enhanced[i, cflank_start:cflank_end, :]  # [c_len, dim]

                # Apply mask if available
                if peptide_mask is not None:
                    cflank_mask = peptide_mask[i, cflank_start:cflank_end]  # [c_len]
                    if cflank_mask.any():  # Only pool if there are valid positions
                        cflank_features_masked = cflank_features[cflank_mask]  # [valid_len, dim]

                        if pooling == 'mean':
                            cflank_pool[i] = cflank_features_masked.mean(dim=0)
                        else:  # max pooling
                            cflank_pool[i] = cflank_features_masked.max(dim=0)[0]
                else:
                    # No mask, pool all positions
                    if pooling == 'mean':
                        cflank_pool[i] = cflank_features.mean(dim=0)
                    else:  # max pooling
                        cflank_pool[i] = cflank_features.max(dim=0)[0]

        # Concatenate N-flank and C-flank features
        flank_features = torch.cat([nflank_pool.unsqueeze(-2), cflank_pool.unsqueeze(-2)], dim=-2)  # [batch, 2*dim]

        return flank_features, peptide_core

    def forward(
        self,
        peptide_tokens: torch.Tensor,
        mhc_tokens: torch.Tensor,
        peptide_mask: Optional[torch.Tensor] = None,  # [batch, pep_len] True=valid
        mhc_mask: Optional[torch.Tensor] = None,      # [batch, mhc_len] True=valid
        nflank_len: Optional[torch.Tensor] = None,    # [batch] N-terminal flank lengths
        cflank_len: Optional[torch.Tensor] = None,    # [batch] C-terminal flank lengths
        original_peptide_len: Optional[torch.Tensor] = None,  # [batch] original lengths
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through DinoMHC.

        Args:
            peptide_tokens: Tokenized peptide sequence [batch, pep_len]
            mhc_tokens: Tokenized MHC sequence [batch, mhc_len]
            peptide_mask: Boolean mask where True = valid, False = padding
            mhc_mask: Boolean mask where True = valid, False = padding
            return_attention: Whether to return attention weights

        Returns:
            Dict containing:
                - prediction: Task-specific output (affinity/presentation/contact_map)
                - attention_info: (optional) Attention weights
        """
        # === Step 1: Encode sequences ===
        # Note: For ESM encoders, masks are adjusted (BOS/EOS stripped)
        peptide_emb, mhc_emb, peptide_mask_adj, mhc_mask_adj = self.encode_sequences(
            peptide_tokens, mhc_tokens,
            peptide_mask=peptide_mask, mhc_mask=mhc_mask
        )
        
        # === Step 2: Extract groove from MHC ===
        # Note: groove_emb has fixed size (num_groove_tokens), so no mask needed for it
        groove_emb, groove_attn = self.groove_extractor(
            mhc_emb, peptide_emb, 
            mhc_mask=mhc_mask_adj,
            peptide_mask=peptide_mask_adj,
            return_attention=return_attention
        )

        # === Step 3: Position-aware peptide encoding ===
        peptide_enhanced, position_importance = self.position_encoder(
            peptide_emb,
            peptide_mask=peptide_mask_adj,
            return_importance=return_attention
        )

        # === Step 3.5: Extract flank features if using flanks ===
        flank_features = None
        peptide_core = peptide_enhanced  # Default: use full sequence
        peptide_core_mask = peptide_mask_adj  # Default: use full mask

        if self.config.get('use_flanks', False) and nflank_len is not None and cflank_len is not None:
            # Extract core peptide region (without flanks) for binding interaction
            # and pool flank regions separately for context
            flank_features, peptide_core = self._extract_flank_features(
                peptide_enhanced, nflank_len, cflank_len, original_peptide_len,
                peptide_mask=peptide_mask_adj,
                pooling=self.config.get('flank_pooling', 'mean')
            )

            # Create mask for peptide_core
            # peptide_core shape: [batch, max_core_len, dim]
            batch_size = peptide_core.shape[0]
            max_core_len = peptide_core.shape[1]
            peptide_core_mask = torch.zeros(batch_size, max_core_len, dtype=torch.bool, device=peptide_core.device)

            # Set valid positions for each sample based on actual core length
            for i in range(batch_size):
                peptide_core_mask[i, :original_peptide_len[i]] = True

        # === Step 4: Groove-Peptide Fusion ===
        # groove_emb has no padding (fixed learned tokens), so groove_mask=None
        # Use peptide_core (without flanks) for binding interaction
        peptide_fused, groove_fused, fusion_attn = self.groove_peptide_fusion(
            peptide_core, groove_emb,
            peptide_mask=peptide_core_mask,
            groove_mask=None,  # Groove tokens are learned, no padding
            return_attention=return_attention
        )

        # === Step 5: Interface Geometry Reasoning ===
        # Contact prediction needs pair_rep, others need interface_rep
        # Pass peptide_core_mask so InterfaceGeometryModule knows valid positions
        return_pair = (self.config['task_head'] == 'contact')
        interface_rep, pair_rep = self.interface_geometry(
            peptide_fused, groove_fused,
            peptide_mask=peptide_core_mask,
            groove_mask=None,  # Groove tokens are learned, no padding
            return_pair_rep=return_pair
        )

        # === Step 6: Task-specific Prediction ===
        outputs = {}

        # Use appropriate input for the task head
        if self.config['task_head'] == 'contact':
            # Contact prediction uses pair representation
            outputs['prediction'] = self.task_head(pair_rep)
        else:
            # Affinity and presentation use interface representation
            # Pass flank_features for additional context if available
            outputs['prediction'] = self.task_head(interface_rep, flank_features=flank_features)

        if return_attention:
            outputs['attention_info'] = {
                'groove_extraction': groove_attn,
                'position_importance': position_importance,
                'fusion': fusion_attn
            }

        return outputs 
# END============================================================================
# Main Net
# ===============================================================================


if __name__ == "__main__":
    import sys
    sys.path.append('/home/duongtt/Workdir/MHC/digermhc/')
    from src.data_module import MHCPeptideDataset
    
    dataset = MHCPeptideDataset(
        data_path='/home/duongtt/Workdir/MHC/digermhc/datasets/el/NEWEST/fold_0_test_tmp.csv',
        tokenizer_type='esm2',
        esm_model_name='facebook/esm2_t33_650M_UR50D',
        use_flanks=True
    )
    
    tokenizer = dataset.tokenizer

    i = 0
    test_batch = []
    # Find until get the length is 16
    while True:
        test_sample = dataset[i]
        if 24 in test_sample['peptide_tokens']:
            print(test_sample)
            exit()
        if len(test_batch) == 1:
            break
    
    # for test_sample in test_batch:
    #     print(' '.join([tokenizer.alphabet.get_tok(x) for x in test_sample['peptide_tokens']]))
    #     print(test_sample['peptide_mask'])
    #     print('---')
        
    peptide_tokens_batch = torch.stack([sample['peptide_tokens'] for sample in test_batch], dim=0)
    mhc_tokens_batch = torch.stack([sample['mhc_tokens'] for sample in test_batch], dim=0)
    peptide_mask_batch = torch.stack([sample['peptide_mask'] for sample in test_batch], dim=0)
    mhc_mask_batch = torch.stack([sample['mhc_mask'] for sample in test_batch], dim=0)
    nflank_len_batch = torch.stack([sample['nflank_len'] for sample in test_batch], dim=0)
    cflank_len_batch = torch.stack([sample['cflank_len'] for sample in test_batch], dim=0)
    original_peptide_len_batch = torch.stack([sample['original_peptide_len'] for sample in test_batch], dim=0)
    
    model = DinoMHC()
    
    outputs = model(
        peptide_tokens=peptide_tokens_batch,
        mhc_tokens=mhc_tokens_batch,
        peptide_mask=peptide_mask_batch,
        mhc_mask=mhc_mask_batch,
        nflank_len=nflank_len_batch,
        cflank_len=cflank_len_batch,
        original_peptide_len=original_peptide_len_batch,
        return_attention=True
    )