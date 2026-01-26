import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple, Dict, Any


# ============================================================================
# Custom Modules
# ============================================================================
class MultiHeadAttentionWithWeights(nn.Module):
    """Multi-head attention that returns attention weights"""
    num_heads: int
    qkv_features: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key: jnp.ndarray,
        value: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key.shape
        head_dim = self.qkv_features // self.num_heads
        
        # Project Q, K, V
        query_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            axis=-1,
            name='query'
        )(query) # [batch, q_len, num_heads, head_dim]

        key_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            axis=-1,
            name='key'
        )(key) # [batch, kv_len, num_heads, head_dim]

        value_proj = nn.DenseGeneral(
            features=(self.num_heads, head_dim),
            axis=-1,
            name='value'
        )(value) # [batch, kv_len, num_heads, head_dim]

        # Transpose for attention: [batch, num_heads, seq_len, head_dim]
        query_proj = jnp.transpose(query_proj, (0, 2, 1, 3))
        key_proj = jnp.transpose(key_proj, (0, 2, 1, 3))
        value_proj = jnp.transpose(value_proj, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = jnp.sqrt(head_dim).astype(query.dtype)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', query_proj, key_proj) / scale

        # Apply mask (True=attend, False=ignore)
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, jnp.finfo(attn_weights.dtype).min)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)
        
        # Apply dropout
        attn_weights = nn.Dropout(rate=self.dropout_rate)(
            attn_weights, deterministic=deterministic
        )

        # Compute output
        attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, value_proj)

        # Transpose back: [batch, q_len, num_heads, head_dim]
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3))

        # Project output
        output = nn.DenseGeneral(
            features=self.qkv_features,
            axis=(-2, -1),
            name='output'
        )(attn_output)

        return output, attn_weights


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""
    dim: int
    ff_dim: int
    dropout: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        x = nn.Dense(self.ff_dim, name='fc1')(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        x = nn.Dense(self.dim, name='fc2')(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        return x



# ============================================================================
# Groove Extractor
# ============================================================================
class HierarchicalGrooveExtractor(nn.Module):
    dim: int
    num_groove_tokens: int = 60
    num_regions: int = 3
    num_heads: int = 8
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        mhc_emb: jnp.ndarray,
        peptide_emb: Optional[jnp.ndarray] = None,
        mhc_mask: Optional[jnp.ndarray] = None,
        peptide_mask: Optional[jnp.ndarray] = None,
        return_attention: bool = False,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, Optional[Dict]]:
        batch_size = mhc_emb.shape[0]

        # Level 1: Region Identification
        region_queries = self.param(
            'region_queries',
            nn.initializers.normal(stddev=0.02),
            (self.num_regions, self.dim)
        )

        region_queries_batch = jnp.broadcast_to(
            region_queries[None, :, :],
            (batch_size, self.num_regions, self.dim)
        )

        # Create attention mask for MHC
        # In Flax, mask uses True=attend, False=ignore
        if mhc_mask is not None:
            mhc_attn_mask = mhc_mask[:, None, None, :] # [batch, 1, 1, mhc_len]
        else:
            mhc_attn_mask = None

        # Region attention
        region_features, region_attn = MultiHeadAttentionWithWeights(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            name='region_attention'
        )(
            region_queries_batch, mhc_emb, mhc_emb,
            mask=mhc_attn_mask,
            deterministic=deterministic
        )

        region_features = nn.LayerNorm(name='region_norm')(region_features)

        # Compute region importance
        region_importance = jnp.mean(region_attn, axis=1)

        # Weight MHC by region importance
        weighted_mhc = jnp.einsum('brl,bld->brd', region_importance, mhc_emb)

        # Gate and combine with original
        gate = nn.Sequential([
            nn.Dense(self.dim, name='gate_dense'),
            nn.sigmoid
        ], name='region_gate')(weighted_mhc)

        mhc_region_enhanced = mhc_emb + jnp.einsum(
            'brd,brl->bld', gate * weighted_mhc, region_importance
        )

        # Level 2: Fine-grained Groove Extraction
        groove_queries = self.param(
            'groove_queries',
            nn.initializers.normal(stddev=0.02),
            (self.num_groove_tokens, self.dim)
        )

        groove_queries_batch = jnp.broadcast_to(
            groove_queries[None, :, :],
            (batch_size, self.num_groove_tokens, self.dim)
        )

        # Modulate queries with peptide information if available
        if peptide_emb is not None:
            if peptide_mask is not None:
                masked_peptide = peptide_emb * peptide_mask[:, :, None].astype(peptide_emb.dtype)
                pep_lengths = peptide_mask.sum(axis=1, keepdims=True).astype(peptide_emb.dtype)
                pep_lengths = jnp.maximum(pep_lengths, 1.0)
                pep_context = masked_peptide.sum(axis=1, keepdims=True) / pep_lengths[:, :, None]
            else:
                pep_context = jnp.mean(peptide_emb, axis=1, keepdims=True)
            
            pep_modulation = nn.Dense(self.dim, name='peptide_modulation')(pep_context)
            groove_queries_batch = groove_queries_batch + pep_modulation

        # Attend to region-enhanced MHC
        groove_features, groove_attn = MultiHeadAttentionWithWeights(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            name='groove_attention'
        )(
            groove_queries_batch, mhc_region_enhanced, mhc_region_enhanced,
            mask=mhc_attn_mask,
            deterministic=deterministic
        )

        groove_features = nn.LayerNorm(name='groove_norm')(groove_features)

        # Output projection
        groove_features = nn.Dense(self.dim, name='output_proj')(groove_features)
        groove_features = nn.LayerNorm(name='output_norm')(groove_features)

        attention_info = None
        if return_attention:
            attention_info = {
                'region_attention': region_attn,
                'groove_attention': groove_attn,
                'region_importance': region_importance
            }
        
        return groove_features, attention_info



# ============================================================================
# Groove-Peptide Fusion
# ============================================================================
class GeometricBiasModule(nn.Module):
    """Learns geometric/distance biases between peptide and groove positions."""
    dim: int
    num_heads: int
    hidden_dim: Optional[int] = None
    dropout: float = 0.1
    max_query_len: int = 15
    max_key_len: int = 60

    @nn.compact
    def __call__(
        self,
        query_emb: jnp.ndarray,
        key_emb: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        
        hidden_dim = self.hidden_dim or self.dim // 2
        batch_size, query_len, _ = query_emb.shape
        _, key_len, _ = key_emb.shape

        # Create all pairs by broadcasting
        query_expanded = query_emb[:, :, None, :]
        key_expanded = key_emb[:, None, :, :]

        # Broadcast to create pair matrix
        query_expanded = jnp.broadcast_to(
            query_expanded, (batch_size, query_len, key_len, self.dim)
        )
        key_expanded = jnp.broadcast_to(
            key_expanded, (batch_size, query_len, key_len, self.dim)
        )

        # Concatenate
        pairs = jnp.concatenate([
            query_expanded,
            key_expanded
        ], axis=-1)

        # Pair encoder MLP
        x = nn.Dense(hidden_dim, name='pair_enc_1')(pairs)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        x = nn.Dense(hidden_dim, name='pair_enc_2')(x)
        x = nn.relu(x)
        x = nn.Dropout(self.dropout, deterministic=deterministic)(x)
        bias = nn.Dense(self.num_heads, name='pair_enc_out')(x)


        # Reshape to attention format
        bias = jnp.transpose(bias, (0, 3, 1, 2))

        # Position-specific learned biases
        query_pos_bias = self.param(
            'query_pos_bias',
            nn.initializers.zeros,
            (1, self.num_heads, self.max_query_len, 1)
        )

        key_pos_bias = self.param(
            'key_pos_bias',
            nn.initializers.zeros,
            (1, self.num_heads, 1, self.max_key_len)
        )

        # Handle variable sequence lengths with dynamic slicing
        # For query position bias
        query_bias_len = jnp.minimum(query_len, self.max_query_len)
        qpb = jax.lax.dynamic_slice(
            query_pos_bias,
            (0, 0, 0, 0),
            (1, self.num_heads, query_bias_len, 1)
        )
        # Pad if query is longer than max
        if query_len > self.max_query_len:
            pad_width = ((0, 0), (0, 0), (0, query_len - self.max_query_len), (0, 0))
            qpb = jnp.pad(qpb, pad_width, mode='constant', constant_values=0)
        
        # For key position bias
        key_bias_len = jnp.minimum(key_len, self.max_key_len)
        kpb = jax.lax.dynamic_slice(
            key_pos_bias,
            (0, 0, 0, 0),
            (1, self.num_heads, 1, key_bias_len)
        )
        # Pad if key is longer than max
        if key_len > self.max_key_len:
            pad_width = ((0, 0), (0, 0), (0, 0), (0, key_len - self.max_key_len))
            kpb = jnp.pad(kpb, pad_width, mode='constant', constant_values=0)
        
        bias = bias + qpb + kpb

        return bias

class CrossAttentionWithGeometricBias(nn.Module):
    """Cross-attention module with geometric bias"""
    dim: int
    num_heads: int = 8
    head_dim: Optional[int] = None
    dropout: float = 0.1
    use_geometric_bias: bool = True
    max_query_len: int = 15
    max_key_len: int = 60

    @nn.compact
    def __call__(
        self,
        query: jnp.ndarray,
        key_value: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_attention: bool = False
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        head_dim = self.head_dim or self.dim // self.num_heads
        batch_size, query_len, _ = query.shape
        _, kv_len, _ = key_value.shape
        scale = head_dim ** -0.5

        # Project to Q, K, V
        Q = nn.Dense(
            self.num_heads * head_dim, use_bias=False, name='q_proj'
        )(query)

        K = nn.Dense(
            self.num_heads * head_dim, use_bias=False, name='k_proj'
        )(key_value)

        V = nn.Dense(
            self.num_heads * head_dim, use_bias=False, name='v_proj'
        )(key_value)

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, query_len, self.num_heads, head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        
        K = K.reshape(batch_size, kv_len, self.num_heads, head_dim)
        K = jnp.transpose(K, (0, 2, 1, 3))
        
        V = V.reshape(batch_size, kv_len, self.num_heads, head_dim)
        V = jnp.transpose(V, (0, 2, 1, 3))

        # Compute attention score
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', Q, K) * scale

        # Add geometric bias
        if self.use_geometric_bias:
            geo_bias = GeometricBiasModule(
                dim=self.dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                max_query_len=self.max_query_len,
                max_key_len=self.max_key_len,
                name='geometric_bias'
            )(query, key_value, deterministic=deterministic)
            attn_scores = attn_scores + geo_bias
        
        # Apply attention mask (True=attend, False=mask out)
        if attention_mask is not None:
            attn_scores = jnp.where(
                attention_mask,
                attn_scores,
                jnp.finfo(attn_scores.dtype).min
            )

        # Softmax and dropout
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_weights = nn.Dropout(self.dropout, deterministic=deterministic)(attn_weights)

        # Apply attention to values
        output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, V)

        # Reshape back
        output = jnp.transpose(output, (0, 2, 1, 3))
        output = output.reshape(batch_size, query_len, self.num_heads * head_dim)

        # Output projection
        output = nn.Dense(self.dim, name='out_proj')(output)
        output = nn.Dropout(self.dropout, deterministic=deterministic)(output)

        if return_attention:
            return output, attn_weights
        return output, None

class GroovePeptideFusionLayer(nn.Module):
    """Single layer of Groove-Peptide Fusion"""
    dim: int
    num_heads: int = 8
    ff_dim: Optional[int] = None
    dropout: float = 0.1
    max_peptide_length: int = 15
    num_groove_tokens: int = 60

    @nn.compact
    def __call__(
        self,
        peptide_emb: jnp.ndarray,
        groove_emb: jnp.ndarray,
        peptide_mask: Optional[jnp.ndarray] = None,
        groove_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_attention: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[Dict]]:
        ff_dim = self.ff_dim or self.dim * 4
        attention_info = {} if return_attention else None

        # Create cross-attention masks
        pep_to_groove_mask = None
        if groove_mask is not None:
            pep_to_groove_mask = groove_mask[:, None, None, :]
        
        groove_to_pep_mask = None
        if peptide_mask is not None:
            groove_to_pep_mask = peptide_mask[:, None, None, :]
            

        # === Peptide Stream === 
        # Self attention
        pep_self, _ = MultiHeadAttentionWithWeights(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            name='peptide_self_attn'
        )(
            peptide_emb, peptide_emb, peptide_emb,
            mask=groove_to_pep_mask,
            deterministic=deterministic
        )
        peptide_emb = nn.LayerNorm(name='peptide_self_norm')(peptide_emb + pep_self)

        # Cross-attention (peptide queries groove)
        pep_cross, pep_to_groove_attn = CrossAttentionWithGeometricBias(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            use_geometric_bias=True,
            max_query_len=self.max_peptide_length,
            max_key_len=self.num_groove_tokens,
            name='peptide_to_groove_attn'
        )(
            peptide_emb, groove_emb,
            attention_mask=pep_to_groove_mask,
            deterministic=deterministic,
            return_attention=return_attention
        )
        peptide_emb = nn.LayerNorm(name='peptide_cross_norm')(peptide_emb + pep_cross)

        # Feed-forward
        pep_ffn = FeedForward(
            dim=self.dim, ff_dim=ff_dim, dropout=self.dropout,
            name='peptide_ffn'
        )(peptide_emb, deterministic=deterministic)
        peptide_emb = nn.LayerNorm(name='peptide_ffn_norm')(peptide_emb + pep_ffn)

        # === Groove Stream ===
        # Self attention
        groove_self, _ = MultiHeadAttentionWithWeights(
            num_heads=self.num_heads,
            qkv_features=self.dim,
            dropout_rate=self.dropout,
            name='groove_self_attn'
        )(
            groove_emb, groove_emb, groove_emb,
            mask=pep_to_groove_mask,
            deterministic=deterministic
        )
        groove_emb = nn.LayerNorm(name='groove_self_norm')(groove_emb + groove_self)
        
        # Cross-attention (groove queries peptide)
        groove_cross, groove_to_pep_attn = CrossAttentionWithGeometricBias(
            dim=self.dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            use_geometric_bias=True,
            max_query_len=self.num_groove_tokens,
            max_key_len=self.max_peptide_length,
            name='groove_to_peptide_attn'
        )(
            groove_emb, peptide_emb,
            attention_mask=groove_to_pep_mask,
            deterministic=deterministic,
            return_attention=return_attention
        )
        groove_emb = nn.LayerNorm(name='groove_cross_norm')(groove_emb + groove_cross)
        
        # Feed-forward
        groove_ffn = FeedForward(
            dim=self.dim, ff_dim=ff_dim, dropout=self.dropout,
            name='groove_ffn'
        )(groove_emb, deterministic=deterministic)
        groove_emb = nn.LayerNorm(name='groove_ffn_norm')(groove_emb + groove_ffn)

        # Zero out padded positions
        if peptide_mask is not None:
            peptide_emb = peptide_emb * peptide_mask[:, :, None].astype(peptide_emb.dtype)
        if groove_mask is not None:
            groove_emb = groove_emb * groove_mask[:, :, None].astype(groove_emb.dtype)
        
        if return_attention:
            attention_info = {
                'peptide_to_groove': pep_to_groove_attn,
                'groove_to_peptide': groove_to_pep_attn
            }
        
        return peptide_emb, groove_emb, attention_info

class GroovePeptideFusion(nn.Module):
    """Complete Groove-Peptide Fusion Module."""
    dim: int
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: Optional[int] = None
    dropout: float = 0.1
    max_peptide_length: int = 15
    num_groove_tokens: int = 60

    @nn.compact
    def __call__(
        self,
        peptide_emb: jnp.ndarray,   # [batch, pep_len, dim]
        groove_emb: jnp.ndarray,    # [batch, groove_len, dim]
        peptide_mask: Optional[jnp.ndarray] = None,
        groove_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        return_attention: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[Dict]]:
        all_attention = {} if return_attention else None
        
        for i in range(self.num_layers):
            peptide_emb, groove_emb, attn_info = GroovePeptideFusionLayer(
                dim=self.dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout,
                max_peptide_length=self.max_peptide_length,
                num_groove_tokens=self.num_groove_tokens,
                name=f'layer_{i}'
            )(
                peptide_emb, groove_emb,
                peptide_mask=peptide_mask,
                groove_mask=groove_mask,
                deterministic=deterministic,
                return_attention=return_attention
            )
            
            if return_attention:
                all_attention[f'layer_{i}'] = attn_info
        
        # Final normalization
        peptide_fused = nn.LayerNorm(name='peptide_final_norm')(peptide_emb)
        groove_fused = nn.LayerNorm(name='groove_final_norm')(groove_emb)
        
        # Ensure padded positions stay zeroed
        if peptide_mask is not None:
            peptide_fused = peptide_fused * peptide_mask[:, :, None].astype(peptide_fused.dtype)
        if groove_mask is not None:
            groove_fused = groove_fused * groove_mask[:, :, None].astype(groove_fused.dtype)
        
        return peptide_fused, groove_fused, all_attention  


if __name__ == '__main__':
    import os
    os.environ['TPU_CHIPS_PER_HOST_BOUNDS'] = '2,2,1'
    os.environ['TPU_HOST_BOUNDS'] = '1,1,1'

    from jax import random
    
    # Initialize model
    model = GroovePeptideFusion(
        dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_peptide_length=15,
        num_groove_tokens=60
    )
    
    # Create dummy inputs
    key = random.PRNGKey(42)
    batch_size = 4
    pep_len = 12
    groove_len = 60
    
    peptide_emb = random.normal(key, (batch_size, pep_len, 256))
    groove_emb = random.normal(key, (batch_size, groove_len, 256))
    peptide_mask = jnp.ones((batch_size, pep_len), dtype=jnp.bool_)
    groove_mask = jnp.ones((batch_size, groove_len), dtype=jnp.bool_)
    
    # Initialize parameters
    variables = model.init(
        {'params': key, 'dropout': random.PRNGKey(1)},
        peptide_emb, groove_emb,
        peptide_mask, groove_mask,
        deterministic=True,
        return_attention=False
    )
    
    print(f"Total parameters: {sum(p.size for p in jax.tree_util.tree_leaves(variables['params'])):,}")
    
    # Forward pass (inference)
    peptide_out, groove_out, attn = model.apply(
        variables,
        peptide_emb, groove_emb,
        peptide_mask, groove_mask,
        deterministic=True,
        return_attention=True
    )
    
    print(f"Peptide output shape: {peptide_out.shape}")  # [4, 12, 256]
    print(f"Groove output shape: {groove_out.shape}")    # [4, 60, 256]
    print(f"Attention keys: {list(attn.keys())}")
    
    # Forward pass (training)
    peptide_out, groove_out, _ = model.apply(
        variables,
        peptide_emb, groove_emb,
        peptide_mask, groove_mask,
        deterministic=False,
        return_attention=False,
        rngs={'dropout': random.PRNGKey(2)}
    )
    
    print("\nModel test passed!")