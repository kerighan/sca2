"""
SeqCond HuggingFace configuration.
"""

from transformers import PretrainedConfig


class SeqCondConfig(PretrainedConfig):
    """
    Configuration class for SeqCond models.

    SeqCond is a hybrid recurrent-transformer architecture that interleaves
    SeqCond (sequential conditioning) blocks with standard Transformer decoder
    blocks. SeqCond blocks replace softmax attention with a closed-form
    complex-exponential accumulator, enabling O(1) per-token decoding.

    Args:
        d_model: Hidden dimension.
        d_ff: Feed-forward dimension (typically 3×d_model).
        num_layers: Total number of blocks (SeqCond + Transformer).
        vocab_size: Vocabulary size.
        maxlen: Maximum sequence length (also sets KV-cache size).
        dropout: Dropout rate (0.0 disables).
        tie_weights: Whether to tie embedding and LM-head weights.
        num_heads: Number of attention heads in Transformer blocks.
        num_kv_heads: Number of KV heads (GQA). None = full MHA.
        qk_norm: Whether to apply QK-normalization in Transformer blocks.
        qk_norm_eps: Epsilon for QK-norm.
        seqcond_heads: Number of SeqCond memory heads (K).
        num_query_heads: Number of query heads in SeqCond (K_q, must divide K).
        num_thetas: Number of frequency components per head (M).
        derivative_order: Unused — kept for checkpoint compatibility.
        num_anchor_heads: Number of anchor heads (no decay) in SeqCond.
        conv_kernel_size: Depthwise conv kernel size inside SeqCond.
        expand_factor: Inner expansion factor for SeqCond memory dimension.
        out_expand_factor: SwiGLU expansion factor in SeqCond.
        use_positional_embedding: Whether to add learnable positional embeddings.
        seqcond_ratio: Block interleaving ratio. Every (seqcond_ratio+1)-th
            block (1-indexed) is a Transformer block; the rest are SeqCond.
        chunk_size: Chunk size for chunked computation (unused in PyTorch path).
        use_square_matrix: Unused — kept for checkpoint compatibility.
    """

    model_type = "seqcond"

    def __init__(
        self,
        # Core
        d_model: int = 768,
        d_ff: int = 2304,
        num_layers: int = 12,
        vocab_size: int = 100300,
        maxlen: int = 768,
        dropout: float = 0.0,
        tie_weights: bool = True,
        # Transformer block params
        num_heads: int = 8,
        num_kv_heads=None,
        qk_norm: bool = True,
        qk_norm_eps: float = 1e-6,
        # SeqCond block params
        seqcond_heads: int = 32,
        num_query_heads: int = 6,
        num_thetas: int = 4,
        derivative_order: int = 0,
        num_anchor_heads: int = 0,
        conv_kernel_size: int = 4,
        expand_factor: float = 2.0,
        out_expand_factor: int = 3,
        use_positional_embedding: bool = False,
        seqcond_ratio: int = 5,
        chunk_size: int = 128,
        use_square_matrix: bool = False,
        # Special token IDs (filled in by convert_checkpoint.py)
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=None,
        **kwargs,
    ):
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.tie_weights = tie_weights

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.qk_norm = qk_norm
        self.qk_norm_eps = qk_norm_eps

        self.seqcond_heads = seqcond_heads
        self.num_query_heads = num_query_heads
        self.num_thetas = num_thetas
        self.derivative_order = derivative_order
        self.num_anchor_heads = num_anchor_heads
        self.conv_kernel_size = conv_kernel_size
        self.expand_factor = expand_factor
        self.out_expand_factor = out_expand_factor
        self.use_positional_embedding = use_positional_embedding
        self.seqcond_ratio = seqcond_ratio
        self.chunk_size = chunk_size
        self.use_square_matrix = use_square_matrix

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
