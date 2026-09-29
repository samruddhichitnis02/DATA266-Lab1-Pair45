import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    Configuration for the character-level GPT model.
    """

    vocab_size: int
    block_size: int = 512

    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6

    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """
    Masked multi-head self-attention.

    The causal mask prevents a character from looking
    at future characters.
    """

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size

        # One layer produces Query, Key, and Value together
        self.qkv_projection = nn.Linear(
            config.n_embd,
            3 * config.n_embd
        )

        self.output_projection = nn.Linear(
            config.n_embd,
            config.n_embd
        )

        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)

        # Lower-triangular mask
        # It allows each position to attend only to
        # itself and previous positions.
        causal_mask = torch.tril(
            torch.ones(
                config.block_size,
                config.block_size
            )
        )

        self.register_buffer(
            "causal_mask",
            causal_mask.view(
                1,
                1,
                config.block_size,
                config.block_size
            )
        )

    def forward(self, x):
        batch_size, sequence_length, embedding_size = x.shape

        # Create query, key, and value
        qkv = self.qkv_projection(x)

        # Shape:
        # [batch, sequence, 3, heads, head_dimension]
        qkv = qkv.view(
            batch_size,
            sequence_length,
            3,
            self.n_head,
            self.head_dim
        )

        # Shape after permutation:
        # [3, batch, heads, sequence, head_dimension]
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attention_scores = (
            query @ key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Apply the causal mask
        attention_scores = attention_scores.masked_fill(
            self.causal_mask[
                :, :, :sequence_length, :sequence_length
            ] == 0,
            float("-inf")
        )

        # Convert scores into probabilities
        attention_weights = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        # Weighted combination of values
        attention_output = attention_weights @ value

        # Change shape back to:
        # [batch, sequence, embedding]
        attention_output = attention_output.transpose(
            1,
            2
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            embedding_size
        )

        attention_output = self.output_projection(
            attention_output
        )

        attention_output = self.residual_dropout(
            attention_output
        )

        return attention_output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(self, config):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                config.n_embd,
                4 * config.n_embd
            ),
            nn.GELU(),
            nn.Linear(
                4 * config.n_embd,
                config.n_embd
            ),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.network(x)


class TransformerBlock(nn.Module):
    """
    One Transformer block containing:

    1. Layer normalization
    2. Masked self-attention
    3. Residual connection
    4. Layer normalization
    5. Feed-forward network
    6. Residual connection
    """

    def __init__(self, config):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            config.n_embd
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.layer_norm_2 = nn.LayerNorm(
            config.n_embd
        )

        self.feed_forward = FeedForward(
            config
        )

    def forward(self, x):
        # Attention residual connection
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Feed-forward residual connection
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x


class CharacterGPT(nn.Module):
    """
    Character-level GPT language model.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        # Converts character IDs into vectors
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # Gives each sequence position a vector
        self.position_embedding = nn.Embedding(
            config.block_size,
            config.n_embd
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.n_embd
        )

        # Converts hidden vectors into vocabulary scores
        self.output_layer = nn.Linear(
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module):
        """
        Initialize model weights.
        """

        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids, targets=None):
        """
        Parameters:
            input_ids:
                Shape [batch_size, sequence_length]

            targets:
                Shape [batch_size, sequence_length]

        Returns:
            logits:
                Shape [batch_size, sequence_length, vocab_size]

            loss:
                Cross-entropy loss if targets are provided
        """

        batch_size, sequence_length = input_ids.shape

        if sequence_length > self.config.block_size:
            raise ValueError(
                "Input sequence is longer than block_size."
            )

        # Position IDs: [0, 1, 2, ..., sequence_length - 1]
        position_ids = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        # Token embeddings
        token_vectors = self.token_embedding(
            input_ids
        )

        # Position embeddings
        position_vectors = self.position_embedding(
            position_ids
        )

        # Add token and position information
        x = token_vectors + position_vectors

        x = self.embedding_dropout(x)

        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.final_layer_norm(x)

        # Vocabulary logits
        logits = self.output_layer(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0
    ):
        """
        Generate new characters one at a time.
        """

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the most recent context
            input_context = input_ids[
                :, -self.config.block_size:
            ]

            # Get model predictions
            logits, _ = self(input_context)

            # Use only the final position
            logits = logits[:, -1, :]

            # Control randomness
            logits = logits / temperature

            # Convert scores to probabilities
            probabilities = F.softmax(
                logits,
                dim=-1
            )

            # Select the next character
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Add it to the sequence
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

        return input_ids