import os
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from typing import Dict, Optional, Callable, Literal
from datasets import load_dataset

# Hyperparameters
SEED = 1337
BATCH_SIZE = 16  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 128  # what is the maximum context length for predictions?
MIN_SEQUENCE_LEN = 16  # minimum sequence length to train from and make predictions on (for cleaner data)
MAX_ITERS = 5000
EVAL_INTERVAL = MAX_ITERS // 10
LEARNING_RATE = 3e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EVAL_ITERS = 100
EMBEDDING_DIM = 128
NUM_HEADS = 4
NUM_TRANSFORMER_BLOCKS = 6
DROPOUT = 0.20
# ------------
torch.manual_seed(SEED)
CHECKPOINT_PATH = "model_checkpoint.pt"


def load_hinglish_english_dataset(split: Literal['train', 'validation', 'test']) -> (list[str], list[str]):
    # Loads the CMU Hinglish Dog dataset from HuggingFace Datasets and plucks output the input/output sequences

    def _valid_sequence_len(s: str) -> bool:
        # 1. for simplicity, simply filter out data points with more characters than will fit in the model
        # 2. for cleanliness of the data, only consider sequence pairs with a minimum number of characters in both input and output
        return MIN_SEQUENCE_LEN <= len(s) <= BLOCK_SIZE - 2

    dataset = load_dataset("cmu_hinglish_dog", split=split)

    input_dataset = []
    output_dataset = []
    for _ in range(len(dataset)):
        i = dataset[_]['translation']['hi_en']
        o = dataset[_]['translation']['en']
        if _valid_sequence_len(i) and _valid_sequence_len(o):
            input_dataset.append(i)
            output_dataset.append(o)

    return input_dataset, output_dataset


train_dataset_input, train_dataset_output = load_hinglish_english_dataset("train")
val_dataset_input, val_dataset_output = load_hinglish_english_dataset("validation")
test_dataset_input, test_dataset_output = load_hinglish_english_dataset("test")

input_corpus = ' '.join(train_dataset_input + val_dataset_input + test_dataset_input).lower()
output_corpus = ' '.join(train_dataset_output + val_dataset_output + test_dataset_output).lower()  # Use lower to effectively clean the dataset as the hinglish input is always lowercase

START_CHAR = '\u0002'
END_CHAR = '\u0003'
PAD_CHAR = '\u0000'


# Generate a character-level encoder and decoder for a corpus of text
def text_corpus_encoder_decoder_generator(corpus: str) -> (int, Callable[[str], list[int]], Callable[[list[int]], str]):
    chars = sorted(list(set(corpus)))
    assert START_CHAR not in chars
    assert END_CHAR not in chars
    assert PAD_CHAR not in chars
    chars = [START_CHAR, END_CHAR, PAD_CHAR] + chars
    vocab_size = len(chars)
    
    # Create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s.lower()]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    
    return vocab_size, encode, decode


INPUT_VOCAB_SIZE, INPUT_ENCODER, INPUT_DECODER = \
    text_corpus_encoder_decoder_generator(input_corpus)
OUTPUT_VOCAB_SIZE, OUTPUT_ENCODER, OUTPUT_DECODER = \
    text_corpus_encoder_decoder_generator(output_corpus)

INPUT_PAD_CHAR = torch.tensor(INPUT_ENCODER(PAD_CHAR), dtype=torch.long, device=DEVICE)
OUTPUT_PAD_CHAR = torch.tensor(OUTPUT_ENCODER(PAD_CHAR), dtype=torch.long, device=DEVICE)


# Add start, end, and pad chars as necessary and convert to tensors
def data_to_tensor_with_special_chars(data_input: list[str], data_output: list[str], is_generation: bool = False) -> (Tensor, Tensor):
    assert len(data_input) == len(data_output), \
        f"Batch size of inputs: {len(data_input)} should be the same as batch size of outputs: {len(data_output)}"

    def _input_string_transform(s: str) -> str:
        return START_CHAR + s + END_CHAR + PAD_CHAR * (BLOCK_SIZE - len(s) - 2)

    def _output_string_transform(s: str) -> str:
        if is_generation:
            # During generation, we don't want to add END_CHAR automatically as the model is supposed to tell us when the output is finished
            return START_CHAR + s + PAD_CHAR * (BLOCK_SIZE - len(s) - 1)
        else:
            return START_CHAR + s + END_CHAR + PAD_CHAR * (BLOCK_SIZE - len(s) - 2)

    return torch.tensor([INPUT_ENCODER(_input_string_transform(x)) for x in data_input]), \
        torch.tensor([OUTPUT_ENCODER(_output_string_transform(x)) for x in data_output])


def shift_output_idx_for_loss(output_idx: Tensor) -> Tensor:
    # Input: (B, T)
    # Output: (B, T[1:]+PAD) = (B, T)
    return torch.cat((output_idx[:, 1:], OUTPUT_PAD_CHAR.expand(output_idx.shape[0], 1)), dim=1)


train_data_input, train_data_output = data_to_tensor_with_special_chars(train_dataset_input, train_dataset_output)
val_data_input, val_data_output = data_to_tensor_with_special_chars(val_dataset_input, val_dataset_output)
test_data_input, test_data_output = data_to_tensor_with_special_chars(test_dataset_input, test_dataset_output)


# Function to retrieve a batch of data
def get_batch(split: Literal['train', 'validation', 'test']) -> (Tensor, Tensor):
    # Generate a small batch of data of inputs x and targets y with appropriate special chars and padding
    match split:
        case 'train':
            data_input, data_output = (train_data_input, train_data_output)
        case 'validation':
            data_input, data_output = (val_data_input, val_data_output)
        case _:
            data_input, data_output = (test_data_input, test_data_output)

    indices = torch.randint(data_input.shape[0], (BATCH_SIZE,))
    x = data_input[indices].to(DEVICE)
    y = data_output[indices].to(DEVICE)
    return x, y


class GenericAttentionHead(nn.Module):
    """ One head of attention, can be for self-attention, causal self-attention, or cross-attention """

    def __init__(self, head_size: int, needs_future_mask: bool):
        super().__init__()
        self.query = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.key = nn.Linear(EMBEDDING_DIM, head_size, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, head_size, bias=False)

        self.needs_future_mask = needs_future_mask
        if self.needs_future_mask:
            # We don't need this to be a model parameter; will remain unchanged by any gradients
            self.register_buffer('future_mask', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)).type(torch.bool).unsqueeze(0))  # (1, T, T)

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, source_embedding: Tensor, target_embedding: Tensor, pad_mask: Tensor) -> Tensor:
        B, T, C = source_embedding.shape
        assert (B, T, C) == target_embedding.shape
        assert T == BLOCK_SIZE
        assert (B, T) == pad_mask.shape

        # Query comes from target embedding
        q = self.query(target_embedding)  # (B, T, HS)
        # Key and Value come from source embedding. In self-attention, source and target are the same
        k = self.key(source_embedding)  # (B, T, HS)
        v = self.value(source_embedding)  # (B, T, HS)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, HS) @ (B, HS, T) -> (B, T, T)
        
        # Masking affinities for pad characters
        pad_mask_unsqueezed = pad_mask.unsqueeze(dim=1)  # (B, 1, T)
        wei = wei.masked_fill(~pad_mask_unsqueezed, float('-inf'))  # (B, T, T)

        # If needed, apply the causal (future) mask so past tokens can't use input from future tokens in the target
        if self.needs_future_mask:
            wei = wei.masked_fill(~self.future_mask, float('-inf'))  # (B, T, T)
        
        # Convert to weighted affinities
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        out = wei @ v  # (B, T, T) @ (B, T, HS) -> (B, T, HS)
        return out


class SelfAttentionHead(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.head = GenericAttentionHead(head_size, needs_future_mask=False)

    def forward(self, embedding: Tensor, pad_mask: Tensor) -> Tensor:
        # Self attention head uses the same src and target embedding
        return self.head(embedding, embedding, pad_mask)


class CausalSelfAttentionHead(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        # Causal self-attention head requires a future mask
        self.head = GenericAttentionHead(head_size, needs_future_mask=True)

    def forward(self, embedding: Tensor, pad_mask: Tensor) -> Tensor:
        # Self attention head uses the same src and target embedding
        return self.head(embedding, embedding, pad_mask)


class CrossAttentionHead(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.head = GenericAttentionHead(head_size, needs_future_mask=False)

    def forward(self, source_embedding: Tensor, target_embedding: Tensor, pad_mask: Tensor) -> Tensor:
        return self.head(source_embedding, target_embedding, pad_mask)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(SelfAttentionHead(head_size) for _ in range(num_heads))
        self.projection = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, embedding: Tensor, pad_mask: Tensor) -> Tensor:
        # B, T, C = embedding.shape
        out = torch.cat([head(embedding, pad_mask) for head in self.heads], dim=-1)  # (B, T, HS * num_heads = C)
        out = self.dropout(self.projection(out))  # (B, T, C) -> (B, T, C)
        return out


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(CausalSelfAttentionHead(head_size) for _ in range(num_heads))
        self.projection = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, embedding: Tensor, pad_mask: Tensor) -> Tensor:
        # B, T, C = embedding.shape
        out = torch.cat([head(embedding, pad_mask) for head in self.heads], dim=-1)  # (B, T, HS * num_heads = C)
        out = self.dropout(self.projection(out))  # (B, T, C) -> (B, T, C)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.heads = nn.ModuleList(CrossAttentionHead(head_size) for _ in range(num_heads))
        self.projection = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, source_embedding: Tensor, target_embedding: Tensor, pad_mask: Tensor) -> Tensor:
        # B, T, C = source_embedding.shape = target_embedding.shape
        out = torch.cat([head(source_embedding, target_embedding, pad_mask) for head in self.heads], dim=-1)  # (B, T, HS * num_heads = C)
        out = self.dropout(self.projection(out))  # (B, T, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """ A simple linear layer followed by non-linearity """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class EncoderTransformerBlock(nn.Module):
    """ Encoder Transformer Block:  """

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        assert embedding_dim % num_heads == 0
        head_size = embedding_dim // num_heads
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)  # per batch, per token normalization
        self.self_attention = MultiHeadSelfAttention(num_heads, head_size)

        self.layer_norm_2 = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, source_embedding: Tensor, source_pad_mask: Tensor) -> Tensor:
        # NOTE: Layer normalization is done before self attention unlike the original paper due to it being empirically better
        # NOTE: We add source_embedding to itself to do the "residual" or "skip" connection
        #       This lets us propagate gradients due to supervision all the way to the early part of the network
        source_embedding = source_embedding + self.self_attention(self.layer_norm_1(source_embedding), source_pad_mask)
        source_embedding = source_embedding + self.feed_forward(self.layer_norm_2(source_embedding))
        return source_embedding


class DecoderTransformerBlock(nn.Module):
    """ Decoder Transformer Block:  """

    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()
        assert embedding_dim % num_heads == 0
        head_size = embedding_dim // num_heads

        self.layer_norm_self = nn.LayerNorm(embedding_dim)  # per batch, per token normalization
        self.self_attention = CausalMultiHeadSelfAttention(num_heads, head_size)

        self.layer_norm_src = nn.LayerNorm(embedding_dim)
        self.layer_norm_tgt = nn.LayerNorm(embedding_dim)
        self.cross_attention = MultiHeadCrossAttention(num_heads, head_size)

        self.layer_norm_final = nn.LayerNorm(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)

    def forward(self, 
                source_embedding: Tensor, target_embedding: Tensor,
                source_pad_mask: Tensor, target_pad_mask: Tensor) -> Tensor:
        # NOTE: Layer normalization is done before self attention unlike the original paper due to it being empirically better
        # NOTE: We make extensive use of residual or skip connections here.
        #       This lets us propagate gradients due to supervision all the way to the early part of the network

        # First, masked self-attention on the target embedding
        target_embedding = target_embedding + self.self_attention(
            self.layer_norm_self(target_embedding), 
            target_pad_mask)

        # Then, cross-attention using the source embedding learned by the decoder
        target_embedding = target_embedding + self.cross_attention(
            self.layer_norm_src(source_embedding), 
            self.layer_norm_tgt(target_embedding),
            source_pad_mask)
        
        # Finally a feedforward layer
        target_embedding = target_embedding + self.feed_forward(self.layer_norm_final(target_embedding))
        return target_embedding


class SeqToSeqTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the embeddings for the next token from lookup table
        self.input_token_embedding_table = nn.Embedding(INPUT_VOCAB_SIZE, EMBEDDING_DIM)
        self.output_token_embedding_table = nn.Embedding(OUTPUT_VOCAB_SIZE, EMBEDDING_DIM)
        
        # Using learned position embeddings here unlike the fixed used in the original paper for simplicity
        self.input_pos_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)
        self.output_pos_embedding_table = nn.Embedding(BLOCK_SIZE, EMBEDDING_DIM)

        # Stacking Encoder Blocks
        self.encoder_blocks = nn.ModuleList(EncoderTransformerBlock(EMBEDDING_DIM, NUM_HEADS) for _ in range(NUM_TRANSFORMER_BLOCKS))

        # Stacking Decoder Blocks
        self.decoder_blocks = nn.ModuleList(DecoderTransformerBlock(EMBEDDING_DIM, NUM_HEADS) for _ in range(NUM_TRANSFORMER_BLOCKS))

        # Final layer norm and linear layer
        self.layer_norm_final = nn.LayerNorm(EMBEDDING_DIM)
        self.linear_model_head = nn.Linear(EMBEDDING_DIM, OUTPUT_VOCAB_SIZE)

    def forward(self, input_idx: Tensor, output_idx: Tensor, need_loss: bool = True) -> (Tensor, Optional[Tensor]):
        B, T = input_idx.shape
        assert (B, T) == output_idx.shape, \
            f"Input shape: ({B, T}) must match output shape: {output_idx.shape} in forward pass"

        # Retrieve input and output token and positional embeddings
        input_token_embs = self.input_token_embedding_table(input_idx)  # (B, T, C)
        input_pos_embs = self.input_pos_embedding_table(torch.arange(T, device=DEVICE))  # (T, C)
        input_embedding = input_token_embs + input_pos_embs  # (B,T,C) + (T,C) broadcast

        output_token_embs = self.output_token_embedding_table(output_idx)  # (B, T, C)
        output_pos_embs = self.output_pos_embedding_table(torch.arange(T, device=DEVICE))  # (T, C)
        output_embedding = output_token_embs + output_pos_embs  # (B,T,C) + (T,C) broadcast

        # Produce the pad masks
        input_pad_mask = input_idx != INPUT_PAD_CHAR
        output_pad_mask = output_idx != OUTPUT_PAD_CHAR

        # Encoder stack (B, T, C) -> (B, T, C)
        for encoder_block in self.encoder_blocks:
            input_embedding = encoder_block(input_embedding, input_pad_mask)

        # Decoder stack (B, T, C) -> (B, T, C)
        for decoder_block in self.decoder_blocks:
            output_embedding = decoder_block(input_embedding, output_embedding, input_pad_mask, output_pad_mask)

        # A final layer norm and linear layer to project to vocabulary space
        logits = self.linear_model_head(self.layer_norm_final(output_embedding))  # (B, T, OUTPUT_VOCAB_SIZE)

        if not need_loss:
            loss = None
        else:
            B, T, logits_vocab_size = logits.shape
            assert logits_vocab_size == OUTPUT_VOCAB_SIZE
            # Pytorch cross entropy loss requires the probability distribution to be the second dim
            logits_rearranged = logits.permute(0, 2, 1)  # (B, T, OUTPUT_VOCAB_SIZE) => (B, OUTPUT_VOCAB_SIZE, T)
            # Shift the output to calculate the loss in order to avoid revealing the true answer to the model
            # NOTE: If you don't do the shift below, the model will achieve virtually 0 loss very quickly (from experience)
            output_idx_shifted = shift_output_idx_for_loss(output_idx)

            # Calculate loss as cross entropy and ignoring all pad chars in the output
            loss = F.cross_entropy(logits_rearranged, output_idx_shifted, ignore_index=OUTPUT_PAD_CHAR.item())  # (B, OUTPUT_VOCAB_SIZE, T) and (B, T) cross entropy

        return logits, loss

    def generate(self, input_str: str, generation_mode: Literal['greedy', 'stochastic'] = 'greedy') -> str:

        if not MIN_SEQUENCE_LEN <= len(input_str) <= BLOCK_SIZE - 2:
            print("An unsupported input string length was passed in", file=sys.stderr)
            return "UNTRANSLATED"

        output_str = ''
        while (len(output_str) == 0 or output_str[-1] != END_CHAR) and len(output_str) <= BLOCK_SIZE - 2:
            input_idx, output_idx = data_to_tensor_with_special_chars([input_str], [output_str], is_generation=True)  # (1, T)
            assert input_idx.shape == output_idx.shape
            x = input_idx.to(DEVICE)
            y = output_idx.to(DEVICE)
            model_logits, _ = self(x, y)  # (1, T, C)

            # Pluck the first time step after output_str (including the start character)
            logits = model_logits[:, len(output_str), :]
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Generate the next token from the probability distribution
            match generation_mode:
                case 'greedy':
                    output_next_idx = torch.argmax(probs, dim=-1)  # (1, 1)
                case 'stochastic':
                    output_next_idx = torch.multinomial(probs, num_samples=1)  # (1, 1)
                case _:
                    print(f"Unsupported generation_mode {generation_mode} provided", file=sys.stderr)
                    return "UNTRANSLATED"

            # Append sampled token into the running sequence
            output_str += OUTPUT_DECODER(output_next_idx.tolist())

        return output_str


model = SeqToSeqTransformerModel()
model = model.to(DEVICE)
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


@torch.no_grad()
def calculate_loss(*splits: Literal['train', 'validation', 'test']) -> Dict[Literal['train', 'validate', 'test'], Tensor]:
    out = {}
    model.eval()
    for split in splits:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            xb, yb = get_batch(split)
            _, loss_node_i = model(xb, yb)
            losses[k] = loss_node_i.item()
        out[split] = losses.mean()
    model.train()
    return out


def estimate_loss() -> Dict[Literal['train', 'validation'], Tensor]:
    return calculate_loss('train', 'validation')


def test_loss() -> Dict[str, Tensor]:
    return calculate_loss('test')


# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Load checkpoint if available
if os.path.isfile(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    print(f"Loaded checkpoint from {CHECKPOINT_PATH} at iteration {iteration}")
else:
    iteration = 0
    print(f"No checkpoint found at {CHECKPOINT_PATH}")

# Training Loop
model.train()
try:
    for iteration in range(iteration, MAX_ITERS):
        
        # Every once in a while evaluate the loss on train and val sets
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            est_losses = estimate_loss()
            print(f"step {iteration}: train loss { est_losses['train']:.4f}, val loss {est_losses['validation']:.4f}")

            # Save checkpoint on evaluation
            print("Saving model checkpoint...")
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, CHECKPOINT_PATH)

        # Sample a batch of data
        x_train, y_train = get_batch('train')

        # Evaluate the loss
        output_logits, loss_node = model(x_train, y_train)

        # Backpropagate
        optimizer.zero_grad(set_to_none=True)
        loss_node.backward()
        optimizer.step()

except KeyboardInterrupt:
    print(f"Training was manually killed at iteration: {iteration}")
    pass


# Calculate test loss after training
test_losses = test_loss()
print(f"after training: test loss {test_losses['test']:.4f}")

# Translate using the model after training
for test_input_str, test_output_str in zip(test_dataset_input[:5], test_dataset_output[:5]):
    if not MIN_SEQUENCE_LEN <= len(test_input_str) <= BLOCK_SIZE - 1:
        continue
    sampled_output_str = model.generate(test_input_str)
    print(f"Sentence: {test_input_str}\n"
          f"Data Output: {test_output_str}\n"
          f"Model Output: {sampled_output_str}({len(sampled_output_str)})")
