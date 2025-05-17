import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len  # Maximum sequence length allowed for source/target sequences

        self.ds = ds  # The raw dataset consisting of source-target translation pairs
        self.tokenizer_src = tokenizer_src  # Tokenizer for the source language
        self.tokenizer_tgt = tokenizer_tgt  # Tokenizer for the target language
        self.src_lang = src_lang  # Source language identifier
        self.tgt_lang = tgt_lang  # Target language identifier

        # Define special tokens using the target tokenizer
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)  # Start of sequence
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)  # End of sequence
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)  # Padding token

    def __len__(self):
        return len(self.ds)  # Number of items in the dataset

    def __getitem__(self, idx):
        # Extract a pair of source and target texts
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Convert raw texts to token IDs using the tokenizers
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Compute how much padding is needed to reach seq_len after adding special tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # +2 for [SOS] and [EOS]
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # +1 for [SOS], [EOS] will go to label

        # If either source or target is too long, raise an error
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Create encoder input: [SOS] + token IDs + [EOS] + padding
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create decoder input: [SOS] + token IDs + padding (no EOS here)
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Create label: token IDs + [EOS] + padding (no SOS here)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Sanity check: all sequences must be of length seq_len
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # Token IDs for encoder (seq_len,)
            "decoder_input": decoder_input,  # Token IDs for decoder input (seq_len,)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len) - padding mask for encoder
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len, seq_len) - padding & causal mask for decoder
            "label": label,  # Target token IDs with EOS included (seq_len,)
            "src_text": src_text,  # Raw source sentence
            "tgt_text": tgt_text,  # Raw target sentence
        }

# Function to create causal mask for decoder self-attention
# Ensures that each token can only attend to itself and previous tokens

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0  # Returns boolean mask where True means allowed to attend
