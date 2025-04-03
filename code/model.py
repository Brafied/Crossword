import torch.nn as nn
from transformers import AutoModel
from hyperparameters import (
    MAX_ANSWER_LENGTH,
    ANSWER_VOCABULARY_SIZE,
    ENCODER,
    DECODER_MAX_SEQUENCE_LENGTH,
    DECODER_EMBEDDING_DIMENSIONS,
    DECODER_LAYER_ATTENTION_HEAD_COUNT,
    DECODER_LAYER_FEEDFORWARD_DIMENSIONS,
    DECODER_LAYER_COUNT,
    CLASSES_COUNT,
)
import torch

class Crossword(nn.Module):
        
    def __init__(self):
        super(Crossword, self).__init__()
        
        self.encoder = AutoModel.from_pretrained(ENCODER)

        self.embedding = nn.Embedding(
            num_embeddings=ANSWER_VOCABULARY_SIZE,
            embedding_dim=DECODER_EMBEDDING_DIMENSIONS
        )
        self.positional_embedding = nn.Embedding(
            num_embeddings=DECODER_MAX_SEQUENCE_LENGTH,
            embedding_dim=DECODER_EMBEDDING_DIMENSIONS
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=DECODER_EMBEDDING_DIMENSIONS,
            nhead=DECODER_LAYER_ATTENTION_HEAD_COUNT,
            dim_feedforward=DECODER_LAYER_FEEDFORWARD_DIMENSIONS
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=DECODER_LAYER_COUNT
        )

        self.output_layer = nn.Linear(
            in_features=DECODER_EMBEDDING_DIMENSIONS,
            out_features=CLASSES_COUNT
        )

    def forward(self, clue_tokens, answer_tokens):
        # clue_tokens['input_ids'] shape : (BATCH_SIZE, MAX_CLUE_LENGTH)
        # clue_tokens['attention_mask'] shape : (BATCH_SIZE, MAX_CLUE_LENGTH)
        # answer_tokens['input_ids'] shape : (BATCH_SIZE, MAX_ANSWER_LENGTH)
        # answer_tokens['attention_mask'] shape : (BATCH_SIZE, MAX_ANSWER_LENGTH)

        # clue_encoding shape : (BATCH_SIZE, MAX_CLUE_LENGTH, ENCODER_EMBEDDING_DIMENSIONS)
        clue_encoding = self.encoder(**clue_tokens).last_hidden_state
        # clue_encoding shape : (MAX_CLUE_LENGTH, BATCH_SIZE, ENCODER_EMBEDDING_DIMENSIONS)
        clue_encoding = clue_encoding.permute(1, 0, 2)

        # answer_embedding shape : (BATCH_SIZE, MAX_ANSWER_LENGTH, DECODER_EMBEDDING_DIMENSIONS)
        answer_embedding = self.embedding(answer_tokens['input_ids'])
        # answer_token_indices shape : (BATCH_SIZE, MAX_ANSWER_LENGTH)
        answer_token_indices = torch.arange(MAX_ANSWER_LENGTH).unsqueeze(0).expand(answer_embedding.shape[0], -1)
        # answer_positional_embedding shape : (BATCH_SIZE, MAX_ANSWER_LENGTH, DECODER_EMBEDDING_DIMENSIONS)
        answer_positional_embedding = self.positional_embedding(answer_token_indices)
        answer_embedding = answer_embedding + answer_positional_embedding
        # answer_embedding shape : (MAX_ANSWER_LENGTH, BATCH_SIZE, DECODER_EMBEDDING_DIMENSIONS)
        answer_embedding = answer_embedding.permute(1, 0, 2)
        # decoder_output shape : (MAX_ANSWER_LENGTH, BATCH_SIZE, DECODER_EMBEDDING_DIMENSIONS)
        decoder_output = self.decoder(
            tgt=answer_embedding,
            memory=clue_encoding,
            tgt_key_padding_mask=answer_tokens['attention_mask'],
            memory_key_padding_mask=(clue_tokens['attention_mask'] == 0)
        )

        # logits shape : (MAX_ANSWER_LENGTH, BATCH_SIZE, CLASSES_COUNT)
        logits = self.output_layer(decoder_output)

        return logits
