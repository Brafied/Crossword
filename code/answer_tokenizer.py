import string
import torch
from hyperparameters import (
    MAX_ANSWER_LENGTH,
    MASK_DECAY_RATE
)
import random

class AnswerTokenizer:
    def __init__(self):
        # Define character-to-token and token-to-character mappings
        characters = string.ascii_uppercase
        self.character_to_token = {character: token for token, character in enumerate(characters)}
        self.token_to_character = {token: character for token, character in enumerate(characters)}
        self.mask_token = 26
        self.pad_token = 27

    def tokenize_and_mask(self, answers, current_epoch):
        # Tokenize answers
        tokenized_answers = [
            torch.tensor([self.character_to_token[character] for character in answer])
            for answer in answers
        ]
        # Pad tokenized answers
        tokenized_answers = torch.stack(
            [torch.nn.functional.pad(tokenized_answer, (0, MAX_ANSWER_LENGTH - len(tokenized_answer)), value=self.pad_token) for tokenized_answer in tokenized_answers]
        )
        # Create attention mask
        tokenized_answers_attention_mask = (tokenized_answers == self.pad_token)
        # Mask random tokens
        masked_tokenized_answers = tokenized_answers.clone()
        for i in range(len(masked_tokenized_answers)):
            mask_count = random.randint(len(answers[i]) - min((current_epoch // MASK_DECAY_RATE), len(answers[i]) - 1), len(answers[i]))
            mask_indices = random.sample(range(len(answers[i])), mask_count)
            masked_tokenized_answers[i][mask_indices] = self.mask_token 
        masked_tokenized_answers_mask = (masked_tokenized_answers == self.mask_token)

        return {'tokenized_answers': tokenized_answers,
                'tokenized_answers_attention_mask': tokenized_answers_attention_mask,
                'masked_tokenized_answers': masked_tokenized_answers,
                'masked_tokenized_answers_mask': masked_tokenized_answers_mask
        }

    def detokenize(self, tokens):
        # Convert tokens back to characters
        tokens = tokens.tolist()
        return ''.join([self.token_to_character[token] for token in tokens])
