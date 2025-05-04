
from hyperparameters import (
    CHECKPOINT_TITLE,
    ENCODER,
    MAX_CLUE_LENGTH,
    MAX_ANSWER_LENGTH
)
from model import Crossword
import torch
from transformers import AutoTokenizer
from answer_tokenizer import AnswerTokenizer

def load_model(filename):
    model = Crossword()
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def prepare_input(clue, partial_answer, clue_tokenizer, answer_tokenizer):
    clue_tokens = clue_tokenizer(
        [clue],
        padding='max_length',
        max_length=MAX_CLUE_LENGTH,
        truncation=True,
        return_tensors='pt'
    )

    masked_tokenized_answer = []
    for char in partial_answer.upper():
        if char in answer_tokenizer.characters:
            masked_tokenized_answer.append(answer_tokenizer.character_to_token[char])
        else:
            masked_tokenized_answer.append(answer_tokenizer.mask_token)
    masked_tokenized_answer = torch.tensor(masked_tokenized_answer)
    masked_tokenized_answer = torch.nn.functional.pad(masked_tokenized_answer, (0, MAX_ANSWER_LENGTH - len(masked_tokenized_answer)), value=answer_tokenizer.pad_token)
    tokenized_answer_attention_mask = (masked_tokenized_answer == answer_tokenizer.pad_token)
    masked_tokenized_answer_mask = (masked_tokenized_answer == answer_tokenizer.mask_token)

    return {
        'clue_input_ids': clue_tokens['input_ids'],
        'clue_attention_mask': clue_tokens['attention_mask'],
        'answer_input_ids': masked_tokenized_answer.unsqueeze(0),
        'answer_attention_mask': tokenized_answer_attention_mask.unsqueeze(0),
        'masked_tokenized_answers_mask': masked_tokenized_answer_mask.unsqueeze(0)
    }

def inference(clue, partial_answer):
    model = load_model(CHECKPOINT_TITLE)

    clue_tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    answer_tokenizer = AnswerTokenizer()
    input = prepare_input(
        clue=clue,
        partial_answer=partial_answer,
        clue_tokenizer=clue_tokenizer,
        answer_tokenizer=answer_tokenizer
    )

    with torch.no_grad():
        clue_tokens = {
            'input_ids': input['clue_input_ids'],
            'attention_mask': input['clue_attention_mask']
        }
        masked_answer_tokens = {
            'input_ids': input['answer_input_ids'],
            'attention_mask':  input['answer_attention_mask']
        }
        logits = model(clue_tokens, masked_answer_tokens)

        predicted_answer_tokens = input['answer_input_ids'].squeeze(0)
        predictions = logits.argmax(dim=-1).squeeze(1)
        mask = input['masked_tokenized_answers_mask'].squeeze(0)
        predicted_answer_tokens[mask] = predictions[mask]

        predicted_answer = answer_tokenizer.detokenize(predicted_answer_tokens[:len(partial_answer)])
        print(f"Clue: {clue}")
        print(f"Partial Answer: {partial_answer}")
        print(f"Predicted Answer: {predicted_answer}")

if __name__ == '__main__':
    clue = "Yale."
    partial_answer = "___"
    inference(
        clue=clue,
        partial_answer=partial_answer
    )