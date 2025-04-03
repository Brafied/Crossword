from hyperparameters import MAX_CLUE_LENGTH
import torch

def collate_fn(batch, current_epoch, clue_tokenizer, answer_tokenizer):
    clues = [item['clue'] for item in batch]
    answers = [item['answer'] for item in batch]
    clue_tokens = clue_tokenizer(
        clues,
        padding='max_length',
        max_length=MAX_CLUE_LENGTH,
        truncation=True,
        return_tensors='pt'
    )
    answer_tokens = answer_tokenizer.tokenize_and_mask(
        answers=answers, 
        current_epoch=current_epoch
    )

    return {
        'clue_input_ids': clue_tokens['input_ids'],
        'clue_attention_mask': clue_tokens['attention_mask'],
        'answer_input_ids': answer_tokens['masked_tokenized_answers'],
        'answer_attention_mask': answer_tokens['tokenized_answers_attention_mask'],
        'tokenized_answers': answer_tokens['tokenized_answers'],
        'masked_tokenized_answers_mask': answer_tokens['masked_tokenized_answers_mask']
    }


class DynamicDataLoader:
    def __init__(self, current_epoch, dataset, clue_tokenizer, answer_tokenizer, batch_size, shuffle):
        self.current_epoch = current_epoch
        self.dataset = dataset
        self.clue_tokenizer = clue_tokenizer
        self.answer_tokenizer = answer_tokenizer
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.dynamic_collate
        )

    def set_epoch(self, epoch):
        self.current_epoch = epoch  

    def dynamic_collate(self, batch):
        return collate_fn(
            batch=batch,
            current_epoch=self.current_epoch,
            clue_tokenizer=self.clue_tokenizer,
            answer_tokenizer=self.answer_tokenizer
        )

    def __iter__(self):
        return iter(self.dataloader) 

    def __len__(self):
        return len(self.dataloader)
