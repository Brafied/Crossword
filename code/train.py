from datasets import load_from_disk
from hyperparameters import (
    ENCODER,
    CLUES_SPLIT,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    EVALUATION_INTERVAL,
    RUN
)
from transformers import AutoTokenizer
from answer_tokenizer import AnswerTokenizer
from dynamic_data_loader import DynamicDataLoader
from model import Crossword
import torch
import os
from tqdm import tqdm
import json

def process_batch(batch, model, criterion):
    clue_tokens = {
        'input_ids': batch['clue_input_ids'],
        'attention_mask': batch['clue_attention_mask']
    }
    masked_answer_tokens = {
        'input_ids': batch['answer_input_ids'],
        'attention_mask':  batch['answer_attention_mask']
    }
    logits = model(clue_tokens, masked_answer_tokens)
    logits = logits.permute(0, 2, 1).contiguous().view(-1, 26)
    labels = batch['tokenized_answers']

    masked_tokens = batch['masked_tokenized_answers_mask']
    filtered_logits = logits[masked_tokens.view(-1)]
    filtered_labels = labels.masked_select(masked_tokens).view(-1)

    loss = criterion(filtered_logits, filtered_labels)

    return loss

def save_checkpoint(model, optimizer, train_losses, validation_losses, current_epoch):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'validation_losses': validation_losses,
        'current_epoch': current_epoch
    }
    filename = f"checkpoints/{RUN}_{current_epoch}.pth"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(checkpoint, filename)
    print("Checkpoint saved")

def evaluate(validation_dataloader, model, criterion):
    validation_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in validation_dataloader:
            loss = process_batch(
                batch=batch, 
                model=model, 
                criterion=criterion
            )
            validation_loss += loss.item()
    return validation_loss / len(validation_dataloader)

def train(train_dataloader, validation_dataloader, model, optimizer, current_epoch, train_losses, validation_losses, criterion):
    model.train()

    for epoch in range(current_epoch, EPOCHS):
        train_dataloader.set_epoch(epoch)
        validation_dataloader.set_epoch(epoch)

        for batch_number, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit='batch')):
            optimizer.zero_grad()
            loss = process_batch(
                batch=batch,
                model=model,
                criterion=criterion
            )
            loss.backward()
            optimizer.step()

            if batch_number == 0 and (epoch % EVALUATION_INTERVAL) == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    train_losses=train_losses, 
                    validation_losses=validation_losses,
                    current_epoch=epoch
                )   
                train_losses.append(round(loss.item(), 5))
                validation_loss = evaluate(
                    validation_dataloader=validation_dataloader,
                    model=model, 
                    criterion=criterion
                )
                validation_losses.append(round(validation_loss, 5))
                with open("loss/training_loss_log.json", 'w') as f:
                    json.dump(train_losses, f)
                with open("loss/validation_loss_log.json", 'w') as f:
                    json.dump(validation_losses, f)
                print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_number + 1}, Train Loss: {loss.item()}, Validation Loss: {validation_loss}")

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint found")
        return checkpoint['current_epoch'], checkpoint['train_losses'], checkpoint['validation_losses']
    else:
        print("No checkpoint found")
        return 0, [], []
    
def main():
    model = Crossword()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=LEARNING_RATE
    )
    current_epoch, train_losses, validation_losses = load_checkpoint(
        model=model,
        optimizer=optimizer,
        filename='checkpoint.pth'
    )

    clues_split = load_from_disk(CLUES_SPLIT)
    clue_tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    answer_tokenizer = AnswerTokenizer()
    train_dataloader = DynamicDataLoader(
        current_epoch=current_epoch,
        dataset=clues_split['train'],
        clue_tokenizer=clue_tokenizer,
        answer_tokenizer=answer_tokenizer,
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    validation_dataloader = DynamicDataLoader(
        current_epoch=current_epoch,
        dataset=clues_split['validation'], 
        clue_tokenizer=clue_tokenizer,
        answer_tokenizer=answer_tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=answer_tokenizer.pad_token
    )

    train(
        model=model,
        optimizer=optimizer,
        current_epoch=current_epoch,
        train_losses=train_losses,
        validation_losses=validation_losses,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        criterion=criterion
    )

if __name__ == '__main__':
    main()
