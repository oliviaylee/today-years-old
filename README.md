# Today Years Old
Repository for CS 224N Final Project "Today Years Old: Adapting Adapting Language Models to Word Shifts"

Below are the main components of our project and their respective relevant files.

## Approach 1: Averaging Method for Embedding Initialization

## Approach 2: Finetuning Models to Learn Definition-Embedding Mappings
This approach involves finetuning a large language model (GPT-2 or RoBERTa) to learn mappings between word definitions and word embeddings in the model. It is then used to predict the word embedding of a novel word given its definition.
- `gpt2_embed_predict.py`: Model training and prediction of new word embeddings, using (unidirectional) GPT-2 as a base model.
- `roberta_embed_predict.py`: Model training and prediction of new word embeddings, using (biidirectional) RoBERTa as a base model.
- `dataset.py`: Processing of JSon dictionary datasets into a PyTorch dataset that can be used by a dataloader.

## Evaluation Methods

## Datasets
Datasets can be found in the `datasets` folder. In particular,
- `dict_wn.json`: Dictionary of common words (used for training in Approach 2)
- Urban Dictionary data: Used to provide definitions of novel lexical items to be incorporated into the models (used in both Approach 1 and 2.
- `test.py`: Test to count the number of common words in Urban Dictionary data (disguised as "novel").
