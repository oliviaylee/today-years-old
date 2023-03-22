# Today Years Old
Repository for CS 224N Final Project "Today Years Old: Adapting Adapting Language Models to Word Shifts"

Below are the main components of our project and their respective relevant files.

## Model Training: Finetuning Models to Learn Definition-Embedding Mappings (Authored by: Olivia Lee)
This approach involves finetuning a large language model (GPT-2 or RoBERTa) to learn mappings between word definitions and word embeddings in the model. It is then used to predict the word embedding of a novel word given its definition.
- `gpt2_embed_predict.py`: Model training and prediction of new word embeddings, using (unidirectional) GPT-2 as a base model.
- `roberta_embed_predict.py`: Model training and prediction of new word embeddings, using (bidirectional) RoBERTa as a base model.
- `dataset.py`: Preprocessing training datasets into PyTorch dataset / dataloader for model training.

## Evaluation Methods: GPT-2 Causal LM and RoBERTa Masked LM (Authored by: Jason Chen)
The first three scripts are used to evaluate the trained model from above, as well as  baselines (embedding matrix expanded with new embeddings initialized via averaging method, or off-the-shelf GPT-2/RoBERTa models).
- `gpt2_eval.py`: Evaluation using GPT-2 Causal Language Modeling task
- `roberta_eval.py`: Evaluation using RoBERTa Masked Language Modeling task
- `qualitative_analysis.py`: Qualitative analysis of trained model's embedding space
- `winodict_eval.ipynb`: Preprocessing / evaluation of BERT on WinoDict (Authored by: Zachary Xi)

## Datasets
Datasets can be found in the `datasets` folder. In particular,
- `dict_wn.json`: Dictionary of common words (used for training model in our approach)
- Urban Dictionary data: (file too big for upload) Used to provide definitions of novel lexical items to be incorporated into the models (used for model expansion/evaluation).
- `urban_preprocess.py`: Script to preprocess Urban Dictionary data