

#Code by Jason Chen for the Qualitative Analysis
from transformers import RobertaForMaskedLM, RobertaTokenizer
model = RobertaForMaskedLM.from_pretrained('roberta-base', return_dict = True)
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
tokenizer.add_tokens(['[PAD]'])
model.resize_token_embeddings(len(tokenizer))
# state_dict = torch.load('roberta_final_model_ep3_commononly', map_location=torch.device('cpu'))
# model.load_state_dict(state_dict)

from torch.nn.functional import cosine_similarity

embedding_layer_train = model.get_input_embeddings()
embeddings_train = embedding_layer_train.weight.detach()

targets = ['Ġtool', 'Ġlit', 'Ġcat', 'Ġschool', 'Ġscene', 'Ġhot', 'og', 'bb', 'Ġ420', 'Ġalpha', 'Ġstraight', 'Ġk', 'Ġmetal', 'Ġdm', 'Ġvine']
vocab = tokenizer.vocab
# for target in targets:
target = 'awesome'
if target in vocab:
    target_index = vocab[target]
else:
    print('fob')
    print(target)

print(target_index)
target_embedding = embeddings_train[target_index]

# Compute cosine similarity
similarities = cosine_similarity(target_embedding.unsqueeze(0), embeddings_train)

n = 50
top_n_indices = torch.topk(similarities, n).indices

# Print the top-n closest tokens
for index in top_n_indices:
    token = tokenizer.convert_ids_to_tokens(index.item())
    print(f"Token: {token}, Similarity: {similarities[index]}")

# vocab = tokenizer.vocab

embedding_layer = model.get_input_embeddings()
embeddings = embedding_layer.weight.detach()

state_dict = torch.load('roberta_final_model_ep', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

embedding_layer_train = model.get_input_embeddings()
embeddings_train = embedding_layer_train.weight.detach()

targets = ['Ġtool', 'Ġlit', 'Ġcat', 'Ġschool', 'Ġscene', 'Ġhot', 'og', 'bb', 'Ġ420', 'Ġalpha', 'Ġstraight', 'Ġk', 'Ġmetal', 'Ġdm', 'Ġvine']
vocab = tokenizer.get_vocab()
for target in targets:
    if target in vocab:
        target_index = vocab[target]
    else:
        print('not inside')
        print(target)
        continue

    print(target_index)
    target_embedding = embeddings[target_index]
    target_embedding_train = embeddings_train[target_index]

    print(target_embedding.size())
    print(target_embedding.unsqueeze(0).size())

    # Compute cosine similarity
    similarities = cosine_similarity(target_embedding.unsqueeze(0), embeddings)
    similarities_train = cosine_similarity(target_embedding_train.unsqueeze(0), embeddings_train)

    n = 25
    top_n_indices = torch.topk(similarities, n).indices
    top_n_indices_train = torch.topk(similarities_train, n).indices

    # Print the top-n closest tokens
    tokens1 = set()
    tokens2 = set()
    for index in top_n_indices:
        token = tokenizer.convert_ids_to_tokens(index.item())
        tokens1.add(token)
    #     print(f"Token: {token}, Similarity: {similarities[index]}")
    for index in top_n_indices_train:
        token = tokenizer.convert_ids_to_tokens(index.item())
        tokens2.add(token)
    print(tokens1 - tokens2)

# vocab = tokenizer.vocab