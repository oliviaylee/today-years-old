import transformers 
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel
import torch
import datasets
import json
import tqdm
from tqdm import tqdm


def retrieve_embeddings():
    #colab mounting
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    pre_expansion_embeddings = model.get_input_embeddings().weight
    print(len(pre_expansion_embeddings))

    # local variable doesn't seem to exist (?)
    tokenizer.add_tokens(words_to_add)
    model.resize_token_embeddings(len(tokenizer))

    embeddings = model.get_input_embeddings().weight #params['embeddings.word_embeddings.weight']
    print(len(embeddings))

    #code to sample embeddings adapted from hewitt's blog (? -- check this portion)
    mu = torch.mean(pre_expansion_embeddings, dim=0)
    n = pre_expansion_embeddings.size()[0]
    sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
            mu, covariance_matrix=1e-5*sigma)

    #sample from distribution
    num_words = 961
    with torch.no_grad():
        new_embeddings = torch.stack(tuple((dist.sample() for _ in range(num_words))), dim=0)
        embeddings[-num_words:,:] = new_embeddings

    #novel approach: retrieving trained model weights
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name, return_dict = True)
    tokenizer.add_tokens(['[PAD]'])
    model.resize_token_embeddings(len(tokenizer))

    #edited URL
    state_dict = torch.load('drive/MyDrive/a2modelscopy/gpt2_final_model_ep3_commononly_5000', map_location=torch.device('cuda'))
    model.load_state_dict(state_dict)

    return embeddings, new_embeddings

# Compute cosine similarities between the target embedding and all other embeddings.

# k indices for highest similarity scores
def compute_embeddings(k):
    embeddings, new_embeddings = retrieve_embeddings()
    similarities = torch.nn.functional.cosine_similarity(embeddings, new_embeddings, dim=1)

    # Select the top K indices with the highest similarity scores.
    #k = 50
    top_k_indices = torch.topk(similarities, k).indices

    # Print out the top K indices.
    print(f"Top {k} indices closest to target embedding:")
    print(top_k_indices)

# calling function to compute embeddings
compute_embeddings(10)