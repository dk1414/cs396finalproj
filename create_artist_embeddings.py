import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import json


#get lyrics data
df = pd.read_csv("final_data.csv")

#load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#load bert model and put in eval mode
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()

#this will be a dictionary mapping an artist to that artist's respective embedding and genre, (embedding,genre), the contents of this dict will later be saved to a json file
artist_embedding_dict = dict()


def get_embeddings(row):

    #get info from row
    artist = row["artist"]
    genre = row["genre"]
    lyrics = row["lyrics"]

    #convert lyrics to list of tokens
    list_of_tokens = tokenizer.tokenize(lyrics)

    #split the entire list of tokens into lists of 510 tokens, as that will be the max number of tokens we can make an
    #embedding for after adding the special tokens

    lyrics_split_list = []

    temp_list = []
    for i,token in enumerate(list_of_tokens):
        
        if i % 510 == 0 and i != 0:
            lyrics_split_list.append(temp_list)
            temp_list = []
        
        temp_list.append(token)

    lyrics_split_list.append(temp_list)


    #for each list of tokens, add special tokens to front and back
    # and then map tokens to their vocabulary indices for each list of tokens
    # and make segment ids, this will just be lists of all 1's 
    indexed_tokens_list = []
    segment_ids = []

    for l in lyrics_split_list:
        l.insert(0,"[CLS]")
        l.append("[SEP]")
        segment_ids.append([1] * len(l))
        indexed_tokens_list.append(tokenizer.convert_tokens_to_ids(l))
    
    #convert to tensors
    tokens_tensor_list = []
    segment_tensor_list = []

    for i, l in enumerate(indexed_tokens_list):
        tokens_tensor_list.append(torch.tensor([l]))
        segment_tensor_list.append(torch.tensor([segment_ids[i]]))

    #run tensors through bert and get the hidden states
    with torch.no_grad():

        hidden_state_list = []

        for i, l in enumerate(tokens_tensor_list):
            output = model(l, segment_tensor_list[i])
            hidden_state_list.append(output[2])
    
    embeddings = []

    for state in hidden_state_list:
        token_vectors = state[-2][0]
        embeddings.append(torch.mean(token_vectors,dim=0))

    artist_embedding = torch.mean(torch.stack(embeddings), dim=0)

    #make it a python list so it is json serializable
    artist_embedding_dict[artist] = (artist_embedding.tolist(),genre)


rows = df[:5]

rows.apply(get_embeddings, axis=1)


with open("embeddings.json", "w+") as file:
    json.dump(artist_embedding_dict,file)