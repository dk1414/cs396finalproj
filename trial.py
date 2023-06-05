# %%
import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import json

import concurrent.futures

# %%
#get lyrics data
data = pd.read_csv("final_data.csv")

#trim data, it was taking too long to do all of them
rows_to_drop = data.sample(n=1000, random_state=42).index

data = data.drop(rows_to_drop)
data

# %%
#load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#load bert model and put in eval mode
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
model.eval()

# %%
def create_embeddings(start_index, end_index, df):

        #this will be a dictionary mapping an artist to that artist's respective embedding and genre, (embedding,genre), the contents of this dict will later be saved to a json file
    artist_embedding_dict = dict()



    for index, row in df.iloc[start_index:end_index].iterrows():

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
        artist_embedding_dict[artist] = artist_embedding.tolist()

    return artist_embedding_dict


# %%
short_df = data[:8]
chunk_size = 2

results = []


def process_dataframe(df, num_threads):
    results = []
    chunk_size = len(df) // num_threads
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        start_index = 0
        for i in range(num_threads):
            end_index = start_index + chunk_size if i < num_threads - 1 else None
            future = executor.submit(create_embeddings, start_index, end_index, df)
            futures.append(future)
            start_index = end_index
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.extend(result)
    return results



# Set the number of threads
num_threads = 4

# Process the dataframe using 4 threads
processed_data = process_dataframe(short_df, num_threads)

print(processed_data)

for l in processed_data:
    print(type(l))


# %%


