from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import re
import json
import numpy as np
import random
import argparse
import torch
import re
import ast
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

parser = argparse.ArgumentParser()

parser.add_argument('--context_length', type=int, default = 512)
parser.add_argument('--model', type=str, default = "meta-llama/Llama-2-7b-chat-hf") 
parser.add_argument('--reinforced_model', type=str, default = "/scratch/mk8475/Newnew/temp")
parser.add_argument('--dict_file', type=str, default = "/scratch/mk8475/Newnew/dicts_new.npy")
parser.add_argument('--input_file', type=str, default = "/scratch/mk8475/Newnew/raw_data.txt")
parser.add_argument('--output_file', type=str, default = "/scratch/mk8475/Newnew/generic_predictions.hf")
parser.add_argument('--bootstrap_coef', type=float, default = 5)
parser.add_argument('--device', type=str, default = "cuda:0")

args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir = '/scratch/mk8475/cache')


def get_tokenizer_variations(string):
    return [tokenizer.encode(string)[1:], tokenizer.encode("\n" + string)[3:]]

def prepare_dict(filename):
    def parse_dict(s):
        s = s.replace("\n", "")
        # Using regular expressions to find the dictionary in the string
        match = re.search(r'translations\s*=\s*({.*?})', s)

        if match:
            dict_str = match.group(1)
            try:
                dict_str = re.sub(r',\s*([}\]])', r'\1', dict_str)
                dict_str = re.sub(r'#.*?(,|})', r'\1', dict_str)
                my_dict = json.loads(dict_str) 

                if my_dict is None:
                    my_dict = {}

                return my_dict
            
            except:                
                print(f"Couldn't parse the string: {dict_str}")
                return {}
        else:
            return {}

    def consolidate_dicts(dict_list):
        consolidated = {}
        
        for d in dict_list:
            for key, value in d.items():
                if key not in consolidated:
                    consolidated[key] = []
                if value not in consolidated[key]:  # ensures unique values
                    consolidated[key].append(value)
                    
        return consolidated

    dicts = np.load(filename)
    dicts = [parse_dict(dict) for dict in dicts]
    consolidated_dict = consolidate_dicts(dicts)

    def splittable_key(dict, key):   
     
        if key[-2:] == "'s" and key[:-2] in dict.keys():
            return True

        words = key.split()
        if len(words) == 1:
            return False
        
        return all([word in dict.keys() for word in words])

    consolidated_dict = {k: v for k, v in consolidated_dict.items() if not splittable_key(consolidated_dict, k)}

    print("Total number of entries in anchor expressions dictionary: ", len(consolidated_dict))
    return consolidated_dict

def tokenize_and_index_dict(input_dict):
    
    def add_tokenized_entries(key, value, target_dict):
        key = key.strip()
        value = [item.strip() for item in value]

        # Get all possible variations for each key
        key_tok_variations = get_tokenizer_variations(key)
        val_tok_variations = [[] for _ in key_tok_variations]
        
        for item in value:
            for i, variation in enumerate(get_tokenizer_variations(item)):
                val_tok_variations[i].append(variation)

        for key_tok, value_tok in zip(key_tok_variations, val_tok_variations):
            if key_tok[0] not in target_dict:
                target_dict[key_tok[0]] = {}

            target_dict[key_tok[0]][tuple(key_tok)] = value_tok

    tokenized_dict = {}

    for key, val in input_dict.items():
        add_tokenized_entries(key, val, tokenized_dict)   
    
    return tokenized_dict

# Prepare the anchored expressions dictionary
anchored_expressions_dictionary = tokenize_and_index_dict(prepare_dict(args.dict_file))

# Randomize a value for each given key
def get_trans_dict():
    return {key: {inner_key: random.choice(inner_value) for inner_key, inner_value in value.items()} for key, value in anchored_expressions_dictionary.items()}

model = AutoModelForCausalLM.from_pretrained(args.model,cache_dir = '/scratch/mk8475/cache').to(args.device)    
reinforced_model = AutoModelForCausalLM.from_pretrained(args.reinforced_model, cache_dir = '/scratch/mk8475/Newnew/temp').to(args.device)  

def translate_and_map(original_tokens):    
    
    translated_tokens = []
    mapping = []

    orig_idx = 0 # Current index of token in original sequence
    trans_idx = 0 # Current index of token in translated sequence
    previously_matched = [] # Keep a track of keys that were previously match, to prevent inconsistencies
    forbidden_list = []

    # This function generates a random translation for each key and returns the dictionary that we'll use
    trans_dict = get_trans_dict()

    while orig_idx < len(original_tokens):
        matched = False

        curr_token = original_tokens[orig_idx].item()

        if curr_token in trans_dict:

            # Try to find a match for each tokenized key in the dictionary.
            for key_tokens, value_tokens in trans_dict[curr_token].items():
                length_key = len(key_tokens)

                # If a match is found in the sequence of tokens.                
                if orig_idx + length_key < len(original_tokens) + 1 and key_tokens == tuple(original_tokens[orig_idx: orig_idx + length_key].tolist()):

                    # Add translation of the key found
                    translated_tokens.extend(value_tokens)

                    if tokenizer.decode(key_tokens) in previously_matched:
                        mapping[-1] = -1

                    mapping.extend([-1] * length_key)

                    # Add the value token and all its variations to the list of tokens that we don't want to amplify (again, to avoid inconsistencies)
                    forbidden_list.append([item[0] for item in get_tokenizer_variations(tokenizer.decode(value_tokens))])
                    forbidden_list.extend([[] for _ in range(len(value_tokens) - 1)])

                    # Advance the indices that keep track of source and target locations
                    orig_idx += length_key
                    trans_idx += len(value_tokens)

                    # The last token should already be integrated into the loss as its output is predicting
                    # the token that comes *after* the anchor
                    mapping[-1] = trans_idx - 1

                    # Keep track of replaced keys, so that next time this key is found, we'll indicate
                    # not to integrate it into the loss
                    previously_matched.append(tokenizer.decode(key_tokens))
                    matched = True
                    break

        # If the current token doesn't match any tokenized key, move on to the next token.
        if not matched:
            translated_tokens.append(original_tokens[orig_idx].item())
            mapping.append(trans_idx)
            forbidden_list.append([])
            orig_idx += 1
            trans_idx += 1
        
    return torch.tensor(translated_tokens), torch.tensor(mapping), forbidden_list


def process_chunk(example):    

    IGNORE_TOKEN_ID = -100
    
    original_tokens = torch.tensor(example["tokens"])

    translated_tokens, mapping, forbidden_predictions = translate_and_map(original_tokens)
    mapping = mapping.to(args.device)
    original_tokens = original_tokens.to(args.device)
    translated_tokens = translated_tokens.int().to(args.device)

    mask = (mapping != -1)
    true_indices = mask.nonzero(as_tuple=True)[0]
    
    with torch.no_grad():
        predictions_on_translated = model.forward(translated_tokens.unsqueeze(0)).logits[0] 
        all_forbidden_predictions = [sum(forbidden_predictions[:i], []) for i in range(translated_tokens.shape[0])]

        for i, tokens in enumerate(all_forbidden_predictions):
            predictions_on_translated[i, torch.tensor(tokens).long()] = predictions_on_translated[i].mean()
        
        generic_predictions = predictions_on_translated[mapping[mask]]
        reinforced_predictions = reinforced_model.forward(original_tokens.unsqueeze(0).to(reinforced_model.device)).logits[0][mask]

        offset_predictions = reinforced_predictions - generic_predictions
        offset_predictions = torch.clamp(offset_predictions, min=0)

        return_dict = {"input_ids" : original_tokens.tolist()}
      
        for coef_factor in [0.5, 1, 2]:            
            
            final_labels_on_masked_tokens = (generic_predictions - coef_factor * args.bootstrap_coef * offset_predictions).argmax(dim=1)

            final_predictions = torch.full_like(original_tokens, IGNORE_TOKEN_ID)
            final_predictions[true_indices] = final_labels_on_masked_tokens
            
            final_predictions = [IGNORE_TOKEN_ID] + torch.tensor(final_predictions).tolist()[:-1]

            if coef_factor == 1:
                return_dict["labels"] = final_predictions
            else:
                return_dict[f"labels_{coef_factor}"] = final_predictions

    return return_dict

CHUNK_SIZE = 10000 

def read_file_into_chunks(filename, context_length):
    tokenized_text = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(CHUNK_SIZE)
            
            if not chunk:
                break
                
            chunk_tokens = tokenizer.encode(chunk, return_tensors="pt")[0]
            tokenized_text.extend(chunk_tokens)
    
    # Break tokenized content into blocks of size context_length
    return [tokenized_text[i:i+context_length] for i in range(0, len(tokenized_text) - context_length, context_length)]

# Read the input text file, split it into chunks of size context_length and create a dataset from it
chunks = read_file_into_chunks(args.input_file, args.context_length)
dataset = Dataset.from_dict({'tokens': chunks})

# Main processing of dataset
processed_dataset = dataset.map(process_chunk)

# Save processed dataset
processed_dataset.save_to_disk(args.output_file)