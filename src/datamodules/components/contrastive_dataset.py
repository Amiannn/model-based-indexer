import os
import json
import torch
import numpy as np

from torch.utils.data import Dataset
from transformers     import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    # in-batch negative
    x_row = []
    x_col = []
    y     = []
    batch_size    = len(dataset)
    negative_mask = torch.eye(batch_size)

    for i in range(batch_size):
        text  = dataset[i]['text']
        label = dataset[i]['label']
        
        positive_len = len(dataset[i]['positive_ctxs'])
        pos_text     = text
        if positive_len > 0:
            pos_index = np.random.randint(0, positive_len)
            pos_text  = dataset[i]['positive_ctxs'][pos_index]['text']
            pos_label = dataset[i]['positive_ctxs'][pos_index]['label']
        x_row.append(text)
        x_col.append(pos_text)
        y.append(label)
    
    neg_mask_col = np.array(y * batch_size).reshape(batch_size, -1)
    neg_mask_row = neg_mask_col.T
    negative_mask= torch.tensor(neg_mask_col == neg_mask_row).float()

    x_row_idx = tokenizer(
        text=x_row,
        max_length       = args.max_seq_length,
        padding          = "max_length",
        return_tensors   = "pt",
        truncation       = True,
    )

    x_col_idx = tokenizer(
        text=x_col,
        max_length       = args.max_seq_length,
        padding          = "max_length",
        return_tensors   = "pt",
        truncation       = True,
    )

    y_idx = tokenizer(
        text=y,
        max_length       = args.max_length,
        padding          = "max_length",
        return_tensors   = "pt",
        truncation       = True,
    )

    return {
        'x_row'   : x_row_idx,
        'x_col'   : x_col_idx,
        'y'       : y_idx,
        'neg_mask': negative_mask,
    }

class ContrastiveDataset(Dataset):
    def __init__(self, datas_path):
        with open(datas_path, 'r', encoding='utf-8') as f:
            self.datas = json.load(f)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]

if __name__ == '__main__':
    dataset = [
        {
            "text": "document: [Pennie Lane Trumbull] Pennie Lane Trumbull Pennie Ann Trumbull (born July 3, 1954), known as Pennie Lane, is an American socialite, philanthropist, businesswoman, and entrepreneur. During the 1970s she formed the group \"The Flying Garter",
            "label": "43778",
            "positive_ctxs": [
                
            ],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        },
        {
            "text": "question: n. the political assembly of citizens of an ancient greek state",
            "label": "12263",
            "positive_ctxs": [
                {
                    "text": "document: [Cleisthenes] Cleisthenes Cleisthenes (; , \"Kleisth\u00e9n\u0113s\"; also Clisthenes or Kleisthenes) was an ancient Athenian lawgiver credited with reforming the constitution of ancient Athens and setting it on a democratic footing in 508 BC.",
                    "label": "12263"
                }
            ],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        },
        {
            "text": "document: [World population] World population In demographics, the world population is the total number of humans currently living, and was estimated to have reached 7.7 billion people as of November 2018. It took over 200,000",
            "label": "4457",
            "positive_ctxs": [
                {
                    "text": "question: when did the world population reached 8 billion",
                    "label": "4457"
                }
            ],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        },
        {
            "text": "document: [Pennie Lane Trumbull] Pennie Lane Trumbull Pennie Ann Trumbull (born July 3, 1954), known as Pennie Lane, is an American socialite, philanthropist, businesswoman, and entrepreneur. During the 1970s she formed the group \"The Flying Garter",
            "label": "43778",
            "positive_ctxs": [
                {
                    "text": "question: who was penny lane based on in almost famous",
                    "label": "43778"
                }
            ],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        },
        {
            "text": "question: where was the first commercial radio station to broadcast located in the united states",
            "label": "22602",
            "positive_ctxs": [
                {
                    "text": "document: [KDKA (AM)] KDKA (AM) KDKA (1020 kHz AM) is a Class A (clear channel) radio station, owned and operated by Entercom and licensed to Pittsburgh, Pennsylvania. Its studios are located at the combined Entercom",
                    "label": "22602"
                }
            ],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        },
        {
            "text": "document: [Legal drinking age] Legal drinking age The legal drinking age is the age at which a person can legally consume alcoholic beverages. These laws cover a wide range of issues and behaviors, addressing when and",
            "label": "44842",
            "positive_ctxs": [
                {
                    "text": "question: what is the drinking age in dominican republic",
                    "label": "44842"
                }
            ],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        },
        {
            "text": "question: what is the dog in i am legend",
            "label": "59876",
            "positive_ctxs": [
                {
                    "text": "document: [I Am Legend (film)] I Am Legend (film) I Am Legend is a 2007 American post-apocalyptic science fiction drama film based on the novel of the same name, directed by Francis Lawrence and starring Will Smith,",
                    "label": "59876"
                }
            ],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        },
        {
            "text": "question: two ways to conserve water electricity and soil",
            "label": "16705",
            "positive_ctxs": [
                {
                    "text": "document: [Water conservation] Water conservation Water conservation includes all the policies, strategies and activities to sustainably manage the natural resource of fresh water, to protect the hydrosphere, and to meet the current and future human",
                    "label": "16705"
                }
            ],
            "negative_ctxs": [],
            "hard_negative_ctxs": []
        },
    ]

    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    class args:
        max_seq_length= 100
        max_length=30

    datas = DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=1,
        collate_fn=lambda examples: preprocess_batch_for_hf_dataset(examples, tokenizer, args),
        pin_memory=True,
        shuffle=True,
    )

    for data in datas:
        for key in data:
            print(f'{key}: {data[key]}')
        print()
    
