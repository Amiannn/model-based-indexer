# modify from https://github.com/ArvinZhuang/DSI-transformers/blob/main/data/NQ/create_NQ_train_vali.py
import os
import json
import random
import argparse
import datasets
import numpy as np

from tqdm import tqdm
from utils.fileio import write_json

from utils.docid import (
    create_semantic_docid
)

random.seed(17)

def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.0f%s' % (num, ['', 'k', 'm', 'g', 't', 'p'][magnitude])

def process_document(tokens, truncate=-1):
    token_inds = np.where(np.array(tokens['is_html']) == False)[0]
    tokens     = np.array(tokens['token'])[token_inds]
    doc_text   = " ".join(tokens[:truncate])
    return doc_text

def process_train_dataset(keys, dataset_datas):
    dataset = []
    for title in dataset_datas:
        identifier = dataset_datas[title]['identifier']
        document  = {
            'text'      : dataset_datas[title]['document'],
            'prefix'    : 'document',
            'identifier': identifier
        }
        if title in keys:
            question = [{
                'text'      : q,
                'prefix'    : 'question',
                'identifier': identifier
            } for q in dataset_datas[title]['questions']]
            
            datas = [document] + question.copy()
            idx   = list(range(len(datas))) * 2
            for i in range(len(datas)):
                _data = datas[i].copy()
                _data['positive_ctxs'] = [datas[j] for j in idx[i+1:i+len(datas)]]
                dataset.append(_data)
        else:
            question = [document.copy()]
            datas = document.copy()
            datas['positive_ctxs'] = question
            dataset.append(datas)

    return dataset

def process_test_dataset(keys, dataset_datas):
    dataset = []
    for title in keys:
        identifier = dataset_datas[title]['identifier']
        document  = {
            'text'      : dataset_datas[title]['document'],
            'prefix'    : 'document',
            'identifier': identifier 
        }
        for q in dataset_datas[title]['questions']:
            dataset.append({
                'text'      : q,
                'prefix'    : 'question',
                'identifier': identifier,
                'positive_ctxs': [document]
            })
            
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create nature question datasets.")
    parser.add_argument("--document_length", type=int, default=1000, help="Document total length.")
    parser.add_argument("--split_ratio"    , type=float, default=[0.7, 0.1, 0.2], help="Train Dev Test ratio.")
    parser.add_argument("--semantic_model" , type=str, default="sentence-transformers/sentence-t5-base", help="The model use to do the semantic docid generation.")
    parser.add_argument("--output_dir"     , type=str, default='./data', help="Dataset path.")
    args = parser.parse_args()

    datas = datasets.load_dataset(
        'natural_questions', 
        cache_dir='cache', 
        beam_runner='DirectRunner',
        ignore_verifications=True
    )['train']

    idxs  = list(range(len(datas)))
    random.shuffle(idxs)

    processed_datas = {}
    for i, idx in tqdm(enumerate(idxs)):
        data = datas[idx]
        if len(processed_datas.keys()) >= args.document_length:
            break
        
        title    = data['document']['title']
        question = data['question']['text']
        document = process_document(data['document']['tokens'], truncate=100)

        try:
            processed_datas[title]['questions'].append(question)
        except:
            processed_datas[title] = {
                # 'document'  : '[{}] {}'.format(title, document),
                'document'  : document,
                'questions' : [question],
                'identifier': {
                    'atomic': str(i)
                }
            }
    
    documents  = [processed_datas[title]['document'] for title in processed_datas]
    atomic_ids = [processed_datas[title]['identifier']['atomic'] for title in processed_datas]
    atomic2semantic = create_semantic_docid(documents, atomic_ids, model_type='sentence-transformers/sentence-t5-base')

    for title in processed_datas:
        atomic_id   = processed_datas[title]['identifier']['atomic']
        semantic_id = atomic2semantic[atomic_id]
        processed_datas[title]['identifier']['semantic'] = semantic_id

    idx_keys = list(processed_datas.keys())
    random.shuffle(idx_keys)

    train_lenght = int(len(idx_keys) * args.split_ratio[0])
    dev_lenght   = int(len(idx_keys) * args.split_ratio[1])
    test_lenght  = len(idx_keys) - train_lenght - dev_lenght

    train_keys = idx_keys[:train_lenght] 
    dev_keys   = idx_keys[train_lenght:train_lenght + dev_lenght] 
    test_keys  = idx_keys[train_lenght + dev_lenght:]


    # create dataset
    train_datas = process_train_dataset(train_keys, processed_datas)
    dev_datas   = process_test_dataset(dev_keys, processed_datas)
    test_datas  = process_test_dataset(test_keys, processed_datas)

    # create data folder
    output_dataset_dir = os.path.join(args.output_dir, 'nq_{}'.format(human_format(args.document_length)))
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir, exist_ok=True)

    output_path = os.path.join(output_dataset_dir, 'nq-train.json')
    write_json(train_datas, output_path)

    output_path = os.path.join(output_dataset_dir, 'nq-dev.json')
    write_json(dev_datas, output_path)

    output_path = os.path.join(output_dataset_dir, 'nq-test.json')
    write_json(test_datas, output_path)