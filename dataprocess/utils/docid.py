import enum
import os
import json
import pickle
import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans

from sentence_transformers import SentenceTransformer

PASSAGE_EMBEDDING_PATH = './datasets/doclevel/embedding/docs_w32_10k_embedding.tsv'
OUTPUT_PATH            = './datasets/datas/tmp'

n_clusters   = 10
index_tokens = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
kmeans       = KMeans(n_clusters=n_clusters)

EMBEDDING_PATH = './logs/embedding/runs/2022-12-14_01-44-57/embedding.pickle'

def read_pickle(path):
    datas = None
    with open(EMBEDDING_PATH, 'rb') as f:
        datas = pickle.load(f)
    return datas

def read_file(path):
    pids, embeddings = [], []
    with open(path, 'r', encoding='utf-8') as frs:
        for i, fr in tqdm(enumerate(frs)):
            pid, embedding = (fr.replace('\n', '')).split('\t')
            embedding = json.loads(embedding)
            pids.append(pid)
            embeddings.append(embedding)
    return pids, embeddings

def clustering(dx):
    kmeans.fit(dx)
    dy = kmeans.predict(dx)
    return dy

def hierarchical_clustering(ids, dx):
    c     = dict([[i, []] for i in range(n_clusters)])
    c_ids = dict([[i, []] for i in range(n_clusters)])
    dy    = clustering(dx)

    for i, y in enumerate(dy):
        c[y].append(dx[i])
        ids[i].append(y)
        c_ids[y].append(ids[i])

    for n in c:
        c[n] = np.array(c[n])

    for n in c:
        if len(c[n]) > n_clusters:
            hierarchical_clustering(c_ids[n], c[n])
        else:
            for i in range(len(c[n])):
                c_ids[n][i].append(i)
    return

def token(ids):
    return [index_tokens[_id] for _id in ids]

def create_semantic_docid(documents, atomic_ids, model_type):
    model = SentenceTransformer(model_type)

    # start encoding
    embeddings  = model.encode(documents)
    semantic_ids= [[] for i in atomic_ids]

    print('clustering...')
    hierarchical_clustering(semantic_ids, embeddings)
    
    semantic_ids = [''.join(token(sid)) for sid in semantic_ids]
    
    id_datas = {aid: sid for aid, sid in zip(atomic_ids, semantic_ids)}
    return id_datas
    

def write_file(path, datas):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write('\t'.join(data) + '\n')

if __name__ == '__main__':
    datas = read_pickle(EMBEDDING_PATH)
    embeddings = datas['vectors'].numpy()
    labels     = datas['labels']

    print('clustering...')
    ids = [[] for i in labels]
    hierarchical_clustering(ids, embeddings)
    
    ids = [''.join(token(sid)) for sid in ids]
    
    id_datas = zip(labels, ids)
    output_path = './test.tsv'
    write_file(output_path, id_datas)
