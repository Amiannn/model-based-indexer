import os
import json

from tqdm import tqdm

def read_file(path, sp='\t'):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in tqdm(frs):
            data = fr.replace('\n', '')
            data = data.split(sp)
            datas.append(data)
    return datas

def read_json(path):
    datas = None
    with open(path, 'r', encoding='utf-8') as fr:
        datas = json.load(fr)
    return datas

def write_file(datas, path, sp='\t'):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(sp.join(data) + '\n')

def write_json(datas, path):
    with open(path, 'w', encoding='utf-8') as fr:
        json.dump(datas, fr, indent=4)