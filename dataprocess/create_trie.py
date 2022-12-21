import os
import pickle
import argparse

from tqdm            import tqdm
from transformers    import T5Tokenizer
from utils.fileio    import read_json
from utils.trie      import Trie


def get_token(text, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    return [0] + input_ids[0].tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create trie for constrained beam search")
    parser.add_argument("--input_path"   , type=str, default=None, help="Document id path.")
    parser.add_argument("--model_type"   , type=str, default='t5-base', help="model arch type")
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.model_type)

    datas = read_json(args.input_path)
    sids = [data['identifier']['semantic'] for data in datas]
    # create trie
    # sids_tokens = [get_token(' '.join(list(text)), tokenizer) for text in sids]
    sids_tokens = [get_token(''.join(list(text)), tokenizer) for text in sids]
    trie = Trie(sids_tokens)

    trie_dict = trie.trie_dict
    print(trie_dict)
    
    output_dir  = '/'.join(args.input_path.split('/')[:-1])
    output_path = os.path.join(output_dir, 'trie.pickle')
    
    with open(output_path, 'wb') as f:
        pickle.dump(trie_dict, f)

# USAGE:
# python3 utils/dsi/create_trie.py \
# --input_path ./data/10k/sid/docs_w32_10k_st5_pid2sid.tsv \
# --output_dir data/10k/trie \
# --model_type "t5-base"