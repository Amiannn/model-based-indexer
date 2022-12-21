import os

from torch.utils.data import Dataset
from transformers     import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    idx = tokenizer.prepare_seq2seq_batch(
        src_texts=[
            prefix + ": " + input_text
            for prefix, input_text, _ in dataset
        ],
        tgt_texts        = [tgt for _, _, tgt in dataset],
        max_length       = args.max_seq_length,
        max_target_length= args.max_length,
        padding          = "max_length",
        return_tensors   = "pt",
        truncation       = True,
    )
    return idx['input_ids'], idx['attention_mask'], idx['labels']

class Seq2SeqDataset(Dataset):
    def __init__(self, datas_path):
        datas = []
        with open(datas_path, 'r') as frs:
            for fr in frs:
                data = fr.replace('\n', '')
                data = data.split('\t')
                datas.append(data)

        self.datas = datas[1:]

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, index):
        return self.datas[index]

if __name__ == '__main__':
    dataset = [
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['question', 'who made the song king of the road famous', 'CICA'],
        ['document', '[2018 Little League World Series results] 2018 Little ', 'BEAB']
    ]

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    datas = DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=1,
        collate_fn=lambda examples: preprocess_batch_for_hf_dataset(examples, tokenizer, 100, 100),
        pin_memory=True,
        shuffle=True,
    )
    for data in datas:
        for key in data:
            print(f'{key}: {data[key].shape}')
        print()
    
