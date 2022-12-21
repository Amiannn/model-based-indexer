import torch

from torch import nn
from torch import Tensor as T
from typing import Tuple, List

from transformers import (
    T5Config, 
    T5EncoderModel, 
    T5ForConditionalGeneration
)

class HFT5Encoder(nn.Module):
    def __init__(
        self,
        cfg_name: str,
        embedding_path: str=None
    ):
        super().__init__()
        cfg           = T5Config.from_pretrained(cfg_name if cfg_name else "t5-base")
        self.model    = T5EncoderModel.from_pretrained(cfg_name, config=cfg)
        self.cfg_name = cfg_name
        self.embedding_path = embedding_path
        
    def forward(
        self,
        input_ids: T,
        attention_mask: T,
    ):
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = out.last_hidden_state
        pooled_output   = torch.mean(sequence_output, 1)
        return pooled_output

class HFT5Model(nn.Module):
    def __init__(
        self,
        cfg_name : str,
        trie_path: str=None
    ):
        super().__init__()
        cfg            = T5Config.from_pretrained(cfg_name if cfg_name else "t5-base")
        self.model     = T5ForConditionalGeneration.from_pretrained(cfg_name, config=cfg)
        self.cfg_name  = cfg_name
        self.trie_path = trie_path

    def forward(
        self,
        input_ids: T,
        attention_mask: T,
        decoder_input_ids: T=None,
    ):
        if decoder_input_ids != None:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids
            )
        else:
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        logits = out.logits
        encoder_last_hidden_state = out.encoder_last_hidden_state
        encoder_pooled_output = torch.mean(encoder_last_hidden_state, 1)

        return logits, encoder_pooled_output

    def constrained_bs(
        self,
        input_ids: T,
        num_beams: int = 10,
        num_return_sequences=10,
        **kwargs
    ) -> List[str]:

        outputs = self.model.generate(
            input_ids,
            min_length=0,
            max_length=20,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs
        )
        return outputs
