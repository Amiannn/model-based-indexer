import torch
import pickle
import numpy as np
import torch.nn.functional as F

from torch             import Tensor as T
from torch              import nn
from typing            import Any, List
from transformers      import T5Tokenizer
from pytorch_lightning import LightningModule

class ContrastiveModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net

        # # loss function
        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        self.embedding = None
        if self.net.embedding_path != None:
            with open(self.net.embedding_path, 'rb') as f:
                embedding = pickle.load(f)
            self.embedding = embedding
        self.top_k = 10
        # load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.net.cfg_name)
        
        self.val_acc = []

    def _accuracy(self, preds, labels):
        r_index = torch.tensor(range(preds.size(0)))
        c_index = preds

        return torch.sum(labels[r_index, c_index]) / preds.size(0)

    def _dot_product_scores(self, q_vectors: T, ctx_vectors: T):
        # row_vector: n1 x D, col_vectors: n2 x D, result n1 x n2
        r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
        return r

    def _supConLoss(self, log_softmax_scores, neg_mask):
        # supervised contrastive loss
        return torch.sum(-1 * (log_softmax_scores * neg_mask) / torch.sum(neg_mask, dim=1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        return self.net(x, x_mask)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best = None

    def step(self, batch: Any):
        x_row, x_col, y, neg_mask = batch['x_row'], batch['x_col'], batch['y'], batch['neg_mask']

        x_row_idx, x_row_mask = x_row['input_ids'], x_row['attention_mask']
        x_col_idx, x_col_mask = x_col['input_ids'], x_col['attention_mask']
        y_idx = y['input_ids']

        x_row_vectors = self.forward(x_row_idx, x_row_mask)
        x_col_vectors = self.forward(x_col_idx, x_col_mask)

        scores = self._dot_product_scores(x_row_vectors, x_col_vectors)

        if len(x_row_vectors.size()) > 1:
            x_row_num = x_row_vectors.size(0)
            scores    = scores.view(x_row_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss  = self._supConLoss(softmax_scores, neg_mask)
        preds = torch.argmax(softmax_scores, dim=1)
        return loss, preds, neg_mask

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        acc = self._accuracy(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc" , acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        acc = self._accuracy(preds, targets)
        self.val_acc.append(acc)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc" , acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc[-1]                 
        self.val_acc_best = max(self.val_acc)  
        self.log("val/acc_best", self.val_acc_best, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        acc = self._accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc" , acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_predict_start(self):
        assert self.embedding != None
        self.vectors    = self.embedding['vectors'].to(self.device)
        self.vectors_id = np.array(self.embedding['labels'])

    def predict_step(self, batch: Any, batch_idx: int):
        x_row, x_col, y, neg_mask = batch['x_row'], batch['x_col'], batch['y'], batch['neg_mask']

        x_row_idx, x_row_mask = x_row['input_ids'], x_row['attention_mask']
        x_col_idx, x_col_mask = x_col['input_ids'], x_col['attention_mask']
        y_idx = y['input_ids']

        x_row_vectors = self.forward(x_row_idx, x_row_mask)
        scores        = self._dot_product_scores(x_row_vectors, self.vectors)
        pred_index    = torch.argsort(-scores).cpu().detach().numpy()[:, :self.top_k]

        outputs = self.vectors_id[pred_index].tolist()
        return outputs

    def embedding_step(self, batch: Any, batch_idx: int):
        x_row, x_col, y, neg_mask = batch['x_row'], batch['x_col'], batch['y'], batch['neg_mask']

        x_row_idx, x_row_mask = x_row['input_ids'], x_row['attention_mask']
        y_idx = y['input_ids']

        x_row_vectors = self.forward(x_row_idx, x_row_mask)

        return x_row_vectors

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

