# Model-Based Indexer

## Installation
```bash
# Step 1 使用 git 下載專案
git clone https://github.com/Amiannn/model-based-indexer.git
cd model-based-indexer

# Step 2 使用 conda 建立虛擬 python 環境
conda create --name drscl python=3.7
conda activate drscl

# Step 3 安裝套件
pip3 install -r requirements.txt
```

## Dataset
- [Natural Question](https://huggingface.co/datasets/natural_questions)
- 可以使用`python3 dataprocess/create_nq_train.py --document_length 10000`來產生訓練資料集。

## Model Training
```bash
# train dsi model
./scripts/dsi/train.sh

# train dpr model
./scripts/dpr/train.sh

# train drscl model
./scripts/drscl/train.sh
```

## Model Inference
```bash
# predict dsi model
./scripts/dsi/predict.sh

# predict dpr model (需要先產生documents的embedding，用embedding.sh)
./scripts/dpr/predict.sh

# predict drscl model
./scripts/drscl/predict.sh
```

## Reference
- [Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)
- [Transformer Memory as a Differentiable Search Index](https://openreview.net/pdf?id=Vu-B0clPfq)
- [Building an Enhanced Autoregressive Document Retriever
Leveraging Supervised Contrastive Learning](https://aclanthology.org/2022.rocling-1.34/)
