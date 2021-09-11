# TempoQR
This is the code for the manuscript [Temporal Question Reasoning for Question Answering over Temporal Knowledge Graphs](https://drive.google.com/file/d/1-dOQa0B_vm9bMoO4uuFjCabgoxAH7waK/view?usp=sharing).


## Installation

Clone and create a conda environment
``` 
git clone git@github.com:apoorvumang/CronKGQA.git
cd CronKGQA
conda create --prefix ./tempoqr_env python=3.7
conda activate ./tempoqr_env
```
<!-- Make sure ``python`` and ``pip`` commands point to ``./tempoqr_env``. Output of ``which`` should be something like
```
which python
[...]/TempoQR/tempoqr_env/bin/python
```
If this is not the case, try replacing ``python`` with ``python3``. If that works, replace ``python`` with ``python3`` in all commands below.
 -->
The implementation is based on CronKGQA in [Question Answering over Temporal Knowledge Graphs](https://arxiv.org/abs/2106.01515) and their code from https://github.com/apoorvumang/CronKGQA.

We use TComplEx KG Embeddings as proposed in [Tensor Decompositions for temporal knowledge base completion](https://arxiv.org/abs/2004.04926). We use a slightly modified version of their code from https://github.com/facebookresearch/tkbc,
as in CronKGQA

Install TempoQR requirements
```
conda install --file requirements.txt -c conda-forge
```

## Dataset and pretrained models download

Download and unzip ``data.zip`` and ``models.zip`` in the root directory.

Drive: https://drive.google.com/drive/folders/1aS2s5sZ0qlDpGZ9rdR7HcHym23N3pUea?usp=sharing.

## Running the code


TempoQR:
```
python ./train_qa_model.py --model tempoqr --supervision soft
python ./train_qa_model.py --model tempoqr --supervision hard
 ```
 
Other models: "entityqr" and "cronkgqa" with hard and soft supervisions.
 
To use a corrupted TKG change to "--tkg_file train_corXX.txt" and "--tkbc_model_file tcomplex_corXX.ckpt", where XX=20,33,50.

To evaluate on unseen complex questions change to "--test test_bef_and_aft" or "--test test_fir_las_bef_aft".

Please explore more argument options in train_qa_model.py.

