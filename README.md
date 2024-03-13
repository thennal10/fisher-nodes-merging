# Fisher Mask Node Merging
This repos contains a basic implementation for the model merging procedure in **Fisher Mask Nodes for Language Model Merging**. The code is based on the original implementations of the papers [Merging Models with Fisher-Weighted Averaging](https://arxiv.org/abs/2111.09832), [A Fast Post-Training Pruning Framework for Transformers](https://arxiv.org/abs/2204.09656), and [Editing Models with Task Arithmetic](https://arxiv.org/abs/2212.04089).
## Setup
To install the required packages, run:
```bash
pip install -r requirements.txt
```
## Usage
`example.ipynb` contains an example of how to use the code to merge two models. To recreate our evaluations, run `eval_compare.py` with the appropriate `config.json` file. An example of the `config.json` file is provided in this repo.
