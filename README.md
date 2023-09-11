# RCMHA: Relative Convolutional Multi-Head Attention
This repository contains the implementation code for the RCMHA (Relative Convolutional Multi-Head Attention) framework, which addresses the limitations of traditional Multi-Head Attention (MHA) in the context of language modeling within Natural Language Processing (NLP).

## Introduction
The Attention module plays a crucial role in various NLP tasks, and Multi-Head Attention (MHA) is a fundamental component of many state-of-the-art models. However, MHA relies on absolute positional encoding, which imposes constraints on token length and consumes substantial memory resources during processing. While solutions like Transformer-XL and Relative Multi-Head Attention (RMHA) have attempted to mitigate these challenges, they still demand significant memory.

In this study, we introduce RCMHA, a novel approach that enhances the performance of MHA while minimizing memory usage. RCMHA achieves this by combining relative positional encoding with the Depth-Wise Convolutional Layer architecture.

## Usage
edit config.ini to use neptune, and install package with
```bash
pip install -r requirements.txt
```
then run main
```bash
python main.py
```

## Citation
If you find RCMHA useful in your research or projects, please consider citing our paper:

```sql
@misc{sugiharto2023rcmha,
      title={RCMHA: Relative Convolutional Multi-Head Attention for Natural Language Modelling}, 
      author={Herman Sugiharto and Aradea and Husni Mubarok},
      year={2023},
      eprint={2308.03429},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
We would like to thank the contributors and the research community for their valuable insights and support in the development of RCMHA.

##