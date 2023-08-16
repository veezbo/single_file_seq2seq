# single_file_seq2seq
A character-level seq2seq transformer from scratch in a single file [seq2seq.py](https://github.com/veezbo/single_file_seq2seq/blob/main/seq2seq.py).

Optimized for readability and learnability.

## features
- single file
- as readable as possible
- comments for learnings and common errors
- working code that:
  - trains on paired sequences of text
  - given input text, generates the corresponding output text

## demo
We train a character-level seq2seq transformer to translate from Hinglish (a modern hybrid of Hindi and English) to English.

After training, the same model is used to translate sample Hinglish sentences to English.

The dataset used is [cmu-hinglish-dog](https://huggingface.co/datasets/cmu_hinglish_dog) on Huggingface, which provides samples of movie reviews written in Hinglish that have been translated to English.

## dependencies
```
python >= 3.10
torch >= 2.0
datasets
```

## install

```
pip install torch
pip install datasets
```

## run
```
python seq2seq.py
```

## contributing
All contributions in the form of confusions, concerns, suggestions, or improvements are welcome!

## acknowledgements
This repo was motivated by my previous "single file" repo [single_file_gpt](https://github.com/veezbo/single_file_gpt), which in turn was influenced by Andrej Karpathy's [nanogpt](https://github.com/karpathy/nanoGPT/tree/master).

The demo in this repo uses the [cmu-hinglish-dog](https://huggingface.co/datasets/cmu_hinglish_dog) dataset on Huggingface,  orignally produced by [Zhou et al., 2018][1]. This dataset can also be found in the [datasets-CMU_DoG](https://github.com/festvox/datasets-CMU_DoG) repo on Github.

## license
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[1]: https://aclanthology.org/D18-1076/ "Zhou, Kangyan, Prabhumoye, Shrimai, & Black, Alan W. (2018). A Dataset for Document Grounded Conversations. Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing."
