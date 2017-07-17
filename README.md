# Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling

Implementation for "[Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling](https://arxiv.org/abs/1611.01462)"

This paper tries to utilize the diversity of word meaning to train the Deep Neural Network.

## Summary of Paper

### Motivation

In the language modeling (prediction of the word sequence), we want to express the diversity of word meaning.  
For example, when predicting the word next to "Banana is delicious ___", the answer is "fruit", but "sweets", "food" is also ok.
But ordinary one-hot vector teaching is not suitable to achieve it. Because any similar words ignored, but the exact answer word.

![motivation.PNG](./doc/motivation.PNG)

If we can use not one-hot but "distribution", we can teach this variety.

### Method

So we use "distribution of the word" to teach the model. This distribution acquired from the answer word and embedding lookup matrix.

![formulation.PNG](./doc/formulation.PNG)

![architecture.PNG](./doc/architecture.PNG)

If we use this distribution type loss, then we can prove the equivalence between input embedding and output projection matrix.

![equivalence.PNG](./doc/equivalence.PNG)

To use the distribution type loss and input embedding and output projection equivalence restriction improves the perplexity of the model.

## Experiments

### Implementation

* [Keras](https://github.com/fchollet/keras): to implements model
* [chazutsu](https://github.com/chakki-works/chazutsu): to download Dataset

### Result

![result.PNG](./doc/result.PNG)

* Run the 15 epoch on Penn Treebank dataset.
  * `perplexity` score is large, I couldn't have confidence of [its implementation](https://github.com/icoxfog417/tying-wv-and-wc/blob/master/model/one_hot_model.py#L50). I'm waiting pull request!
* `augmentedmodel` works better than the baseline(`onehotmodel`), and `augmentedmodel_tying` outperforms the baseline!
* You can run this experiment by `python train.py`

## Additional validation

* At the beginning of the training, embedding matrix to produce "teacher distribution" is not trained yet. So proposed method has a little handicap at first.
  * But the delay of training was not observed 
* Increasing the temperature (alpha) gradually may improve training speed.
* To use the pre-trained word vector, or fixing the embedding matrix weight for some interval (fixed target technique at the reinforcement learning (please refer [*Deep Reinforcement Learning*](http://www.iclr.cc/lib/exe/fetch.php?media=iclr2015:silver-iclr2015.pdf))) will also have effect to the training.

**By the way,  [PyTorch example already use tying method](https://github.com/pytorch/examples/blob/1c6d9d276f3a0c484226996ab7f9df4f90ce52f4/word_language_model/model.py#L28)! Don't be afraid to use it!**
