# Invariant-Representation-for-Continual-Learning
This is the official PyTorch implementation for the paper "Learning Invariant Representation for Continual Learning" at the AAAI Meta-Learning for Computer Vision Workshop (2021).

We propose a new pseudo-rehearsal-based method, named learning Invariant Representation for Continual Learning (IRCL), in which class-invariant representation is disentangled from a conditional generative model and jointly used with class-specific representation to learn the sequential tasks. Disentangling the shared invariant representation helps to learn continually a sequence of tasks, while being more robust to forgetting and having better knowledge transfer. We focus on class incremental learning where there is no knowledge about task identity during inference. We empirically evaluate our proposed method on two well-known benchmarks for continual learning: split MNIST and split Fashion MNIST. The experimental results show that our proposed method outperforms regularization-based methods by a big margin and is better than the state-of-the-art pseudo-rehearsal-based method. Finally, we analyze the role of the shared invariant representation in mitigating the forgetting problem especially when the number of replayed samples for each previous task is small.  

# Requirements
* Python 3.6
* Pytorch 1.2
* torchvision 0.4

# Usage
You can use main.py to run our IRCL method on the Split MNIST benchmark. 

```
python main.py
```

# Reference
If you use this code, please cite our paper:
