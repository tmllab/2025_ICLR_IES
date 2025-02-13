<h2 align="center">Instance-dependent Early Stopping</h2>
<p align="center"><b>ICLR 2025 Spotlight</b> | <a href="https://openreview.net/pdf?id=P42DbV2nuV">[Paper]</a> | <a href="https://github.com/tmllab/2025_ICLR_IES">[Code]</a> </p>
<p align="center"> <a href="https://suqinyuan.github.io">Suqin Yuan</a>, <a href="https://runqilin.github.io">Runqi Lin</a>,  <a href="https://lfeng1995.github.io">Lei Feng</a>, <a href="https://bhanml.github.io">Bo Han</a>, <a href="https://tongliang-liu.github.io">Tongliang Liu</a> </p>

### TL;DR
IES (Instance-dependent Early Stopping) advances the concept of early stopping by applying it at the individual instance level, achieving training speedup without compromising model performance.

### BibTeX
```bibtex
@inproceedings{
yuan2025instancedependent,
title={Instance-dependent Early Stopping},
author={Suqin Yuan and Runqi Lin and Lei Feng and Bo Han and Tongliang Liu},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
```

### Experiments
You should put the [CIFAR datasets](https://www.cs.toronto.edu/~kriz/cifar.html) in the folder `.\cifar-10` and `.\cifar-100` when you have downloaded them.

To run the CIFAR-10 example with IES, run the following:
```bash
python3 cifar_main.py --dataset cifar10 --model resnet18
```

To run the CIFAR-100 example with IES, run the following:
```bash
python3 cifar_main.py --dataset cifar100 --model resnet34
```

To run the examples with baseline, add `--threshold 0`.

The training efficiency can be further improved by adjusting the `--threshold` parameter. We set the default base threshold to `1e-3`, and recommend tuning it between `1e-1` and `1e-5` to achieve different trade-offs between training speed and model performance.

Contact: Suqin Yuan (suqinyuan.cs@gmail.com).
