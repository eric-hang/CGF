# Counterfactual Generation Framework for Few-Shot Learning

By Anonymous author. (To be updated)

This repository is an implementation of the paper "Counterfactual Generation Framework for Few-Shot Learning".

** Apologize for the dirty code. We are working on organizing the code.

## Requirements


```sh
pip install -r requirements.txt
```

## Prepare Datasets

---

To be updated.

## Training

MiniImageNet:

```sh
python -u main.py --phase pretrain_encoder --gpu 1 --save-path "./experiments/" --train-shot 5 --val-shot 1 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet12_inv --dataset miniImageNet --z_disentangle --zd_beta 6.0 --zd_beta_annealing --add_noise 0.2 --temperature 500 --feature_size 640 --generative_model vae --latent_size 64 --attSize 171
```


TieredImageNet:


```sh
python -u main.py --phase pretrain_encoder --gpu 1 --save-path "./experiments/" --train-shot 1 --val-shot 1 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet12_inv --dataset tieredImageNet --z_disentangle --zd_beta 6.0 --zd_beta_annealing --add_noise 0.2 --temperature 500 --feature_size 640 --generative_model vae --attSize 641 --latent_size 64
```

CIFAR-FS:

```sh
python -u main.py --phase pretrain_encoder --gpu 0 --save-path "./experiments/" --train-shot 5 --val-shot 1 --train-query 15 --val-query 15 --head FuseCosNet --network ResNet12_inv --dataset CIFAR-FS --z_disentangle --zd_beta 6.0 --zd_beta_annealing --add_noise 0.2 --temperature 500 --feature_size 640 --generative_model vae --attSize 164 --latent_size 64
```

## Main Results

| Dataset        | Backbone | 5w1s   | 5w5s   |
| -------------- | -------- | ------ | ------ |
| MiniImageNet   | ResNet12 | 80.12% | 86.13% |
| TieredImageNet | ResNet12 | 77.60% | 87.38% |
| CIFAR-FS       | ResNet12 | 87.20% | 89.99% |

## License

To be updated

