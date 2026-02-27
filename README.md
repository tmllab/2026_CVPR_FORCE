<div align="center">

# FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction
[![Paper](https://img.shields.io/badge/paper-CVPR-green)](https://arxiv.org/abs/2509.21029)

</div>

Official implementation of [FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](https://arxiv.org/abs/2509.21029) (CVPR 2026).

## Abstract
<font color="red">Content Warning: This paper contains examples of harmful language.</font>

The integration of new modalities enhances the capabilities of multimodal large language models (MLLMs) but also introduces additional vulnerabilities.
In particular, simple visual jailbreaking attacks can manipulate open-source MLLMs more readily than sophisticated textual attacks.
However, these underdeveloped attacks exhibit extremely limited cross-model transferability, failing to reliably identify vulnerabilities in closed-source MLLMs.
In this work, we analyse the loss landscape of these jailbreaking attacks and find that the generated attacks tend to reside in high-sharpness regions, whose effectiveness is highly sensitive to even minor parameter changes during transfer.
To further explain the high-sharpness localisations, we analyse their feature representations in both the intermediate layers and the spectral domain, revealing an improper reliance on narrow layer representations and semantically poor frequency components.
Building on this, we propose a Feature Over-Reliance CorrEction (FORCE) method, which guides the attack to explore broader feasible regions across layer features and rescales the influence of frequency features according to their semantic content.
By eliminating non-generalizable reliance on both layer and spectral features, our method discovers flattened feasible regions for visual jailbreaking attacks, thereby improving cross-model transferability.
Extensive experiments demonstrate that our approach effectively facilitates visual red-teaming evaluations against closed-source MLLMs.

<p float="left" align="center">
<img src="Draw.pdf" width="650" />

**Figure.** Schematic illustration of the generation and transfer of optimisation-based visual jailbreaking attacks, as well as the feasible regions of such attacks in the input space.
</p>

## Requirements
- This codebase is written for `python3` and 'pytorch'.


## Experiments

### Model
- Please download and place all models into the Model directory.

### Data
- Please download and place all datasets into the Dataset directory.


### Training

Generate FORCE jailbreaking attack

```
python attack_force.py --dataset_path ../Dataset/malicious.txt --image_path ../Dataset/image.png --source_model_name ../llava-v1.5-7b
```

Evaluate FORCE jailbreaking attack

```
python target_Llava1.6_cls.py
python target_Blip_cls.py
python target_Idefics3_cls.py
python target_Llama_cls.py
python target_Qwen_cls.py.
python target_GPT_cls.py
python target_Gemini_cls.py
python target_Claude_cls.py
```


## License and Contributing
- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- Feel free to post issues via Github.

## Reference
If you find the code useful in your research, please consider citing our paper:

<pre>
@article{lin2025force,
  title={FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction},
  author={Lin, Runqi and Paren, Alasdair and Yuan, Suqin and Li, Muyang and Torr, Philip and Bibi, Adel and Liu, Tongliang},
  journal={arXiv preprint arXiv:2509.21029},
  year={2025}
}
</pre>
