# FashionR2R

FashionR2R: Texture-preserving Rendered-to-Real Image Translation with Diffusion Models [NeurIPS 2024] 

[Project Page](https://rickhh.github.io/FashionR2R/)

## SynFashion Dataset:

[Download](https://drive.google.com/drive/folders/1i5uH0fdlhWYc2DUMpHTLSGntIHRHHpJl?usp=drive_link)

The license of the data follows Attribution-NonCommercial-NoDerivatives 4.0 International (CCBY-NC-ND4.0).

SynFashion consists of 10k rendered images in 20 categories, including pants, T-shirt, lingerie and swimwear, half skirt, hoodie, coat, jacket, set, home-wear, hat, Hanfu, jeans, shorts, down jacket, vest and camisole, shirt, suit, dress, sweater and trench coat. For each category, we use Style3D Studio to build 10 to 40 projects in different 3D geometry with corresponding texture and design, and then randomly sample several new textures to change its appearance. There are overall 375 projects in 3D and 500 additional texture collected from Internet. For each textured 3D geometry, we render four views, including front, back, and two randomly sampled views. After rendering, we crop the enlarged garment area of each image and resize it to 768x1024. Due to legal issues, some of the images contain a digital human figure but not the complete face.

## Set up

### Create a Conda Environment
```
conda env create -f enviroment.yaml
conda activate FashionR2R
```

### Domain Knowledge Injection
The training script for positive finetuning and negative domain embedding is based on the implementations in diffusers.

### Run

```
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python ./Realistic_translation.py \
    --model_path=#your model path \
    --source_image_path=#cg_image dir \
    --output_dir=#output_dir \
    --negative_embedding_dir=#your trained negative domain embedding \
    --source_prompt='' \
    --target_prompt='' \
    --replace_steps_ratio=0.9 \
    --denoising_strength=0.3 \
    --cfg_scale=7.5 \
    --attn_replace_layers=256 \
    --inversion_as_start \
    --use_negEmbedding
```


### Acknowledgement

Our source and training code are based on [FPE](https://github.com/alibaba/EasyNLP/tree/master/diffusion/FreePromptEditing) and [Diffusers](https://github.com/huggingface/diffusers). Many Thanks. 

We thank [torch-fidelity](https://github.com/toshas/torch-fidelity) for KID calculation. 