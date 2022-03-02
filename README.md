# VisionTransformer for Tensorflow2

This is an implementation of "VisionTransformer and DeiT" on Keras and Tensorflow.

The implementation is based on papers[[1](https://arxiv.org/abs/2010.11929v2), [2](https://arxiv.org/abs/2012.12877v2)] and official implementations[[3](https://github.com/google-research/vision_transformer), [4](https://github.com/facebookresearch/deit)].

## Model

- Model (Distillation Token is optionally.)
  * ViT-Small
  * ViT-Base
  * ViT-Large
  * ViT-Huge
- Pre-trained weight
  * X

## Requirements

- Python 3
- tensorflow 2
- torch 1.1▲

## Reference

 1. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,
    Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby,
    https://arxiv.org/abs/2010.11929v2
   
 2. Training data-efficient image transformers & distillation through attention,
    Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou,
    https://arxiv.org/abs/2012.12877v2
   
 3. Vision Transformer and MLP-Mixer Architectures,
    google-research,
    https://github.com/google-research/vision_transformer
   
 4. DeiT: Data-efficient Image Transformers,
    facebook-research,
    https://github.com/facebookresearch/deit
   
## Contributor

 * Hyungjin Kim(flslzk@gmail.com)