## simple classify model by pytorch
### How to use
* 1. write config (reference to **configs/classify2050c_densenet121.json**)
* 2. `python train.py --config configs/classify2050c_densenet121.json`

### finished
* multi-GPUs support
* support backbones
    * resnet50
    * densenet121
    * vgg
    * alexnet
    * squeezenet
* visual by visdom
 * ![FCN_results](.temp/0.png)

### TODO
* eval scripts
* more backbones support