# Github Repository for the paper
****

Github repository for the paper *Self-organised edge-of-chaos in heavy-tailed deep neural networks*, currently submitted and under revision.

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/phasetransition_schematic1024_1.jpg" width="650">
</p>
  
## Requirements
****

To install requirements:

````
pip install -r requirements.txt
````


## Pretrained network analysis
****

We extract pretrained networks from Pytorch and iteratively fit the entries of each weight matrix (not including biases or batch-normalization layers) independently as a L\`{e}vy alpha stable and Gaussian distribution respetively via maximum likelihood.

1. Download the weight matrices into the `weights_all` directory:

`python pretrained_workflow/pretrained_download`

2. Fit the distributions

`python pretrained_workflow/pretrained_wfit`

3. Plot the results

`python pretrained_workflow/pretrained_metaplot2.py`


## Network training
****

The main focus of theory is revolved around fully-connected neural networks, the following are examples of training, please see `main.py` for training options. For network types, please see `train_DNN_code/model_loader.py` for loading method and the module `train_DNN_code/models` for model options.

`python main.py  --model=fc5_mnist_tanh  --init_dict=stable  --init_alpha=1.5  --init_scale_multiplier=2  --save_epoch=50  --epochs=650  --lr=0.1  --batch_size=256  --dataset=mnist`


## Post-training analysis
****

### Accuracy transition diagram

To get an accuracy transition diagram as the following, a resolution for $\alpha \in [1,2]$ and $D_w^{1/\alpha}$ must be decided apriori, and a network shall be trained on this grid of initialization schemes:

`python main.py  --model=fc5_mnist_tanh  --init_dict=stable  --init_alpha=$alpha$  --init_scale_multiplier=$D_w$  --save_epoch=50  --epochs=650  --lr=0.1  --batch_size=256  --dataset=mnist`

After training, the accuracies can be extracted and the plot can also be simultaneously generated via:

`python tranasition_plot_functions/grid_accuracy.py/`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/fc10_mnist_tanh_grid_testacc_early_6501024_1.jpg" width="650">
</p>

### Geometry analysis

A circule manifold is propagated through a network at different initialization schemes.

To plot the circular manifold projected on the top 3 principal components (PC):

`python geometry_analysis/great_circle_proy.py`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/fc10_mnist_tanh_grid_testacc_early_6501024_1.jpg" width="650">
</p>

## Citation
****




## References
****

Please refer to the manuscript.


## Code references
****

[1] Anomalous diffusion dynamics of SGD, https://github.com/ifgovh/Anomalous-diffusion-dynamics-of-SGD

[2] Exponential expressivity in deep neural networks through transient chaos, https://github.com/ganguli-lab/deepchaos
