
## 1. Preface

Github repository for the paper *Dynamical and computational properties of heavy-tailed deep neural networks*, currently submitted and under revision.

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/phasetransition_schematic.jpg" width="750">
</p>

To install requirements:

````
pip install -r requirements.txt
````

To use singularity container:
````
PBS_O_WORKDIR="/project/phys_DL/extended-criticality-dnn" 
cpath="../built_containers/FaContainer_v2.sif" 
bpath="/project"
singularity shell -B ${bpath} --home ${PBS_O_WORKDIR} ${cpath}
````

## 2. Pretrained network statistics

We extract pretrained networks from Pytorch and iteratively fit the entries of each weight matrix (not including biases or batch-normalization layers) independently as a Levy alpha-stable and Gaussian distribution respetively via maximum likelihood.

1. Download pretrained image classification DNN performance (can use ChatGPT) from https://pytorch.org/vision/stable/models.html, this is saved in this repo as `tables/torch_pretrained_performance.csv`, need to move this to the appropripate dir as it is used in `pretrained_workflow/pretrained_download.py`.

2. Download weights (same as old version above)

3. Check if weights are all saved:

`python -W ignore -i pretrained_workflow/pretrained_download.py pretrained_store_check`

4. Fit the full distribution

For the first set `with_logl = True`

`python pretrained_workflow/pretrained_wfit.py pretrained_allfit /project/PDLAI/project2_data/pretrained_workflow/np_weights_all 0 True`

    - weight_path: the dir of the stored weights (divided into 2 networks dir from either torch or tensorflow)

    - n_weight (int): integer index of the weight matrix

    - with_logl (bool): return log-likelihood or not

If there are remaining weights that aren't fitted in the sense that neither the params or logl are obtained, run the following to only get the fitted params without getting the logl:

`python pretrained_workflow/pretrained_wfit.py pretrained_allfit /project/PDLAI/project2_data/pretrained_workflow/np_weights_all 0 False`

Then run the following to get the divided logl:

`python pretrained_workflow/pretrained_wfit.py divided_submit divided_logl`


5. Fit the distribution tails

`python pretrained_workflow/pretrained_wfit.py pretrained_ww_plfit /project/PDLAI/project2_data/pretrained_workflow/weights_all /project/PDLAI/project2_data/pretrained_workflow/ww_plfit_all 0 True`

or use submission version:

`python pretrained_workflow/pretrained_wfit.py batch_pretrained_plfit batch_plfit_submit`

    - weight_path: the dir of the stored weights (divided into 2 networks dir from either torch or tensorflow)

    - n_weight (int): integer index of the weight matrix

    - with_logl (bool): return log-likelihood or not    

6. Summarize and plot the results:

`python pretrained_workflow/sep_new_fig1.py`

7. Boxplots on some fitted parameters (optional)

`python pretrained_workflow/pretrained_postfit.py postfit_stats True`

    - pytorch (bool): whether pretrained networks are from torch library

## 3. Random DNN analysis

### Circular manifold projection

A circule manifold is propagated through a network at different initialization schemes. Final hidden layer is evaluated corresponding to a 2D random circular manifold embedded in $\mathbb{R}^N$ and propagated through $L$ layers of a MLP initialized at $(\alpha, \sigma_w)$, the angles $[0, 2\pi)$ are divided into `N_thetas` intervals: 

`python geometry_analysis/great_circle_proj2.py gcircle_save N L N_thetas 100alpha 100sigma_w`

Computation of the top 3 principal components further requires:

`python geometry_analysis/great_circle_proj2.py preplot alpha sigma_w`

To plot the circular manifold projected on the top 3 principal components (PC):

`python geometry_analysis/great_circle_proj2.py gcircle_plot`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/proj3d_single_alpha=1.0_layer=30.jpg" width="750">
</p>



### Phase transition of coefficient of variation

To propagation a circular manifold as above, the following is another method

`python random_dnn/random_dnn.py SEM_save N L N_theta alpha100 g100 rep`

where `N`, $L$, `N_theta are the save as above`; `alpha100` = $100\alpha$ and $g100 = 100 \sigma_w$; `rep` is the repetition of the randomization.

The coefficient of variation (CV) corresponding to Eq. 6 of the main text of a MLP initialized at $(\alpha, \sigma_w)$ up to $L$ layers with `rep` amount of network ensembles can be evaluated and saved using:

`python random_dnn/random_dnn.py SEM_preplot alpha100 g100`

Finally, plot the phase transition figure for the CV for $L = 15,25,35$ as in the main text:

`python random_dnn/random_dnn.py SEM_plot path 15,25,35`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/random_dnn_1by3.jpg" width="750">
</p>



## 4. Network training

The following includes the updated and old scripts for network-training where all the necessary quantities associated in the manuscript are saved accordingly.


### New version

This updated training script is more concise and keeps log of all previously trained networks. To train heavy-tailed/Gaussian initialized multilayer perceptrons (MLPs), i.e. FC10 a MLP with 10 hidden layers, on the MNIST dataset and $(\alpha, \sigma_w) = (1,1)$; standard SGD algorithm with 1024 batch size and learning rate of 0.001 for 650 epochs.

`python train_supervised.py train_ht_dnn mnist 100 100 sgd 1024 None None {root_path} 0.001 650`

Similary, to train heavy-tailed/Gaussian initialized convolution neural networks (CNNs):

`python train_supervised.py train_ht_cnn cifar10 100 100 sgd alexnet fc_default`


### Old version 

The original main focus of theory is revolved around fully-connected neural networks/MLPs, the following are examples of training, please see `main.py` for training options. For network types, please see `train_DNN_code/model_loader.py` for loading method and the module `train_DNN_code/models` for model options.

`python main.py  --model=fc5_mnist_tanh  --init_dict=stable  --init_alpha=1.5  --init_scale_multiplier=2  --save_epoch=50  --epochs=650  --lr=0.1  --batch_size=256  --dataset=mnist`



## 5. Post-training analysis

### Accuracy transition diagram

To get an accuracy transition diagram as the following, a resolution for $\alpha \in [1,2]$ and $D_w^{1/\alpha}$ must be decided apriori, and a network shall be trained on this grid of initialization schemes:

`python main.py  --model=fc5_mnist_tanh  --init_dict=stable  --init_alpha=$alpha$  --init_scale_multiplier=$D_w$  --save_epoch=50  --epochs=650  --lr=0.1  --batch_size=256  --dataset=mnist`

After training, the test accuracy, loss and stopping epoch can be extracted and the plot can also be simultaneously generated via:

`python tranasition_plot_functions/mlp_acc_phase.py mlp_accloss_phase test`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/fc10_mnist_test_epoch=650_grid_all.jpg" width="750">
</p>

In a similar fashion the train accuracy, loss and stopping epoch and also be obtained for AlexNet via

`python tranasition_plot_functions/cnn_acc_phase.py cnn_accloss_phase`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/alexnet_cifar10_train_epoch=500_grid_all.jpg" width="750">
</p>


### Layerwise Jacobian eigenvectors

Saving the input-output layerwise Jacobian eigenvectors (very storage-consuming) for trained MLP (FC10 in the main text):

`python dq_analysis/jac_fcn.py jac_save post alpha100 g100 input_idxs epoch`

- `post`: 0 is preactivation and 1 is post-activation
- `input_idxs`: the index of the train dataset
- `epoch`: epoch of the network trained up to, given that the network parameters are saved for it

Conversion to $D_q$:

`python dq_analysis/jac_fcn.py jac_to_dq alpha100 g100 input_idx epoch post reig`

- `input_idx` has to match the above
- `reig`: 0 is left eigenvector and 1 is right eigenvector

Plotting the results for the Jacobian eigenvectors:

`python dq_analysis/jac_mfrac_plot.py eigvec_magnitude`

<p align="center">
  <img alt="Light" src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/jac_eigvec_mag_pre_r_alpha100=120_g100=100_l=4_epoch=0.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/jac_eigvec_mag_pre_r_alpha100=200_g100=100_l=4_epoch=0.jpg" width="30%">
</p>

`python dq_analysis/jac_mfrac_plot.py dq_vs_q`

<p align="center">
  <img alt="Light" src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/dq_jac_single_pre_r_l=4_epoch=0.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/dq_jac_single_pre_r_l=4_epoch=650.jpg" width="30%">
</p>

`python dq_analysis/jac_mfrac_plot.py d2_vs_depth`

<p align="center">
  <img alt="Light" src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/jac_d2-vs-depth_pre_r_g100=100_epoch=0.jpg" width="30%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Dark" src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/jac_d2-vs-depth_pre_r_g100=100_epoch=650.jpg" width="30%">
</p>


### Neural representation principal components

Similar to above, we analyze the Neural representation principal components (NPCs lol) for networks under different initializations, for getting the neural representations for different layers in the first place:

`python dq_analysis/npc_fcn.py npc_layerwise post alpha100 g100 epochs`

- `epochs`: for example, if you want to plot for epochs 0 and 650 assuming the weights are saved `epochs` has to be a string in the form of `[0,650]` (**no spaces**)

Now to get the corresponding correlation dimension $D_2$:

`python dq_analysis/npc_fcn.py npc_layerwise_d2 post alpha100 g100 epochs`

Finally, plotting the results:

`python dq_analysis/npc_mfrac_post_plot.py metrics_vs_depth`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/fc10_mnist_epoch=0_650_g100=25_100_300_ED_D2_eigvals-vs-depth.jpg" width="750">
</p>



## 6. Pretrained CNNs

### SNR metrics

First, download the pretrained networks from PyTorch

`python pretrained_workflow/pretrained_download.py pretrained_store_dnn`

- `n_model`: index of the model from the library

and TensorFlow

`python pretrained_workflow/pretrained_download.py pretrained_store_dnn_tf`

Then get the signal-to-noise ratio (SNR) and its related geometrical metrics Github Ref. [2]:

`python fewshot-learning/pretrained_macaque_stimuli.py snr_components {model_name} {pretrained}`

- `model_name`: must match the file name, equivalent to the string name for importing the network
- `pretrained`: `True` or `False`, if True then pretrained on ImageNet1K

OR

modify `snr_submit()` function:

`python fewshot-learning/pretrained_macaque_stimuli.py snr_submit snr_components`

Afterwards, compute the CNN's NPC $D_2$ (can be executed simulatenously with the above): 

`python fewshot-learning/pretrained_macaque_stimuli.py snr_d2_mbatch {model_name} {pretrained}`

modify `snr_submit()` function:

`python fewshot-learning/pretrained_macaque_stimuli.py snr_submit snr_d2_mbatch`

Finally plot the results:

`python fewshot-learning/pretrained_macaque_stimuli.py snr_metric_plot`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/pretrained_m=5_metric_alexnet_resnet101.jpg" width="750">
</p>

`python fewshot-learning/pretrained_macaque_stimuli.py extra_metric_plot`

<p align="center">
<img src="https://github.com/CKQu1/anderson-criticality-dnn/blob/master/readme_figs/pretrained_m=5_extra_metrics_alexnet_resnet101.jpg" width="750">
</p>

### PBS scripts

Use the following scripts if any of the computations related to SNR and $D_2$ timed out (with certains changes):

- `PBS_script/ macaque_stimuli_remainder.sh` (first)
- `PBS_script/ macaque_stimuli_d2.sh` (second)

### Resource guide for simulations

- SNR (fewshot-learning/pretrained_macaque_stimuli.py snr_components)
    - level 1: "alexnet" (12GB)
    - level 2: "resnet18", "resnet34", "resnet50",  "resnext50_32x4d", "squeezenet1_1", "wide_resnet50_2" (24GB)
    - level 3: "squeezenet1_0", "resnet101", "resnet152",  "resnext101_32x8d",  "wide_resnet101_2" (32GB)

- d2 (fewshot-learning/pretrained_macaque_stimuli.py snr_d2_mbatch)
    - level 1: "alexnet", "resnet18" (8GB)
    - level 2: "resnet34", "resnet50",  "resnext50_32x4d", "squeezenet1_0", "squeezenet1_1", "wide_resnet50_2" (16GB)
    - level 3: "resnet101", "resnet152",  "resnext101_32x8d",  "wide_resnet101_2" (32GB)

## Citation

```bibtex
@ARTICLE{
    2022arXiv220312967Q,
    author = {{Qu}, Cheng Kevin and {Wardak}, Asem and {Gong}, Pulin},       
    title = "{Extended critical regimes of deep neural networks}",        
    journal = {arXiv e-prints},      
    keywords = {Computer Science - Machine Learning, Condensed Matter - Disordered Systems and Neural Networks, Condensed Matter - Statistical Mechanics, Computer Science - Artificial Intelligence, Statistics - Machine Learning},     
    year = 2022,         
    month = mar,        
    eid = {arXiv:2203.12967},          
    pages = {arXiv:2203.12967},        
    archivePrefix = {arXiv},
    eprint = {2203.12967},       
    primaryClass = {cs.LG}, 
    adsurl = {https://arxiv.org/abs/2203.12967},       
}
```



## References

Please refer to the manuscript.



## Github references

[1] Anomalous diffusion dynamics of SGD, https://github.com/ifgovh/Anomalous-diffusion-dynamics-of-SGD

[2] Exponential expressivity in deep neural networks through transient chaos, https://github.com/ganguli-lab/deepchaos

[3] Neural representational geometry underlies few-shot concept learning, https://github.com/bsorsch/geometry-fewshot-learning

[4] WeightWatcher, https://github.com/CalculatedContent/WeightWatcher


## Gist references

[1] https://gist.github.com/asemptote/fa5de1eb976aa9fcb9d6510265dff6f9 (qsub.py)

[2] https://gist.github.com/asemptote/3c9f901f1346dffb29d21742cb83c933 (original version of train_supervised.py)

[3] https://gist.github.com/CKQu1/c62b1205ac4df74da95927d9b8b78879 (forked version of [2])
