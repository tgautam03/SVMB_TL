# Transfer Learning to predict realistic subsurface velocity models
## Abstract
Understanding of velocity distribution is crucial to image the subsurface in all its complexity. In existing practice, majority processing of the seismic data using conventional and computational intensive algorithms with simplistic assumptions about the wave propagation still leaves out a number of uncertainties in the heterogeneous subsurface depositions. The alternative is to create multiple realization of the velocity model and iterate till the error minimizes in imaging the strati-structural feature of the subsurface as is the case with the high resolution technique of Full Waveform Inversion (FWI). Therefore, fundamentally, it is the initial Velocity Model Building (VMB) that controls the success of the entire process of seismic exploration. Full Waveform Inversion (FWI) can give high resolution subsurface velocity models however, there is a heavy reliance on a good initial model, otherwise, cycle-skipping problems will trap the algorithm in a local minimum. Deep Neural Networks (CNN) can predict velocity models from raw seismic traces, which are fairly close to the ground truth velocity models, but when the required velocity models are from a different distribution as compared to training velocity models, CNN struggles. Techniques like, Transfer Learning can then help mitigate this problem, and we can get velocity models which are very close to ground truth models. Sometimes, it’s hard to get large amount of training data, so we’ve also demonstrated the use of (VAE) to generate realistic subsurface velocity models. This work uses 20 layered CNN with around 30 million trainable parameters. It was initially trained on around 14000 velocity models, and the final mean squared error (MSE) on test velocity models was around 0.03. This CNN was then fine-tuned on randomly picked 304 velocity models from SEG/EAGE salt and was able to achieve the MSE of around 0.0232 on the remaining velocity models.

## Problem Statement
The objective of this work was to show that we can extend the modern algorithms of Machine Learning to seismic velocity modeling effectively. Primarily, we were trying to predict velocity models from SEG/EAGE Salt Body dataset (Aminzadeh, et al., 1997), hence we needed a CNN which understood the mapping between raw seismic traces and corresponding velocity models. To train the CNN, we created velocity models using Netherlands F3 Interpretation Dataset (Silva, et al., 2019). Once the CNN was trained, we tried to predict SEG/EAGE salt models but the results were not satisfactory. To improve the results, we used Transfer Learning (Tan, et al., 2018), and fine-tuned CNN using a subset of SEG/EAGE salt body dataset. 

## Results
The sequence of all the various tests were put in place such that we increased the complexity of various step by step. At first, we split ‘Synthetic Velocity Models’ generated using Netherlands F3 Interpretation into Training (~14000), Validation (~1700) and Testing (~2600) sets. Then CNN was trained on this training set and predictions were analysed on corresponding test set to see if CNN could understand the features in a raw seismic trace and map those to accurate velocity models.

After getting promising results, we took our testing to SEG/EAGE salt models. Here, at first we predicted all of the SEG/EAGE salt models using the CNN trained on ‘Synthetic Velocity Models’ only and this test revealed the limitations of our method. However, Transfer Learning (Tan, et al., 2018), helped us mitigate these limitations, and we demonstrated that with various tests. 

### Synthetic Velocity Models
As the dataset was quite large, we split the dataset into 4 subsets with an approximate of total 4000 training and 800 validation models in each subset. After training was completed on all four subsets, trained CNN was evaluated on 2600 test examples, metrics for each steps avaible in [trained_nets](https://github.com/tgautam03/SVMB_TL/tree/master/trained_nets/DI_DC/metrics). Training curve for each subset is shown below, followed by predictions.

#### Loss curves
![subset 1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/loss_curves/DI_DC/images/NN3_set1_16bit.png)
![subset 2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/loss_curves/DI_DC/images/NN3_set2_16bit.png)
![subset 3](https://github.com/tgautam03/SVMB_TL/blob/master/Images/loss_curves/DI_DC/images/NN3_set3_16bit.png)
![subset 4](https://github.com/tgautam03/SVMB_TL/blob/master/Images/loss_curves/DI_DC/images/NN3_set4_16bit.png)

#### Predictions
Velocity models are plotted followed by the vertical velocity profile for the same.
![ex1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/1110.png)
![log1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/1110_vellogs.png)


![ex2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/1573.png)
![log2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/1573_vellogs.png)


![ex3](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/210.png)
![log3](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/210_vellogs.png)


![ex4](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/2319.png)
![log4](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/2319_vellogs.png)
 
