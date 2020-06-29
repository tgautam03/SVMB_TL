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

##### Loss curves
![subset 1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/loss_curves/DI_DC/images/NN3_set1_16bit.png)
![subset 2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/loss_curves/DI_DC/images/NN3_set2_16bit.png)
![subset 3](https://github.com/tgautam03/SVMB_TL/blob/master/Images/loss_curves/DI_DC/images/NN3_set3_16bit.png)
![subset 4](https://github.com/tgautam03/SVMB_TL/blob/master/Images/loss_curves/DI_DC/images/NN3_set4_16bit.png)

##### Predictions
Velocity models are plotted followed by the vertical velocity profile for the same.
![ex1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/1110.png)
![log1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/1110_vellogs.png)


![ex2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/1573.png)
![log2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/1573_vellogs.png)


![ex3](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/210.png)
![log3](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/210_vellogs.png)


![ex4](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/2319.png)
![log4](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/DI_DC/set4/2319_vellogs.png)
 
### SEG/EAGE Salt Body Models
After training the CNN on ‘Synthetic Velocity Models’ described in the previously, we saved that CNN with its trained weights and used that to perform various tests on 'SEG/EAGE salt models'. First, we predicted all the SEG/EAGE salt models with the CNN trained on 'Synthetic Velocity Models' only (described next) and then we fine-tuned the CNN with a small subset of 'SEG/EAGE Salt models' before predicting rest of the 'SEG/EAGE Salt models'.

#### Predicting with CNN trained on ‘Synthetic Velocity Models’ only
The results where we directly used the CNN trained on Complex Models to predict 'SEG/EAGE salt models', which were from a completely different distribution as compared to the models which the CNN had seen previously .

##### Predictions
Velocity models are plotted followed by the vertical velocity profile for the same.
![ex1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/allPred/1181.png)
![log1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/allPred/1181_vellogs.png)


![ex2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/allPred/330.png)
![log2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/allPred/330_vellogs.png)


![ex3](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/allPred/920.png)
![log3](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/allPred/920_vellogs.png)

#### Predicting after fine-tuning CNN
The approach of Transfer Learning, where we fine-tuned our CNN using a subset of SEG/EAGE salt models. We divided this test into three parts where in each part we used different size of the subset for fine-tuning the CNN. First, we randomly picked 304 velocity models (subset 1) out of total 2028 SEG/EAGE salt models, fine-tuned the CNN and kept rest of the velocity models for testing. After this, we increased the size of training data from 304 to 608 (subset 2) and 1014 (subset 3). After fine-tuning in all three scenarios we compared the accuracy of the CNN for various cases. Alongside fine-tuning, we also trained the same CNN from scratch (Fresh CNN Training) and compared the results.

##### Predictions
Velocity models are plotted followed by the vertical velocity profile for the same.

###### Subset 1
![ex1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set304/1198_models.png)
![log1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set304/1198_vellogs.png)


![ex2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set304/347_models.png)
![log2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set304/347_vellogs.png)

###### Subset 2
![ex1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set608/1102_models.png)
![log1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set608/1102_vellogs.png)


![ex2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set608/250_models.png)
![log2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set608/250_vellogs.png)

###### Subset 3
![ex1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set1014/225_models.png)
![log1](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set1014/225_vellogs.png)


![ex2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set1014/900_models.png)
![log2](https://github.com/tgautam03/SVMB_TL/blob/master/Images/predictions/segsalt/set1014/900_vellogs.png)


## Conclusion
Prediction of Velocity Models using CNN, heavily relies on the distribution of training and testing dataset. Results are quite good if the training and the testing data sets have similar settings but the accuracy takes a hit when the two settings are different. However, if we use fine-tuning on a very small subset, we get exponentially better results.

## References
- Aminzadeh, F., Brac, J. & Kunz, T., 1997. 3D Salt and Overthrust models: Presented at the SEG. s.l., s.n.
- Louboutin, M. et al., 2018. Devito: an embedded domain-specific language for finite differences and geophysical exploration. arXiv preprint arXiv:1808.01995.
- Oliphant, T., 2006. A guide to Numpy. USA, Trelgol Publishing.
- Silva, R. M. et al., 2019. Netherlands dataset: A new public dataset for machine learning in seismic interpretation. arXiv preprint arXiv:1904.00770.
- Tan, C. et al., 2018. A survey on deep transfer learning. Cham, Springer, pp. pp. 270-279.
- Wu, Y. and Lin, Y., 2019. InversionNet: An Efficient and Accurate Data-Driven Full Waveform Inversion. IEEE Transactions on Computational Imaging , Volume 6, pp. 419-433.
- Yang, F. and Ma, J., 2019. Deep-learning inversion: A next-generation seismic velocity model building method. Geophysics , 84(4), pp. R583-R599.

## Note
For additional imformation on plots and metrics, visit [trained_nets](https://github.com/tgautam03/SVMB_TL/tree/master/Images).

Contact : [Tushar Gautam](mailto:tushariitr3@gmail.com?subject=[GitHub]%20Source%20Han%20Sans)
