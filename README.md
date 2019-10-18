# Keras and Tenserflow implementation for the paper : Nested Learning for multi-granular tasks
This anonymous github repository corresponds to the code for the ICLR 2020 submission : [Nested Learning for multi-granular tasks]((https://openreview.net/pdf?id=Byxl-04KvH)). The goal of this open source code is to show how we implemented the concept of nested learning for simple classification problems, but also to encourage the reader to try its own architecture. The architecture we chose is indeed not the main contribution of the paper, so you are welcom to try any new architecture that somehow stickto the general framework presented in section 2 of the paper, described by the following image.
![Architecture](https://github.com/nestedlearning2019/code_iclr/blob/master/framework.png) "Illustrative scheme of the proposed framework"
From left to right, the input data $$x\sim X$$, a first set of layers that extract from $$X$$ a feature representation $$f_1$$, which leads to $$\hat{Y}_1$$ (estimation of the coarse label $$Y_1$$). $$f_1$$ is then jointly exploited in addition with complementary information of the input. This leads to a second representation $$f_2$$ from which a finer classification is obtained. The same idea is repeated until the fine level of classification is achieved.

We will present in this readme the general conditions to run the code.

## Preparing the environment
In order to run the code, you should read this section and make sure your python environment fullfills all the following requirements. Using GPUs and cuda 7.0 is highly recommended for computation time purposes. 
* Download this directory 
* Make sure Keras and Tensorflow are installed in your python 3 environment
* In this directory, create the following directory, using these terminal instructions : 
```
#directory that stores the result csv files
$mkdir results
#directory that stores the Tensorboard runs
$mkdir log_dir
#directory that stores the weights
$mkdir weights
## the following lines depend on which dataset you want to test the model 
$cd weights
$mkdir weights_mnist
$mkdir weights_fashion_mnist
$mkdir weights_cifar10
```


## How to run the code
The 2 important files to run to get results are `main2.py` and `calibration.py`. The order in which you run those two is crucial.
#### Running `main2.py`
This script can either train or test the model chosen for the chosen dataset. The models are defined in the files `models_[dataset].py`, and you can write your own models in those files. You should first run the model in train mode, because the test mode relies on already trained model which do not exist if the model isn't trained.

##### Training
Let's think about an example. We want to try the model we designed for the MNIST dataset, with 30 % of only coarsely annotated samples from the original dataset (2 categories, 18000 samples), 30 % of coarse and intermediate data (samples annotated both with 2 categories and 4 categories, 18000 samples), and 20% of coarse middle and fine data (samples annotated  with 2 coarse categories, 4 internediate categories and 10 fine classes, 12000 samples). In order to train this model, copy paste this in the terminal :
```
$python3 main2.py --dataset "mnist" --traintime True --model-id 0 --coarse-rate 80.0 --middle-rate 50.0 --fine-rate 20.0 
```
The model-id parameter is convenient so that you know which trained model you are testing afterwards if you want to compute several trainings. In order to change the training hyper-parameters, you can also add the corresponding arguments with the desired values, which are also presented in the parser.
Running this line will have several consequences:
* creating a 3 ".h5" file for the trained model, one for each step of the training. The information about the training of this model are stored in its name.
* creating a log_dir sub directory with the training details for Tensorboard.
##### Testing
Now that the model is trained, we can test it. For that we can test several perturbations, all coded in the `preprocessing.py` file. Let's say we want to try both the results of the model on the original distribution, and also on the images distorted with parameter $$(S,T) = (1.0,1.0)$$. You can run the following lines :
```
$python3 main2.py --dataset "mnist"--model-id 0 --coarse-rate 80.0 --middle-rate 50.0 --fine-rate 20.0 --perturbation "original"
$python3 main2.py --dataset "mnist"--model-id 0 --coarse-rate 80.0 --middle-rate 50.0 --fine-rate 20.0 --perturbation "warp" --s 1.0 --t 1.0
```
Running this two lines will result in the creation of a csv file in the results directory with the name 'results\_mnist\_id0\_c80\.0\_m50.0\_f20.0.csv'. Each line of this .csv file stores the accuracies and confidences of the coarse, middle and fine classifier, the perturbation type we tested the model on, and the type of model (single output vs single output).

