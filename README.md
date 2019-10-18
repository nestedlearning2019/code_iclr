# Keras and Tenserflow implementation for the paper : Nested Learning for multi-granular tasks
This anonymous github repository corresponds to the code for the ICLR 2020 submission : [Nested Learning for multi-granular tasks]((https://openreview.net/pdf?id=Byxl-04KvH)). The goal of this open source code is to show how we implemented the concept
of nested learning for simple classification problems, but also to encourage the reader to try its own architecture.
The architecture we chose is indeed not the main contribution of the paper, so you are welcom to try any new architecture that somehow stick
to the general framework presented in section 2 of the paper, described by the following image.
![Architecture](https://github.com/nestedlearning2019/code_iclr/blob/master/framework.png) "Illustrative scheme of the proposed framework.
%
From left to right, the input data $x\sim X$, a first set of layers that extract from $X$ a feature representation $f_1$, which leads to $\hat{Y}_1$ (estimation of the coarse label $Y_1$).
$f_1$ is then jointly exploited in addition with complementary information of the input.
This leads to a second representation $f_2$ from which a finer classification is obtained.
The same idea is repeated until the fine level of classification is achieved.
It is important to highlight that this high level description of the proposed model can be implemented in multiple ways.
In the following sections we present our own implementation and provide details of the specific architecture we chose for experimental validation."
