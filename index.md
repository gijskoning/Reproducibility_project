# Introduction
<!--Erik-->

*Write about Reinforcement learning*
*What the author wanted to achieve*
*How the environment works*

# The design of IAM
<!--Erik-->

# Implementation in Pytorch
<!--Gijs-->
The IAM model for the Warehouse environment is relatively simple to create. 
However, some information in the paper is missing which took more time to figure out.
In short, we have taken these steps to reproduce the paper: Selecting a similar PPO algorithm, 
adapting the Warehouse environment to work with OpenAI Gym and
combine a FNN with GRU in parallel to create the IAM model.

## PPO algorithm
The author implemented his own PPO algorithm, we choose to use an algorithm based on OpenAi PPO from this [repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.).
The code already had some structure for a recurrent network namely the GRU, this why we created our IAM with a GRU instead of an LSTM. 
## Warehouse
The Warehouse environment is created by the Author himself. Fortunately, his code is available, and so we could easily copy the environment. 
A couple of adjustments have been made to get the environment working with the PPO algorithm:
- The observation space property with the correct output was added
- Metadata property set to None

*GRU with FNN. GRU was already available in the implementation*
*Copied model for critic and actor*
*Used this PPO algorithm https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail as the author used a similar algorithm*
*Did use a different epoch of 4 instead of 3 by accident*

## IAM model
The basic IAM model for the Warehouse environment uses an FNN and GRU in parallel. The hidden sizes 
## Other configurations
Since we use a different PPO model as the original paper, to compare the results correctly the baselines used in the paper are also implemented.
These were:
- Single FNN
- Single GRU
## CNN variant
# Results
<!--Erik-->
<!--Zou je misschien kunnen kijken hoe we een legenda kunnen toevoegen aan de plots?-->

*Focused on reproducing Figure 5 of the paper
Minibatch of 8 with four different models
![image](page/images/minibatch8.png)


Notice that FNN 1 obs gives the best results.
We do see that FNN 8 obs performs worse just as in the paper.
FNN 1,8 and GRU on average took less than 4 hours. While IAM took 5 hours.
Minibatch of 32 for just GRU and IAM again showing better results. We ran a minibatch of 32 after we found out we could run it on Linux with 32 processes
![image](page/images/minibatch32.png)

The paper claimed that GRU or LSTM are less stable but actually our results show that GRU is very stable.*

# Reproducibility
<!--Gijs-->

*Missing parameters:
3 runs to determine variations
scale parameter
average done over number of steps
Some inconsistencies when comparing the paper and the appendix about the minibatch
We didn’t find out what time horizon meant as parameter*
The author doesn't matter if the LSTM is used for the RNN. (I think)
# Conclusion
<!--Gijs-->

*Overall the paper was good reproducible. And we want to note that the author gave feedback very quickly which adds to the reproducibility factor. 
However we got different results compared to the paper, IAM gave in our case not the best results, but the FNN with one observation did.*

# Links
Authors implementation: https://github.com/INFLUENCEorg/influence-aware-memory  
Paper: https://arxiv.org/pdf/1911.07643.pdf