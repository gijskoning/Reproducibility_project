![image](poster.png)

# Introduction
*Write about Reinforcement learning*
*What the author wanted to achieve*
*How the environment works*

# The design of IAM
# Implementation in Pytorch
*GRU with FNN. GRU was already available in the implementation*
*Copied model for critic and actor*
*Used this PPO algorithm https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail as the author used a similar algorithm*
*Did use a different epoch of 4 instead of 3 by accident*

# Results
*Focused on reproducing Figure 5 of the paper
Minibatch of 8 with four different models
Notice that FNN 1 obs gives the best results.
We do see that FNN 8 obs performs worse just as in the paper.
FNN 1,8 and GRU on average took less than 4 hours. While IAM took 5 hours.
Minibatch of 32 for just GRU and IAM again showing better results. We ran a minibatch of 32 after we found out we could run it on Linux with 32 processes
The paper claimed that GRU or LSTM are less stable but actually our results show that GRU is very stable.*

# Reproducibility
*Missing parameters:
3 runs to determine variations
scale parameter
average done over number of steps
Some inconsistencies when comparing the paper and the appendix about the minibatch
We didn’t find out what time horizon meant as parameter*

# Conclusion
*Overall the paper was good reproducible. And we want to note that the author gave feedback very quickly which adds to the reproducibility factor. 
However we got different results compared to the paper, IAM gave in our case not the best results, but the FNN with one observation did.*

# Links
Our implementation: https://github.com/gijskoning/Reproducibility_project  
Authors implementation: https://github.com/INFLUENCEorg/influence-aware-memory  
Paper: https://arxiv.org/pdf/1911.07643.pdf