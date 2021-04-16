# Introduction
<!--Erik-->
In this blog, we are going to tell about the IAM [^iam-paper] paper and how we tried to reproduce some results of that paper.
The title of the IAM paper is Influence-aware memory architectures for deep reinforcement learning and authored by Miguel Suau et al.
The idea of the project was to reimplement the IAM model with a different library and to reproduce some results given in the paper.
The main idea of the paper is to propose an architecture where the agent has some influence-aware memory (IAM).
The agent makes its decisions based on a reinforcement learning model.
With this architecture, it is then possible to remember the useful information from the environment while alleviating the training difficulties compare to conventional methods using RNNs.
The goal of the paper is to show that architecture indeed has an important effect on convergence, learning speed and performance of the agent.
We reproduced the implementation and repeated some of the experiments described in the paper to see if we got some similar results. 
The design of the IAM described will be explained in greater detail in the next section.
After that, we will dive into our implementation and the results we got from that implementation. 
At last, we discuss our results in a comparison with that of the paper and give a short conclusion of the project.
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

Our parameters should be the same as in the paper following the default PPO settings. 
However, after generating all the results we found out that the paper uses a ppo epoch of 3
and our implementation 4. This can affect the output, but we are not comparing the absolute results, only the relative results to the baselines so this should not be a problem.

### PPO algorithm
The author implemented his own PPO algorithm, we choose to use an algorithm based on OpenAi PPO from this [repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail.).
The code already had some structure for a recurrent network namely the GRU, this why we created our IAM with a GRU instead of an LSTM. 
Using the GRU will change the results compared to the paper, but we can still relate the performance of the IAM model to a single GRU.
### Warehouse
The Warehouse environment is created by the author himself. It contains one agent in a 7x7 grid which needs to collect items at boundaries of the grid to receive a reward.
These items appear at random and disapppear after 8 seconds so the model needs some kind of memory to save this information for multiple timesteps.  

![image](page/images/warehouse.png)  
*Warehouse visualization. (blue) agent, (yellow) items.*

Fortunately, the code is available on Github, and so we could easily copy the environment.
The environment is also clearly explained in the paper, so we could have created it ourselves.

Two small adjustments have been made to get the environment working with the PPO algorithm:
- The observation space property with the correct output was added
- Metadata property set to None



## IAM model
As described in the paper the basic IAM model for the Warehouse environment uses an FNN and GRU in parallel. 
The paper didn't explain how the IAM model was used in the PPO algorithm. This is an important detail since the algorithm uses an actor and critic model, and these can be used in different combinations. 
For example parts of the model used in the actor and critic can be shared or completely separate. We decided to use two instances of the IAM model, one as actor and one as critic.  
![image](page/images/iam_model.png)  
*Diagram of the IAM model from Figure 3 of the paper.*
### Baselines
Since we use a different PPO model as the original paper, to compare the results correctly,
the baselines used in the paper are also implemented, again using two instances of the model for actor and critic:
- A single FNN of two layers with one or eight observations as input.
- A single GRU.

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

Figure 5 of the paper:  
![image](page/images/paper_figure5.png)

# Reproducibility
<!--Gijs-->
Overall, using the information from the paper was sufficient to implement the model described. 
But as also discussed earlier the paper does miss some important details.

When creating the plots of our result and comparing it to Figure 5 of the paper, we noticed that we need some additional information on how the graphs were created in the paper.
These are the scale of the reward (can be guessed to be multiplied by 100), 
the amount of timesteps used in the rolling average and
how many training runs where done. Luckily, the author replied quickly to our emails to give this additional information and this adds to the reproducibility factor as is mentioned here .

*To be done:*
Some inconsistencies when comparing the paper and the appendix about the observations used. 8 or 32.
We didn’t find out what time horizon meant as parameter*
The author doesn't matter if the LSTM is used for the RNN. (I think)
*Author doesnt mention the absolute training time*
# Conclusion
<!--Gijs-->

*Overall the paper was good reproducible. And we want to note that the author gave feedback very quickly which adds to the reproducibility factor. 
However we got different results compared to the paper, IAM gave in our case not the best results, but the FNN with one observation did.*

# Links
Authors implementation: https://github.com/INFLUENCEorg/influence-aware-memory  
Paper: https://arxiv.org/pdf/1911.07643.pdf  
PPO implementation used: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail