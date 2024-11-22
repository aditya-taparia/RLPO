Thank you for identifying the novelty and appreciating the experiments. We hope the following explanations will clarify the queries for the reviewer.

### Weakness a

To compute TCAV scores, we need two concept image groups. The reason for random grouping is that, initially, we do not have TCAV scores for individual concept images. Random grouping is not problematic–it is the unbiased sampling method. Why this works is, because images in the two groups are different, the TCAV score in one group is slightly higher. Then we fine-tune stable diffusion to generate images similar to that group with higher TCAV. We regenerate two groups of images from the fine-tuned model and iterate this process. Although the images in initial groups are highly variable, overtime they learn how to generate images of a particular type/concept.

We can also think of a different analogy. This sampling step is somewhat analogous to rejection sampling and Metropolis-Hasting (MH). In rejection sampling, we pick a sample from a proposal distribution, which is typically a uniform distribution, and reject it based on some criteria. Similarly, we generate two sets of images randomly and evaluate which one to reject. In M-H also we compare two points (though they are old and new points). Through such iterative rejections and model updates, we can converge to the target distribution.

### Weakness b

We would like to clarify that, the action space is not 20 just seed prompts but the combination of the 20 seed prompts. Assuming 30 mins per run, it will take 2<sup>20</sup> * 30 = 182 years to do this if we brute-force. Since RL intelligently and dynamically picks which prompt combinations to use (and not use), RLPO takes only ~8 hours. Therefore, unlike a static ranking approach, our RL-based framework is much more pragmatic to handle unbounded generative models. The high epsilon case in Table 1 is somewhat similar (yet better) to brute forcing through the seed prompts.

To see the quality of generated images with and without RL (seed prompt “stripes” for the zebra class after 300 steps), please see the images below.

<b>Without RL</b>

<p align="center">
  <img src="../Images/R1.jpg" alt="Image 1" width="30%">
  <img src="../Images/R2.jpg" alt="Image 2" width="30%">
  <img src="../Images/R3.jpg" alt="Image 3" width="30%">
</p>

<b>With RL</b>

<p align="center">
  <img src="../Images/R4.jpg" alt="Image 1" width="30%">
  <img src="../Images/R5.jpg" alt="Image 2" width="30%">
  <img src="../Images/R6.jpg" alt="Image 3" width="30%">
</p>

### Weakness c

We agree that we should have clarified this point in the paper. We can still obtain explanations in real-time because TCAV can run in real-time. However, we agree that RLPO cannot create concept sets in real-time, mainly because of the diffusion fine-tuning step (RL is very fast). However, we do not think there is a need to create concept sets in real time. For instance, if we apply TCAV for identifying a disease from an X-ray, we can create the concept set using a test set ahead of time before deployment, which will take a few hours, and then run TCAV in real-time. Hence, concept set creation is a one-time investment. In case of a long-term distribution shift in a particular application, we can keep adding concepts to the dictionary, if RLPO discovers anything new. Please also note that the traditional method of manually creating a concept set can not only be slow and labor intensive but also can miss the important concepts.

### Weakness d

Let us explain with an analogy. Why does it snow on Mount Denali in Alaska? It could be due to its high elevation, its location in the Arctic, or the orographic effect—all valid explanations. Similarly, if an autonomous vehicle hits a pedestrian, why did it happen? Perhaps the pedestrian was occluded, the AV struggled to identify pedestrians wearing pants, or a reflection might have confused its sensors.

If only engineers can obtain the range of reasons why a neural network triggers for a particular output, they can assess the vulnerabilities of the neural network and fix them. Humans cannot think of all these reasons because they do not understand the neural network’s learning process. That’s where our method shines.

While generating multiple concepts is not mandatory, it provides an added advantage. In critical applications, relying on a single explanation can be risky, especially if it fails to capture the full scope of the model's behavior. A diverse set of concepts ensures that users and domain experts can explore multiple dimensions of the model's reasoning. These explanations could also potentially allow experts to learn from the model itself [1].

1. Schut, Lisa, et al. "Bridging the human-ai knowledge gap: Concept discovery and transfer in alphazero." arXiv preprint arXiv:2310.16410 (2023).

### Weakness e

Compared to, say, PPO, DQN is stable for discrete action spaces and more sample efficient. However, DQN can be replaced with any deep RL algorithm that supports discrete action spaces. While the LoRA weights are updated iteratively, the TCAV scores (or rewards) generated for a given set of concepts remain consistent for the same configuration of weights (we do not change the neural network under test).
