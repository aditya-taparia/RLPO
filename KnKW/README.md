We appreciate your thoughtful review, which acknowledges the originality, experiments, and clarity of our approach. Please see our response below.

### Weakness 1

<b>Are generated images not representative of the seed prompt?</b>
We would like to clarify a potential misunderstanding that the generated images should look like the seed prompt. With the example described in the paper (Figure 5), we wanted to show that there are millions of very diverse images that can be generated using the seed prompt (in this case, “zoo”). However, because of RLPO, the diffusion model learns to only generate images that explain the network’s decisions. Figure 5 demonstrates this process. Please observe that the seed prompt “zoo” becomes more animal-ish at t=10, then animals get more stripes at t=20, then colors appear at t=30, and so on. <i>We have now added this description in the paper.</i>

<b>Are images unrealistic?</b>
Neural network classifiers can provide the same output for different input patterns/concepts. Some of those concepts might be expected whereas some others are unexpected or unrealistic. As we argue with reference to Figure 2, we want RLPO to find such concepts that the human might think of. Therefore, the ability to reveal unrealistic concepts that give a high TCAV score is what we indeed wanted (note that the image quality does not degrade in this case as the TCAV score is higher and the Aesthetics score>3.). Having said that, for the specific case in Figure 5, please note that t=30 is not the final image—we just showed timesteps until the class label flips to tiger.

Aesthetics scorer: https://github.com/discus0434/aesthetic-predictor-v2-5?tab=readme-ov-file

### Weakness 2

<b>What’s the necessity of RL?</b>
Using only GPT was indeed our first attempt, which proved to be inefficient. 

Note that the action space is not 20 seed prompts but the combination of the 20 seed prompts. Assuming 30 mins per run, it will take 2^20 * 30 = 182 years to do this if we brute-force. Since RL intelligently and dynamically picks which prompt combinations to use (and not use), RLPO takes only ~8 hours. Therefore, unlike a static ranking approach, our RL-based framework is much more pragmatic to handle unbounded generative models. The high epsilon case in Table 1 is somewhat similar (yet better) to brute forcing through the seed prompts. 

Images generated after fine-tuning for concept “stripes” for the zebra class with and without RL after 300 steps.

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

### Weakness 3

<b>Is computational complexity an issue?</b>
We can still obtain explanations in real-time because TCAV can run in real-time. However, we agree that RLPO cannot create concept sets in real-time, mainly because of the diffusion fine-tuning step (RL is very fast). However, we do not think there is a need to create concept sets in real time. For instance, if we apply TCAV for identifying a disease from an X-ray, we can create the concept set using a test set before deployment, which will take a few hours, and then run TCAV in real-time. Hence, concept set creation is a one-time investment. In case of a long-term distribution shift in an application, we can keep adding concepts to the dictionary, if RLPO discovers anything new. On a different note, please also note that the traditional method of manually creating a concept set can not only be slow and labor intensive but also can miss the important concepts.

---

### Question 1
<b>Are explanations faithful?</b>
Good point. Compared to a random search algorithm, RL dynamically rewards updates that give a high TCAV score. Therefore, conceptually, we do not expect RLPO to generate totally different concept sets every run.  We now ran the RLPO algorithm 3 times (i.e. 3 trials) for the same seed prompt set. During inference, we calculated the CLIP embedding similarity between trials for the top three seed prompts (stripes, running, and mud for the zebra class - see Figure 6). The low Wasserstein and high cosine similarities indicate that the generations are similar. A low Hotelling's T-squared score (a generalization of the t test for multiple dimensions) also indicates that the generated images are from the same distribution (for reference, this score is 7488.86 for stripes-running). 

<b>Intra-trial Concept Comparisons for “zebra” class across three trial runs:</b>
| Metrics | Stripes-Stripes Concept | Running-Running Concept | Mud-Mud Concept |
|-------------------------------|-----------------------------|-----------------------------|-----------------|
| Average Wasserstein distance | 0.955 ± 0.074 | 0.828 ± 0.074 | |
| Average Cosine similarity | 0.996 ± 0.0008 | 0.997 ± 0.0004 | |
| Average Hotelling's T-squared score | 2.462 ± 0.091 | 2.365 ± 0.081 | 2.663 ± 0.229 |
| Are they from the same distribution? | Yes | Yes | Yes |

### Question 2

In our approach, SD is fine-tuned iteratively: generating a set of images, assessing their relevance, and then refining the model to improve alignment with explainability objectives. This sequential framework allows the model to adaptively optimize towards explanations by building upon the outcomes of previous steps.
Each step in the sequence informs the next, ensuring that the fine-tuning process converges towards increasingly meaningful and interpretable concept representations. This dynamic adjustment is essential for steering SD toward generating high-quality explanations that progressively align with the target class, rather than relying on static, one-shot methods that lack adaptability. Please see this animation we created.

<p align="center">
  <img src="../Images/RLPO Post.gif" alt="animation" width="50%">
  <p style="width: 50%; margin: 0 auto;">Our proposed algorithm, RLPO, iteratively refines the concepts $c_i$ that can be generated by a Stable Diffusion (SD) model by optimizing SD weights based on an action $a_i$. Each step in this update process provides an explanation at a different level of abstraction.</p>
</p>

Also, see the poor performance of non-sequential methods in the new experiments in W2 above.

### Question 3

Yes, fine-tuning one RLPO framework with DQN+Diffusion per class is necessary. This is because each class may have unique features and abstractions that require tailored exploration to generate meaningful and interpretable concept representations. The RLPO framework ensures that the generated concepts align closely with the specific nuances of the class as learned by the model, which a generic approach cannot achieve.
While using an untuned diffusion model with prompts generated by LLMs/VLMs might seem efficient, finding a single long, descriptive prompt that effectively encapsulates the target class's abstractions is highly challenging. Please see experimental evidence in the new benchmark experiment in W2 above.

### Question 4

Thank you for catching this oversight in the appendix. We have rectified this in the revised paper. t<sub>n</sub> refers to the time at which the system reaches an explainable state. After this point, though we can fine-tune SD, TCAV does not change significantly.  
