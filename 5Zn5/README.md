Thank you for your constructive review. We have addressed the concerns in weakness and questions as follows.

### Weakness

The primary goal of our method is to generate human-centric concepts that explain a NN’s decision-making process, without requiring extensive pre-collection of concept data. By leveraging generative models and reinforcement learning, we aim to automatically discover and generate the features and abstractions that are most relevant to the model’s predictions.
To evaluate the concept's relevance to the model, we evaluate the method’s success through multiple metrics, as reflected in the experiments section. Table 4 provides quantitative evidence for the relevance of the generated concepts by measuring TCAV scores. Higher TCAV scores indicate that the concepts discovered are aligned with the model’s internal representations, confirming their significance for decision-making. This shows that the generated concepts "matter" to the model. At the same time, other metrics like cosine similarity and euclidean distance shows that the generated concepts are farther away from the class images indicating that what we generated is not a subset of class data.
We also conducted a human experiment (results shown in Table 2), where when asked for identifying relevant concepts within generated, retrieved or both, most volunteers irrespective of laymen or experts mostly picked retrieved concepts, even though both were equally important to the NN. This experiment indicates that it is not easy for them to consider concepts outside class data.
Furthermore, to help humans understand what each relevant concept represents, we made use of ClipSeg to identify the intersection between generated concept images and test images. Figure 6 highlights the regions each relevant concept represents in the test image.

---

### Question 1

For the experiments presented in the paper, the RLPO framework with DQN+Diffusion typically requires approximately 7 hours per class on a machine equipped with an NVIDIA RTX 4090 GPU to train, with the most computationally intensive step being the iterative fine-tuning of the generative model. We will mention this in the revised version of the paper.

### Question 2

The images in Figure 5 represent intermediate stages of the generative process, where the diffusion model is still iteratively refining concepts. These intermediate images tend to be more abstract, as they capture the initial abstractions learned by the model. On the other hand, the images in Figure 6 are final outputs of the RLPO framework after optimization. These images are generated after the RLPO training gets completed, resulting in higher quality and better alignment images with the target class.
