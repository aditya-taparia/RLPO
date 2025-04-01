We appreciate the reviewer’s constructive comments and recognition of the importance of concept generation in our work. Our method improves upon traditional automatic concept retrieval approaches. While conventional methods extract concepts directly from the dataset—risking semantic information leakage—our approach generates novel concepts that are independent of the dataset.

We have addressed the reviewer’s concerns as follows.

### Q1: Reliability on text prompts.
Our use of seed prompts serves primarily to narrow the search space and provide a reasonable initialization for the concept generation. Given that Stable Diffusion is conditioned on text prompts, starting with semantically meaningful phrases helps the RL agent converge faster toward relevant concept regions. 

While the generated concepts may initially align with the seed prompts, RLPO iteratively optimizes the generation process via TCAV-based rewards, allowing for drift and refinement toward more model-relevant abstractions—even beyond the original prompt scope. We demonstrate this evolution through multiple RL steps (e.g., Figure 2, Appendix D.4), and also show in ablation (Appendix C.4) that random prompts perform significantly worse, indicating the importance of a good starting point rather than dependence.

### Q2: Hyperparametrs used for TCAV score calculation.
TCAV can be sensitive to choices like the layer of activation and the classifier used. To address this, as stated in Appendix C.1, we replaced the default SGD classifier with Logistic Regression, which we found provided more stable CAVs with lower variance. Regarding the choice of layers and target classes, we provide details in Appendix C.3.

### Q3: Bias introduced by pre-trained models.
We acknowledge that the diversity of generated outputs depend on the generative model's capabilities. Issues such as insufficient representation of certain patterns could limit the range of explanations. However, such limitations are not unique to generative approaches—they are also inherent to retrieval-based methods, which are similarly constrained by available data and even human collected concepts, which are influenced by cognitive bias. However, we agree that this is a good point and we will include a discussion of this limitation in the revised manuscript. Specifically, we will highlight the dependency of the explanations on the generative model's capability to produce high-quality and diverse outputs. Thank you for pointing it out.

### Q4: In Figure 6, it is unclear what x-axis “Steps” refers to.
“Steps” in figure 6 refers to a step in c-deletion. At each step we delete a part of the image representing the target concept. We have provided more elaborate examples in Fig 21 and 22 of Appendix. We will clarify this in the figure caption of the revised manuscript.

### Suggestions on related works
Thank you for the suggested references. We will include a discussion of Feature Visualization (Olah et al., Distill) and the more recent diffusion-based concept learning works (e.g., https://arxiv.org/abs/2312.02974, https://arxiv.org/abs/2410.05217). While our method relies on classifier feedback via TCAV and thus differs in motivation, your point about manual curation versus automated discovery is well-taken and worth addressing more directly in Section 2.