We are thankful to the reviewer for the feedback. While we appreciate the reviewer's recognition of the interesting questions posed, we believe there are some key misunderstandings regarding the claims and the experiments of the proposed method.

In this work, we propose a novel approach to “generate” concepts that truly matter to the neural network by utilizing reinforcement learning-based preference optimization (RLPO) on diffusion models. And, we show qualitatively and quantitatively that the generated concepts matter to the neural network. This has been supported by Reviewer 1, 2 and 4. With regards to experiments, please see our detailed answers below.

### Q1: Generalizability beyond visual concepts.

RLPO does not solely depend on direct linguistic grounding. Instead it uses reinforcement learning to evolve and refine the generative model via XAI feedback (from the model under test) - indirectly capturing internal representation that may not have exact language equivalents. The use of language (seed prompts), serves primarily to narrow the search space and provide a reasonable initialization for the concept generation. While the generated concepts may initially align with the seed prompts, RLPO iteratively optimizes the generation process via TCAV-based rewards, allowing for drift and refinement toward more model-relevant abstractions—even beyond the original prompt scope. Additionally, to highlight the generalizability of our approach beyond visual concepts, in Section 4.6 (Fig. 8), we demonstrate how RLPO generalized to non-visual domains like sentiment analysis, using textual input and preference. 

### Q2: Clarification on C-deletion.

We agree with the reviewer that the original C-deletion is performed in the pixel space. But we would like to clarify that, the C-deletion we show in the paper is also performed in the pixel space and **not in the textual space**. As highlighted in Q1, the use of language is to narrow down the search space. Once the training is completed, we generate concepts via the trained diffusion model and map those generated concepts back to the input space using CLIPSeg (as shown in Figure 5). 

The C-deletion graphs shown in paper are obtained from deleting most relevant to least relevant target concepts from the input images. We have provided more elaborate examples in Fig 21 and 22 of Appendix.

### Q3: Potential semantic leakage in diffusion alignment since the vision models are already trained on ImageNet? We need more details on the concept vocabulary, how many semantic names are out of ImageNet vocabulary? How many classes can it generate? Can it generate novel categories/objects?

We understand the reviewer’s concern on potential semantic leakage in diffusion alignment since these models have already been trained on ImageNet data, but we would disagree with the reviewer because the concepts we generate don't come from class data. As shown in Table 4, the generated concepts by RLPO are farthest from the class data. On a contrary, this semantic leakage occurs in retrieval based methods where the concepts are collected from class data. This is the exact problem we are trying to resolve with our method.

### Q4: The action space is not scaled, only a few words. Besides, most prompts are single phrases, and thus cannot scale up to diverse compositions?

Our current action space consists of 20 seed prompts, preprocessed and extracted using VQA (see Appendix C.3). We would like to emphasize that our approach does not rely on a direct mapping from seed prompts to the final generated outputs (concepts). While the generated concepts may initially align with the seed prompts, RLPO iteratively optimizes the generation process via TCAV-based rewards, allowing for drift and refinement toward more model-relevant abstractions—even beyond the original prompt scope. Consequently, the final generated concepts capture more diverse and model-relevant abstractions that extend beyond the limitations of the initial single phrases.

### Q5: What if generated images in both groups are both garbage? Will it still update the SD?

If both image sets yield low TCAV scores (i.e., do not activate the model meaningfully), our method does not update the diffusion model. Only when a concept has the potential to move toward explainable states do we update the diffusion model. We will clarify this further in the revised manuscript.