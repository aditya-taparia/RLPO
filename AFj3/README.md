Thank you for your review. We have addressed each point comprehensively in this response to enhance the clarity and impact of the manuscript.

### Weakness a

To compute TCAV scores, it is necessary to group concept images. The reason for random grouping is that, initially, we do not have TCAV scores for individual concept images. TCAV scores are derived based on the differences in activations between these groups and random concepts; thus, without grouping, the scores cannot be computed.

### Weakness b

### Weakness c

### Weakness d

Generating a diverse set of concept images addresses a critical gap in applying concept-based explanations, especially in domains like autonomous driving and medical diagnostics, where predefined concepts may not adequately capture the nuances of real-world decision-making.
Why Diverse Concepts Matter: In many applications, defining relevant concepts manually is either impractical or impossible because critical features influencing the model's decisions may not be apparent to humans. This was particularly evident in attempts to apply concept-based explanations in autonomous driving and medical domains, where missing key concepts led to incomplete or misleading explanations. Developing a concept generation algorithm ensures that the model itself can surface relevant features, including those we might overlook.
Examples: Consider a model tasked with diagnosing diseases from chest X-rays. While predefined concepts like "opacity" or "shape" of lesions might be useful, there may be other subtle features (e.g., texture patterns, edge gradients) that the model relies on but are not explicitly defined by medical experts. Generating a diverse set of concepts allows us to uncover these additional, previously undefined patterns, providing a more comprehensive understanding of the model’s reasoning process.
While generating multiple concepts is not mandatory, it provides an added advantage. In critical applications, relying on a single explanation can be risky, especially if it fails to capture the full scope of the model's behavior. A diverse set of concepts ensures that users and domain experts can explore multiple dimensions of the model's reasoning, making explanations more robust and actionable. These explanations could also potentially allow experts to learn from the model itself [1].

1. Schut, Lisa, et al. "Bridging the human-ai knowledge gap: Concept discovery and transfer in alphazero." arXiv preprint arXiv:2310.16410 (2023).

### Weakness e

While the LoRA weights are updated iteratively, the TCAV scores generated for a given set of concepts remain consistent for the same configuration of weights. This ensures that, at any specific state of the model, the same action (prompt selection) will produce the same reward (TCAV score). The apparent "change" in rewards over time is a result of progressive model improvement rather than an inherent instability or unpredictability in the environment. To ensure stationarity and address any minor variability introduced during the training process. We leverage a replay buffer, where actions and their associated rewards are stored and revisited during training to stabilize learning.
