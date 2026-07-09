# Machine Learning Notes

## What is Machine Learning?
Machine learning test is a subset of AI that enables systems to learn from data without being explicitly programmed. Instead of writing rules, you feed examples and the model learns patterns.

## Types of Machine Learning

### Supervised Learning
The model learns from labeled data — each input has a known correct output.
- **Classification**: predicting a category (spam vs not spam, dog vs cat)
- **Regression**: predicting a number (house price, temperature)
- Examples: linear regression, decision trees, neural networks

### Unsupervised Learning
No labels — the model finds hidden structure in data.
- **Clustering**: grouping similar items (k-means, DBSCAN)
- **Dimensionality reduction**: compressing data while keeping structure (PCA, t-SNE)
- **Anomaly detection**: finding outliers

### Reinforcement Learning
An agent learns by interacting with an environment and receiving rewards or penalties.
- Used in: game playing (AlphaGo), robotics, recommendation systems

## Key Concepts

### Overfitting vs Underfitting
- **Overfitting**: model memorizes training data but fails on new data (too complex)
- **Underfitting**: model is too simple to capture the patterns
- Solution: regularization, more data, cross-validation

### Gradient Descent
Optimization algorithm that minimizes the loss function by iteratively moving in the direction of steepest descent.
- Learning rate controls step size
- Too high → overshoots; too low → slow convergence
- Variants: SGD, Adam, RMSProp

### Embeddings
Dense vector representations of data (text, images, users).
- Similar items have vectors that are close together in vector space
- Foundation of modern NLP (word2vec, BERT, sentence-transformers)
- Measured with cosine similarity or dot product

## Neural Networks

### Architecture
- **Input layer**: raw features
- **Hidden layers**: learned representations
- **Output layer**: predictions

### Activation Functions
- ReLU: max(0, x) — prevents vanishing gradients
- Softmax: converts logits to probabilities (multi-class)
- Sigmoid: squashes to [0,1] (binary classification)

### Training Loop
1. Forward pass: compute predictions
2. Compute loss (cross-entropy, MSE, etc.)
3. Backward pass (backpropagation): compute gradients
4. Update weights with optimizer

## Transformers and LLMs
Transformers use self-attention to process sequences in parallel.
- Attention score = softmax(QK^T / √d) · V
- Pre-training on massive text corpora gives broad knowledge
- Fine-tuning adapts to specific tasks
- Prompt engineering guides behavior without retraining
