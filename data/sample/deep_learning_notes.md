# Deep Learning Notes

## What is Deep Learning?
Deep learning is a subset of machine learning that uses neural networks with many layers (hence "deep") to learn complex patterns from large amounts of data. It has revolutionized computer vision, natural language processing, and speech recognition.

## Neural Network Fundamentals

### Layers
- **Input layer**: receives raw data (pixels, words, numbers)
- **Hidden layers**: learn increasingly abstract representations
- **Output layer**: produces the final prediction

The more hidden layers, the more abstract features the network can learn. A network with 2+ hidden layers is considered "deep".

### How Learning Works
1. Data flows forward through layers (forward pass)
2. Network makes a prediction
3. Loss function measures how wrong the prediction is
4. Gradients flow backward through layers (backpropagation)
5. Weights update to reduce the loss
6. Repeat millions of times

### Weights and Biases
Every connection between neurons has a weight — a number that scales the signal. Each neuron also has a bias that shifts its activation. Learning = finding the right weights and biases.

## Convolutional Neural Networks (CNNs)

CNNs are designed for grid-like data (images, video). Instead of connecting every neuron to every other neuron, they use filters that slide across the input detecting local patterns.

### Key Layers
- **Conv2D**: applies learnable filters to detect edges, textures, shapes
- **MaxPooling**: reduces spatial dimensions, keeps strongest activations
- **Flatten**: converts 2D feature maps to 1D vector
- **Dense**: fully connected layers for final classification

### Why CNNs Work for Images
Early layers detect edges and corners.
Middle layers detect shapes and textures.
Deep layers detect complex objects like faces, cars, dogs.

Famous CNNs: VGG, ResNet, EfficientNet, MobileNet.

## Recurrent Neural Networks (RNNs)

RNNs process sequential data by maintaining a hidden state that carries information from previous time steps.

### The Vanishing Gradient Problem
In long sequences, gradients shrink exponentially during backpropagation — early time steps receive almost no learning signal. This makes vanilla RNNs bad at long-range dependencies.

### LSTM (Long Short-Term Memory)
LSTMs solve the vanishing gradient problem with gates:
- **Forget gate**: decides what to erase from memory
- **Input gate**: decides what new information to store
- **Output gate**: decides what to output

### GRU (Gated Recurrent Unit)
Simpler than LSTM with fewer parameters. Often performs comparably. Uses update gate and reset gate instead of three separate gates.

## Transformers

Transformers replaced RNNs for most NLP tasks. Instead of processing sequences step-by-step, they process all tokens in parallel using self-attention.

### Self-Attention
For each token, self-attention computes how much to "attend" to every other token in the sequence.

Formula: Attention(Q, K, V) = softmax(QK^T / √d_k) · V

- Q (Query): what am I looking for?
- K (Key): what do I contain?
- V (Value): what do I pass along?

### Why Transformers Dominate
- Parallelizable — no sequential bottleneck like RNNs
- Long-range dependencies captured in one step
- Scale well with data and compute

### BERT vs GPT
- **BERT**: bidirectional, reads left and right — good for understanding tasks (classification, QA)
- **GPT**: unidirectional, reads left to right — good for generation tasks (text completion, chat)

## Regularization Techniques

### Dropout
Randomly sets a fraction of neurons to zero during training. Forces the network to learn redundant representations. Reduces overfitting.

```
dropout_rate = 0.3  # 30% of neurons zeroed out each step
```

### Batch Normalization
Normalizes activations within a mini-batch. Stabilizes training, allows higher learning rates, acts as mild regularizer.

### Weight Decay (L2 Regularization)
Adds a penalty to the loss for large weights: loss = original_loss + λ * Σ(w²). Encourages smaller, more general weights.

### Early Stopping
Monitor validation loss during training. Stop when validation loss stops improving. Prevents memorizing the training set.

## Optimizers

### SGD (Stochastic Gradient Descent)
Basic optimizer. Uses a single sample (or mini-batch) to estimate gradients. Noisy but fast. Needs careful learning rate tuning.

### Adam (Adaptive Moment Estimation)
Most popular optimizer. Adapts the learning rate for each parameter individually. Combines momentum and RMSProp. Default choice for most tasks.

### Learning Rate Scheduling
Start with a high learning rate and reduce it over time:
- **Step decay**: reduce by factor every N epochs
- **Cosine annealing**: smoothly decrease following cosine curve
- **Warmup**: start low, increase, then decrease — used in Transformers

## Transfer Learning

Instead of training from scratch, start with a model pretrained on a large dataset and fine-tune on your specific task.

### Why It Works
Pretrained models have already learned general features (edges, textures, grammar, semantics). Fine-tuning adapts these features to your domain with far less data and compute.

### Common Pretrained Models
- **Image**: ResNet, EfficientNet, ViT (Vision Transformer)
- **Text**: BERT, RoBERTa, GPT, T5, LLaMA
- **Multimodal**: CLIP (image + text), Whisper (audio)

### Fine-tuning Strategies
- **Full fine-tuning**: update all weights — best accuracy, most compute
- **Frozen backbone**: only train the final layers — fast, good for small datasets
- **LoRA**: low-rank adaptation — efficient fine-tuning with very few trainable parameters

## Common Deep Learning Mistakes

1. **Not normalizing inputs** — always scale inputs to [0,1] or zero mean / unit variance
2. **Too large learning rate** — loss explodes or oscillates wildly
3. **Too small learning rate** — training takes forever, may get stuck
4. **Insufficient data** — deep networks need lots of examples; use augmentation
5. **No validation set** — can't detect overfitting without held-out data
6. **Training too long** — use early stopping to prevent memorization
