# Gradient Stabilization Techniques in Deep Learning (PyTorch)

This project documents my hands-on exploration of **gradient stabilization techniques in deep learning using PyTorch**. The work focuses on understanding how weight initialization, activation functions, normalization layers, and gradient clipping help stabilize training and prevent vanishing or exploding gradients.

---

## Objective

The objective of this work is to:
- Understand why gradients become unstable during training
- Explore practical techniques to stabilize gradients
- Implement and compare different normalization strategies
- Apply gradient clipping in recurrent neural networks

---

## 1. Weight Initialization Techniques

### Manual Scaling
- Initialized a linear layer (`nn.Linear`)
- Manually scaled weights and zeroed biases
- Observed how improper initialization can affect gradient flow

### Kaiming (He) Initialization
- Applied `kaiming_uniform_` initialization for ReLU-based networks
- Used custom initialization via `model.apply()`
- Zero-initialized biases explicitly

This ensures stable variance propagation through ReLU activations.

---

## 2. Activation Functions and Gradient Flow

### ReLU
- Used as the default activation
- Noted that ReLU can zero out gradients for negative activations

### Leaky ReLU
- Introduced `LeakyReLU` with configurable negative slope (`alpha`)
- Applied Kaiming initialization specific to leaky ReLU
- Helps reduce vanishing gradients for negative inputs

---

## 3. Batch Normalization

Batch Normalization was applied to stabilize activations during training:
- Used `BatchNorm1d` before and after linear layers
- Demonstrated that BatchNorm reduces the need for external feature scaling
- Inspected learnable parameters and running statistics using:
  - `named_parameters()`
  - `named_buffers()`

Bias terms were removed from linear layers when followed by BatchNorm to avoid redundant computation.

---

## 4. Normalization Order and Design Choices

- Applied BatchNorm **before ReLU** to center activations around zero
- Built a multi-layer fully connected network with interleaved BatchNorm and ReLU layers
- Demonstrated how normalization improves stability in deeper networks

---

## 5. Batch Normalization vs Layer Normalization

### Batch Normalization
- Normalizes across batch dimension (column-wise)
- Depends on batch statistics during training

### Layer Normalization
- Normalizes across feature dimensions per sample (row-wise)
- Independent of batch size

Both methods were demonstrated using simple tensor examples to highlight their differences.

---

## 6. Layer Normalization on Image Tensors

- Applied `LayerNorm` to 4D image tensors
- Demonstrated different normalization scopes:
  - Spatial dimensions only (`[H, W]`)
  - Channel + spatial dimensions (`[C, H, W]`)

This shows how normalization behavior changes based on shape configuration.

---

## 7. Gradient Clipping in Recurrent Networks

To address exploding gradients:
- Built an LSTM model
- Performed forward and backward passes
- Applied `clip_grad_norm_` to rescale gradients without changing direction

This technique is particularly important for RNNs and LSTMs where gradients propagate through many time steps.

---

## Key Observations

- Proper weight initialization is critical for stable gradient flow
- Leaky ReLU helps mitigate dead neurons caused by ReLU
- BatchNorm reduces internal covariate shift and stabilizes training
- LayerNorm is better suited for small batch sizes and sequence models
- Gradient clipping effectively prevents exploding gradients in LSTMs

---

## Conclusion

This project provides a practical exploration of gradient stabilization techniques in PyTorch. By experimenting with initialization strategies, activation functions, normalization layers, and gradient clipping, I developed a deeper understanding of how modern deep learning models remain stable during training.

These techniques form the foundation for training deep and recurrent neural networks reliably.

---
