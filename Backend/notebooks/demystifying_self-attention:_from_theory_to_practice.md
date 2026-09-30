# Demystifying Self-Attention: From Theory to Practice

## Introduction to Attention Mechanisms

In the early days of deep learning, models such as recurrent neural networks (RNNs) and convolutional neural networks (CNNs) processed inputs in a fixed, often sequential or local, manner. While powerful, these architectures struggled with two fundamental challenges:

1. **Long‑range dependencies** – Capturing relationships between distant elements (e.g., words at the beginning and end of a sentence) required many recurrent steps, leading to vanishing gradients and inefficient training.  
2. **Fixed receptive fields** – Convolutions attend only to a limited neighborhood, making it hard to model global context without stacking many layers.

### Why Attention Matters

Attention mechanisms address these issues by allowing a model to **dynamically focus** on the most relevant parts of the input when producing each output. Instead of treating every element equally, attention computes a weighted sum of representations, where the weights (the “attention scores”) reflect the importance of each element for the current task. This yields several practical benefits:

- **Direct access to all positions** → gradients flow more freely, facilitating learning of long‑range patterns.  
- **Interpretability** → the attention weights can be visualized, offering insights into what the model deems important.  
- **Efficiency** – Parallel computation of attention scores enables faster training compared with sequential RNN updates.

### Evolution of Attention

| Milestone | Key Idea | Impact |
|-----------|----------|--------|
| **Bahdanau et al., 2015** (Neural Machine Translation) | Learned alignment between encoder and decoder states. | First demonstration that soft, differentiable attention improves sequence‑to‑sequence models. |
| **Luong et al., 2015** | Introduced global and local attention variants. | Showed flexibility in how much context to attend to. |
| **Vaswani et al., 2017 – “Attention Is All You Need”** | Replaced recurrence entirely with **self‑attention** (also called the Transformer). | Enabled massive parallelism and set new performance standards across NLP, vision, and beyond. |
| **Subsequent works** (e.g., BERT, GPT, Vision Transformers) | Stacked self‑attention layers, added pre‑training objectives, and adapted the mechanism to images and multimodal data. | Turned attention into a universal building block for modern AI. |

### Self‑Attention: The Pivotal Breakthrough

Self‑attention (or intra‑attention) extends the attention concept to **the same sequence**: each element computes a weighted combination of *all* elements—including itself. Formally, for an input matrix \(X \in \mathbb{R}^{n \times d}\) (n tokens, d features), we derive three projections:

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,
\]

where \(W_Q, W_K, W_V\) are learned weight matrices. The attention output is then

\[
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\]

with \(d_k\) the dimension of the keys. This simple operation accomplishes:

- **Content‑based addressing** – Tokens attend to others based on similarity of query and key vectors.  
- **Position‑agnostic modeling** – No recurrence or convolution is needed; all positions interact simultaneously.  
- **Scalability** – By stacking multiple self‑attention layers, models capture hierarchical relationships without explicit depth‑wise design.

In essence, self‑attention turned attention from an auxiliary add‑on into the **core computational engine** of state‑of‑the‑art models, enabling unprecedented performance and flexibility across a spectrum of tasks. The sections that follow will unpack how this mechanism works under the hood and how to harness it in practice.

## What Is Self‑Attention?

Self‑attention (also called intra‑attention) is a mechanism that lets every token in a sequence **directly attend to all other tokens** in the same sequence, computing a weighted representation of the entire context for each position. Formally, given an input sequence of \(n\) token embeddings  
\[
X = \bigl[x_1, x_2, \dots, x_n\bigr] \in \mathbb{R}^{n \times d},
\]  
self‑attention first projects each token into three latent spaces:

* **Queries:** \(Q = XW_Q \in \mathbb{R}^{n \times d_k}\)  
* **Keys:**  \(K = XW_K \in \mathbb{R}^{n \times d_k}\)  
* **Values:** \(V = XW_V \in \mathbb{R}^{n \times d_v}\)

where \(W_Q, W_K \in \mathbb{R}^{d \times d_k}\) and \(W_V \in \mathbb{R}^{d \times d_v}\) are learned weight matrices.

The attention scores between token \(i\) (as a query) and token \(j\) (as a key) are obtained by a scaled dot‑product:

\[
\alpha_{ij} = \frac{\exp\!\bigl(q_i \cdot k_j^\top / \sqrt{d_k}\bigr)}
{\sum_{l=1}^{n} \exp\!\bigl(q_i \cdot k_l^\top / \sqrt{d_k}\bigr)}.
\]

These normalized scores \(\alpha_{ij}\) form an **attention matrix** \(A \in \mathbb{R}^{n \times n}\), where each row sums to 1. The output representation for token \(i\) is then a weighted sum of the value vectors:

\[
\text{SelfAtt}(x_i) = \sum_{j=1}^{n} \alpha_{ij}\, v_j.
\]

Putting it together for the whole sequence:

\[
\text{SelfAtt}(X) = \operatorname{softmax}\!\Bigl(\frac{QK^\top}{\sqrt{d_k}}\Bigr) \, V.
\]

### Contrast with “Traditional” (Encoder‑Decoder) Attention

| Aspect | Traditional (cross) attention | Self‑attention |
|--------|------------------------------|----------------|
| **Source of Keys/Values** | Comes from a **different** sequence (e.g., encoder outputs) | Keys & values are derived from the **same** sequence as the queries |
| **Purpose** | Aligns a target sequence to a source (e.g., translation) | Captures **internal dependencies** within a single sequence (e.g., language modeling) |
| **Attention matrix shape** | \(m \times n\) (target length × source length) | \(n \times n\) (sequence length × sequence length) |
| **Symmetry** | Asymmetric: queries ≠ keys/values | Symmetric in the sense that every token can attend to every other token, including itself |

In other words, traditional attention answers “*Which parts of the **other** sequence are relevant for this token?*”, whereas self‑attention asks “*Which parts of **my own** sequence should I look at to enrich this token’s representation?*”.

### Core Idea Illustrated

Imagine a sentence of five words:  

`[The, quick, brown, fox, jumps]`

For the third word **“brown”**, self‑attention computes a vector that blends information from **all five** words:

```
brown_out = α₃₁·The + α₃₂·quick + α₃₃·brown + α₃₄·fox + α₃₅·jumps
```

The coefficients \(\alpha_{3j}\) are learned so that, for example, the model may give higher weight to “fox” (because adjectives modify nouns) and lower weight to “jumps”. Crucially, **every token repeats this process**, yielding a set of context‑aware embeddings where each position has already “talked” to every other position before any recurrent or convolutional layers are applied.

This all‑pair interaction is what gives Transformer‑based models their remarkable ability to capture long‑range dependencies with a computational cost that scales quadratically with sequence length, but without the sequential bottlenecks of RNNs.

## The Mathematics Behind Self‑Attention

### 1. Core Ingredients: Q, K, V

In a Transformer layer each token (or patch, word, etc.) is first projected into three vectors:

| Symbol | Meaning | Shape (for a single token) |
|--------|---------|----------------------------|
| **Q**  | Query   | \(d_k\) |
| **K**  | Key     | \(d_k\) |
| **V**  | Value   | \(d_v\) |

For a sequence of \(n\) tokens we stack these vectors into matrices:

\[
\mathbf{Q} \in \mathbb{R}^{n \times d_k},\qquad
\mathbf{K} \in \mathbb{R}^{n \times d_k},\qquad
\mathbf{V} \in \mathbb{R}^{n \times d_v}
\]

These matrices are obtained by linear projections of the input embeddings \(\mathbf{X}\in\mathbb{R}^{n\times d_{\text{model}}}\):

\[
\mathbf{Q}= \mathbf{X}\mathbf{W}_Q,\quad
\mathbf{K}= \mathbf{X}\mathbf{W}_K,\quad
\mathbf{V}= \mathbf{X}\mathbf{W}_V,
\]

where \(\mathbf{W}_Q,\mathbf{W}_K\in\mathbb{R}^{d_{\text{model}}\times d_k}\) and \(\mathbf{W}_V\in\mathbb{R}^{d_{\text{model}}\times d_v}\) are learned weight matrices.

---

### 2. Scaled Dot‑Product Attention

The attention scores are the (scaled) dot products between every query and every key:

\[
\mathbf{S}= \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} \in \mathbb{R}^{n \times n}.
\]

* **Why the scaling?**  
  The dot product of two random vectors of dimension \(d_k\) has variance \(d_k\). Dividing by \(\sqrt{d_k}\) keeps the softmax input in a stable range, preventing vanishing/exploding gradients.

Next we convert scores into a probability distribution with the softmax applied row‑wise:

\[
\mathbf{A}= \operatorname{softmax}(\mathbf{S})\quad
\bigl[A_{ij}= \frac{e^{S_{ij}}}{\sum_{k=1}^{n}e^{S_{ik}}}\bigr].
\]

Finally, the weighted sum of values yields the output of the attention head:

\[
\mathbf{O}= \mathbf{A}\mathbf{V}\in\mathbb{R}^{n\times d_v}.
\]

---

### 3. Step‑by‑Step Example

Consider a tiny sequence of **3** tokens with an embedding dimension \(d_{\text{model}}=4\).  
We choose a head size \(d_k=d_v=2\) for illustration.

#### 3.1 Input embeddings

\[
\mathbf{X}= \begin{bmatrix}
1 & 0 & 1 & 0 \\   % token 1
0 & 1 & 0 & 1 \\   % token 2
1 & 1 & 0 & 0      % token 3
\end{bmatrix}
\quad (3\times4)
\]

#### 3.2 Projection matrices (randomly initialized)

\[
\mathbf{W}_Q = \begin{bmatrix}
0.1 & 0.2\\
0.0 & 0.1\\
0.2 & -0.1\\
-0.1 & 0.0
\end{bmatrix},
\quad
\mathbf{W}_K = \begin{bmatrix}
0.0 & 0.1\\
0.1 & 0.0\\
-0.1 & 0.2\\
0.2 & -0.2
\end{bmatrix},
\quad
\mathbf{W}_V = \begin{bmatrix}
0.2 & -0.1\\
0.1 & 0.2\\
-0.2 & 0.0\\
0.0 & 0.1
\end{bmatrix}
\]

#### 3.3 Compute Q, K, V

\[
\mathbf{Q}= \mathbf{X}\mathbf{W}_Q = 
\begin{bmatrix}
0.1 & 0.4\\
0.1 & 0.0\\
0.0 & 0.2
\end{bmatrix},
\qquad
\mathbf{K}= \mathbf{X}\mathbf{W}_K = 
\begin{bmatrix}
0.0 & 0.0\\
0.1 & -0.2\\
0.1 & 0.1
\end{bmatrix},
\qquad
\mathbf{V}= \mathbf{X}\mathbf{W}_V = 
\begin{bmatrix}
0.2 & 0.0\\
0.0 & 0.3\\
0.2 & -0.1
\end{bmatrix}
\]

#### 3.4 Scaled dot‑product scores  

\(d_k = 2\Rightarrow \sqrt{d_k}= \sqrt{2}\approx1.414\)

\[
\mathbf{S}= \frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{2}}=
\frac{1}{1.414}
\begin{bmatrix}
0.1\cdot0.0+0.4\cdot0.0 & 0.1\cdot0.1+0.4\cdot(-0.2) & 0.1\cdot0.1+0.4\cdot0.1\\
0.1\cdot0.0+0.0\cdot0.0 & 0.1\cdot0.1+0.0\cdot(-0.2) & 0.1\cdot0.1+0.0\cdot0.1\\
0.0\cdot0.0+0.2\cdot0.0 & 0.0\cdot0.1+0.2\cdot(-0.2) & 0.0\cdot0.1+0.2\cdot0.1
\end{bmatrix}
=
\begin{bmatrix}
0.0 & -0.212 & 0.354\\
0.0 & 0.071 & 0.071\\
0.0 & -0.283 & 0.141
\end{bmatrix}
\]

#### 3.5 Softmax over rows  

\[
\mathbf{A}= \operatorname{softmax}(\mathbf{S})=
\begin{bmatrix}
0.306 & 0.252 & 0.442\\
0.332 & 0.334 & 0.334\\
0.307 & 0.219 & 0.474
\end{bmatrix}
\]

*(e.g., for the first row:  
\(e^{0}=1,\; e^{-0.212}=0.809,\; e^{0.354}=1.424\);  
normalize by their sum \(1+0.809+1.424=3.233\).)*

#### 3.6 Weighted sum of values  

\[
\mathbf{O}= \mathbf{A}\mathbf{V}=
\begin{bmatrix}
0.306 & 0.252 & 0.442\\
0.332 & 0.334 & 0.334\\
0.307 & 0.219 & 0.474
\end{bmatrix}
\begin{bmatrix}
0.2 & 0.0\\
0.0 & 0.3\\
0.2 & -0.1
\end{bmatrix}
=
\begin{bmatrix}
0.146 & -0.044\\
0.133 & 0.000\\
0.154 & -0.056
\end{bmatrix}
\]

\(\mathbf{O}\) is the **output of one attention head**. In a multi‑head setting we repeat the whole process with different \(\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V\) and finally concatenate the heads, followed by a linear projection back to \(d_{\text{model}}\).

---

### 4. Key Takeaways

| Step | What happens? |
|------|---------------|
| **Projection** | Input → Q, K, V via learned linear maps. |
| **Score computation** | Dot‑product of Q and K, scaled by \(\sqrt{d_k}\). |
| **Normalization** | Row‑wise softmax → attention weights that sum to 1. |
| **Aggregation** | Weighted sum of V using the attention weights. |
| **Result** | Each token now carries information from the whole sequence, weighted by relevance. |

Understanding this pipeline demystifies the “black box” of self‑attention and provides a solid foundation for experimenting with variations (e.g., different scaling, bias terms, or relative positional encodings).

## Self‑Attention in Transformer Architectures

Self‑attention is the core building block that gives Transformers their remarkable ability to model long‑range dependencies. In practice it is never used in isolation – it is **stacked**, **split into multiple heads**, and **woven into the encoder‑decoder pipeline**. Below we unpack each of these steps and show why they unlock massive parallelism.

### 1. From a Single Self‑Attention Layer to Multi‑Head Attention  

| Step | What happens | Why it matters |
|------|--------------|----------------|
| **Linear projections** | The input sequence \(X \in \mathbb{R}^{T \times d_{\text{model}}}\) is multiplied by three learned weight matrices to obtain queries \(Q\), keys \(K\), and values \(V\). | Allows the model to ask “what should I attend to?” (queries) and “what information is available?” (keys/values). |
| **Scaled dot‑product** | Attention scores are computed as \(\text{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V\). | The scaling factor \(\sqrt{d_k}\) stabilizes gradients; the softmax yields a distribution over positions. |
| **Head splitting** | The projection dimension \(d_{\text{model}}\) is divided into \(h\) smaller sub‑spaces: \(d_k = d_v = d_{\text{model}}/h\). Each sub‑space runs its own self‑attention computation in parallel, producing \(h\) “heads”. | Each head can specialize (e.g., focusing on syntax, semantics, positional patterns) while keeping the computational cost linear in \(h\). |
| **Concatenation & final linear** | The \(h\) head outputs are concatenated \(\in \mathbb{R}^{T \times d_{\text{model}}}\) and passed through a final linear layer. | Re‑integrates the diverse information into a single representation for the next layer. |

**Result:** Multi‑head attention preserves the expressive power of a single, very high‑dimensional attention while remaining computationally efficient and easier to train.

### 2. Role in the Encoder‑Decoder Framework  

```
Encoder Stack                Decoder Stack
┌───────────────┐            ┌───────────────┐
│ Self‑Attention│            │ Masked Self‑  │
│ + Feed‑Forward│  ─────►    │ Attention     │
└───────────────┘            │ + Feed‑Forward│
        │                    └───────▲───────┘
        │                            │
        ▼                            │
Cross‑Attention (K,V from Encoder) │
        ▲                            │
        └────────────────────────────┘
```

1. **Encoder layers**: Each layer applies *self‑attention* (full‑sequence) followed by a position‑wise feed‑forward network. The output is a set of context‑rich token embeddings that already incorporate information from the entire input sentence.

2. **Decoder layers**:  
   - **Masked self‑attention** prevents a position from attending to future tokens, preserving autoregressive generation.  
   - **Cross‑attention** (sometimes called “encoder‑decoder attention”) treats the encoder’s final hidden states as keys and values while the decoder’s own hidden states act as queries. This bridges the source and target sequences, letting the decoder “look up” relevant source information for each generated token.

3. **Stacking**: Both encoder and decoder consist of *N* identical layers (commonly 6 or 12). Stacking deepens the model’s capacity to capture hierarchical patterns—early layers may attend to local relations, while deeper layers capture more abstract, long‑range dependencies.

### 3. Why Self‑Attention Enables Parallel Processing  

| Traditional RNN/CNN | Transformer (Self‑Attention) |
|---------------------|------------------------------|
| **Sequential hidden state update** – each time step depends on the previous one. | **All tokens processed simultaneously** – queries, keys, and values are computed in one matrix multiplication. |
| **Limited receptive field** (unless many layers). | **Direct pairwise interactions** – every token can attend to every other token in a single layer. |
| **GPU under‑utilization** – time‑step dependency forces many small operations. | **GPU/TPU friendly** – large dense matrix multiplications map efficiently to hardware accelerators. |
| **Gradient propagation over many steps** → vanishing/exploding issues. | **Shorter effective path** – gradients flow through a few matrix ops, stabilizing training. |

Because the attention scores are obtained via batched matrix multiplications (\(QK^{\top}\) and the subsequent weighted sum), the whole sequence can be processed in **O(T²·d)** time but with **O(1)** depth, allowing the model to fully exploit parallel hardware. The only sequential element is the autoregressive sampling at inference time, not the internal computation of each layer.

---

**Takeaway:** In Transformers, self‑attention is transformed into multi‑head attention, woven into encoder‑decoder stacks, and executed with massive parallelism. This design gives the model both the flexibility to learn diverse relational patterns and the efficiency to train on modern GPUs/TPUs at scale.

## Real‑World Applications

- **Natural Language Processing (NLP)**
  - **BERT & Variants** – Leverages bidirectional self‑attention to capture context from both left and right, powering tasks like question answering, sentiment analysis, and named‑entity recognition.
  - **GPT Series** – Autoregressive self‑attention enables fluent text generation, code synthesis, and conversational agents, scaling up to billions of parameters for few‑shot learning.

- **Computer Vision**
  - **Vision Transformers (ViT)** – Treats image patches as tokens, applying the same self‑attention mechanisms that excel in language to achieve state‑of‑the‑art image classification, object detection, and segmentation without convolutional inductive bias.

- **Speech & Audio Processing**
  - **Speech Transformers** – Self‑attention models capture long‑range temporal dependencies in raw audio or spectrograms, improving automatic speech recognition, speaker diarization, and voice synthesis.

- **Emerging Domains**
  - **Protein Folding & Bioinformatics** – Models like AlphaFold use attention to model interactions between amino‑acid residues, predicting 3‑D structures from sequence data and accelerating drug discovery.
  - **Multimodal Foundations** – Cross‑modal attention bridges text, image, and audio streams, enabling tasks such as video captioning, visual question answering, and robotics perception.

## Advantages, Limitations, and Future Directions

### Advantages  

- **Long‑range dependency modeling**  
  Self‑attention can directly relate any pair of tokens, regardless of their distance in the sequence. This eliminates the vanishing‑gradient problem of recurrent nets and enables the model to capture global context in a single layer.  

- **Full parallelism**  
  Unlike RNNs, the attention matrix is computed from all tokens simultaneously, allowing GPUs/TPUs to process entire sequences in parallel. Training speed therefore scales with hardware throughput rather than sequence length.  

- **Dynamic, content‑based weighting**  
  Each token decides which other tokens are relevant via learned similarity scores, yielding flexible, data‑driven receptive fields that adapt to the task at hand.  

- **Transferability across modalities**  
  The same attention formulation works for text, images, audio, and graph data, making it a universal building block for multimodal architectures.  

### Limitations  

- **Quadratic time and memory complexity**  
  Computing the full attention matrix requires \(O(N^2)\) operations and storage for a sequence of length \(N\). For long documents, videos, or high‑resolution images this quickly becomes prohibitive.  

- **Memory‑bound bottlenecks**  
  The dominant cost is often the intermediate \(QK^\top\) matrix, which can exceed GPU memory limits even when the model itself is modest in size.  

- **Lack of explicit locality bias**  
  Since every token attends to every other token, the model may waste capacity on irrelevant long‑range interactions, especially in tasks where local patterns dominate.  

- **Difficulty with extremely long contexts**  
  Even with efficient hardware, sequences longer than a few tens of thousands of tokens remain challenging for vanilla self‑attention.  

### Future Directions  

| Research Trend | Core Idea | Expected Benefit |
|----------------|----------|------------------|
| **Sparse attention** (e.g., Longformer, BigBird) | Restrict attention to a subset of tokens (local windows, global tokens, random patterns). | Reduces complexity to \(O(N\log N)\) or linear while preserving most long‑range information. |
| **Linear‑complexity attention** (e.g., Performer, Linformer, FAVOR+) | Approximate the softmax kernel with low‑rank or kernel‑based tricks, turning the matrix multiplication into a series of linear operations. | Achieves true \(O(N)\) time and memory, enabling truly long sequences. |
| **Routing‑based mechanisms** (e.g., Routing Transformer, Reformer) | Dynamically select which keys/values each query attends to via learned routing or reversible layers. | Further cuts computation and introduces inductive bias for hierarchical structures. |
| **Hybrid models** (combining convolution, recurrence, or memory modules with attention) | Use convolutional or recurrent layers to capture local patterns, reserving attention for global interactions. | Balances efficiency with expressive power, especially for vision and speech. |
| **Memory‑augmented attention** (e.g., Retrieval‑augmented Transformers) | Offload long‑term knowledge to an external datastore accessed via attention‑like queries. | Allows effectively unbounded context without exploding GPU memory. |
| **Hardware‑aware kernels** | Design attention kernels that exploit sparsity, low‑precision arithmetic, or specialized accelerators (e.g., NVIDIA’s TensorRT, custom ASICs). | Bridges the gap between algorithmic advances and real‑world deployment speed. |

Collectively, these avenues aim to retain the **expressive strength** of self‑attention—its ability to model arbitrary dependencies and parallelize computation—while **taming its quadratic resource demands**. As research converges on efficient, scalable attention mechanisms, we can expect Transformers to become the default backbone for ever larger and more diverse data modalities.

## Hands‑On Implementation in PyTorch

Below is a minimal, from‑scratch implementation of a **self‑attention** layer, a step‑by‑step walkthrough of its forward pass, and an example of plugging it into a tiny model.

---  

### 1️⃣ Self‑Attention Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    Simple self‑attention module (single head).
    Input shape: (batch, seq_len, embed_dim)
    Output shape: (batch, seq_len, embed_dim)
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

        # Linear projections for queries, keys and values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Optional output projection (often used in transformers)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None):
        """
        x   : Tensor of shape (B, T, D)
        mask: Optional bool Tensor of shape (B, T) where True indicates padding.
        """
        B, T, D = x.size()

        # 1️⃣ Project inputs
        Q = self.q_proj(x)   # (B, T, D)
        K = self.k_proj(x)   # (B, T, D)
        V = self.v_proj(x)   # (B, T, D)

        # 2️⃣ Compute scaled dot‑product attention scores
        #    (B, T, D) @ (B, D, T) -> (B, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(D, dtype=torch.float32))

        # 3️⃣ Apply mask (if provided) – set padded positions to -inf
        if mask is not None:
            mask = mask.unsqueeze(1)               # (B, 1, T)
            scores = scores.masked_fill(mask, float('-inf'))

        # 4️⃣ Softmax over the key dimension
        attn_weights = F.softmax(scores, dim=-1)   # (B, T, T)

        # 5️⃣ Weighted sum of values
        context = torch.matmul(attn_weights, V)    # (B, T, D)

        # 6️⃣ Final linear projection
        out = self.out_proj(context)               # (B, T, D)

        return out, attn_weights
```

**Forward‑pass walk‑through**

| Step | Operation | Shape |
|------|-----------|-------|
| Input | `x` | `(B, T, D)` |
| Linear projections | `Q, K, V = proj(x)` | `(B, T, D)` each |
| Scores | `Q·Kᵀ / √D` | `(B, T, T)` |
| (Optional) Mask | `scores.masked_fill` | `(B, T, T)` |
| Softmax | `α = softmax(scores)` | `(B, T, T)` |
| Context | `α·V` | `(B, T, D)` |
| Output projection | `out = W_o(context)` | `(B, T, D)` |

The returned `attn_weights` can be visualised to see which tokens attend to which others.

---  

### 2️⃣ Tiny Model Using the Layer

```python
class SimpleClassifier(nn.Module):
    """
    A toy classifier that:
    1. Embeds token IDs
    2. Applies a single self‑attention block
    3. Pools over the sequence (mean)
    4. Classifies into `num_classes`
    """
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn  = SelfAttention(embed_dim)
        self.fc    = nn.Linear(embed_dim, num_classes)

    def forward(self, token_ids, mask=None):
        """
        token_ids: LongTensor (B, T)
        mask     : BoolTensor (B, T) where True = padding (optional)
        """
        x = self.embed(token_ids)                # (B, T, D)
        x, _ = self.attn(x, mask)                # (B, T, D)

        # Simple mean‑pooling over non‑padded tokens
        if mask is not None:
            lengths = (~mask).sum(dim=1, keepdim=True)  # (B, 1)
            summed = (x * (~mask).unsqueeze(-1).float()).sum(dim=1)
            pooled = summed / lengths.clamp(min=1)
        else:
            pooled = x.mean(dim=1)               # (B, D)

        logits = self.fc(pooled)                 # (B, num_classes)
        return logits
```

**Usage example**

```python
batch_size, seq_len = 4, 10
vocab_size, embed_dim, n_classes = 5000, 64, 3

model = SimpleClassifier(vocab_size, embed_dim, n_classes)

# Dummy data
tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
pad_mask = tokens == 0                     # assume 0 is the padding id

logits = model(tokens, mask=pad_mask)      # (4, 3)
pred   = logits.argmax(dim=-1)             # class predictions
print(pred)
```

---  

### 3️⃣ Why This Works

* **Linear projections** create separate query, key, and value spaces.  
* **Scaled dot‑product** yields a similarity matrix; scaling by √D stabilises gradients.  
* **Masking** prevents attention from looking at padded tokens.  
* **Softmax** turns similarities into a probability distribution, i.e., attention weights.  
* **Weighted sum** aggregates information from all positions, enabling each token to “see” the whole sequence.  

With just a few lines of code you now have a functional self‑attention block that can be stacked, multi‑headed, or combined with positional encodings to build full‑scale transformer models. Happy experimenting!
