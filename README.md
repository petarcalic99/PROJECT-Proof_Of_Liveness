# PROJECT-Proof_Of_Liveness

**Ledger Innovation Lab Project**

## Mission

This project investigates a **decentralized Proof of Liveness** mechanism — a way to verify that a real human (not a bot) is interacting with a system, without relying on a centralized authority.

The core idea is to combine **lightweight neural networks** with **blockchain verification** (StarkNet/Cairo): a user is presented with a visual challenge (e.g. digit recognition), their response is evaluated by a small CNN, and the inference can be verified on-chain using zero-knowledge proofs. Think of it as a **decentralized, cryptographically verifiable CAPTCHA**.

## Project Phases

### Phase 1 — Research & Literature Review (Complete)

A comprehensive bibliography of 22 papers was assembled, covering:
- Biometric liveness detection & bot detection
- Adversarial attacks on neural networks
- Zero-knowledge proofs (ZKSENSE, Cairo)
- Decentralized authentication & KYC via DLT
- Mouse trajectory analysis and click fraud detection

The research is collected in `Bibliographie/` and synthesized in the project specification documents at the root of the repository.

### Phase 2 — Neural Network Experimentation (Complete)

The main experimental work lives in `Experiments/` and explores how to build the **smallest possible CNN** that still achieves high accuracy — a critical constraint for on-chain verification where memory is expensive.

**Key results:**

| Architecture | Parameters | Accuracy | Epochs |
|---|---:|---:|---:|
| 2 Conv (8 filters, 5x5) + 1 FC (128) | 3,106 | 95% | 7 |
| 2 Conv (6 filters, 5x5) + 1 FC (96) | 2,032 | 94% | 14 |
| 2 Conv (5 filters, 5x5) + 1 FC (80) | 1,570 | 93% | 14 |
| 1 Conv (6 filters, 5x5) + 1 FC (864) | 8,806 | 93% | 12 |
| Full power architecture | 21,840 | 97% | 5 |

**Key insight:** *"On StarkNet, we grow a NN in width not height. Computation is cheap, memory is not."*

The experiments also cover:
- **Robustness to spatial augmentation** — testing models with translated inputs (0.1–0.3 range) to simulate real-world handwriting variation
- **Manual CNN implementation** — hand-coded convolution, ReLU, max pooling, and fully connected layers without framework abstractions, preparing the path toward a pure arithmetic implementation suitable for ZK circuits
- **MLP baseline** — a simple 784-32-10 perceptron with manual weight extraction for inference

### Phase 3 — On-Chain Integration (Next Step)

The natural next phase is to **port the neural network inference into a StarkNet/Cairo smart contract** and wrap it in a zero-knowledge proof. The groundwork is laid:
- The Cairo whitepaper has been studied (`Bibliographie/Cairo-WhitePaper.pdf`)
- Manual CNN inference (no framework dependencies) has been implemented
- Model architectures have been optimized for minimal memory footprint

**Objectives for this phase:**
1. Implement the CNN forward pass in Cairo
2. Design the challenge-response protocol (present image, collect answer, verify on-chain)
3. Integrate ZK proof generation so the liveness check is verifiable without revealing the model weights or user input
4. Benchmark gas costs and latency for different model sizes

### Phase 4 — Productionization (Future)

To move from prototype to a usable system:
- Add a `requirements.txt` or `pyproject.toml` for reproducible environments
- Set up CI/CD for automated testing
- Build a user-facing frontend for the challenge interaction
- Harden the system against adversarial attacks (informed by the bibliography)
- Package the solution for integration with existing dApps

## Repository Structure

```
.
├── Decentralized_Test_Of_Liveness.pdf   # Main project specification
├── PrepDoc-POL.pdf                      # Preparation / planning document
├── Presentation *.pdf                   # Stakeholder presentation (19 pages)
│
├── Experiments/                         # ML notebooks and scripts
│   ├── Cnn_Optimisation.ipynb           #   CNN architecture comparison
│   ├── MNIST_Test.ipynb                 #   Robustness & attack simulation
│   ├── Manual_CNN.ipynb                 #   Hand-coded inference pipeline
│   ├── MNIST_ManualPerceptron (1).ipynb #   MLP with manual weight extraction
│   ├── PyTorchDataSets_Study.ipynb      #   Dataset exploration
│   ├── ImageDB.ipynb                    #   Image loading utilities
│   └── testMLP.py                       #   Standalone MLP training script
│
├── Notebooks/                           # Supplementary notebooks
│
└── Bibliographie/                       # 22 research papers
    ├── ZKSENSE.pdf                      #   ZK biometric sensing
    ├── Cairo-WhitePaper.pdf             #   StarkNet's Cairo language
    ├── RobustKYCviaDLT.pdf              #   KYC via distributed ledger
    └── ...                              #   Adversarial ML, bot detection, etc.
```

## Tech Stack

- **Python** — core language
- **PyTorch / TorchVision** — neural network training and evaluation
- **Jupyter Notebooks** — interactive experimentation
- **NumPy / Matplotlib** — data manipulation and visualization

## Documentation

The project's full technical specification, scope, and presentation are available as PDFs at the root of the repository:
- `Decentralized_Test_Of_Liveness.pdf` — detailed technical document
- `PrepDoc-POL.pdf` — project preparation and planning
- `Presentation DECENTRALIZED TEST OF LIVENESS.pdf` — overview presentation
