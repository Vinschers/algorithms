# **Vicentin**

A comprehensive Python library for mathematical optimization, deep learning, computer vision, and classic algorithms. This library is designed with a dual-backend architecture, offering seamless switching between **NumPy** for transparency and **PyTorch** for hardware acceleration and automatic differentiation.

[![CI](https://github.com/Vinschers/algorithms/actions/workflows/publish.yml/badge.svg)](https://github.com/Vinschers/algorithms/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
<a href="https://pypi.org/project/vicentin/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/vicentin"></a>

---

## **Table of Contents**

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Pre-commit Setup](#pre-commit-setup)
- [License](#license)

---

## **Introduction**

`vicentin` is a Python package that contains my personal implementations of a variety of algorithms, data structures, and optimization techniques. It serves as a collection of theoretical and practical programming concepts.

---

## **Features**

- **Data Structures**: [Queue](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/data_structures/queue.py), [Stack](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/data_structures/stack.py), [Tree](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/data_structures/tree.py), [Graph](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/data_structures/graph.py), [Heap](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/data_structures/heap.py), [Priority Queue](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/data_structures/priority_queue.py), [Trie](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/data_structures/trie.py), [Union Find](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/data_structures/union_find.py)
- **Dynamic Programming**: [Knapsack](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/dp/knapsack.py), [Matrix Multiplication](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/dp/matrix_multiplication.py), [Rod Cutting](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/dp/rod_cut.py), [Edit Distance](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/dp/str/edit_distance.py)
- **Graph Algorithms**: [Minimum Spanning Tree (MST)](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/graph/mst.py), [Shortest Path](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/graph/shortest_path.py), [Negative Cycle Detection](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/graph/negative_cycle.py)
- **Image & Video Processing**: [Optical Flow (Horn-Schunck)](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/image/video/optical_flow/horn_schunck/horn_schunck.py), [Differentiation](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/image/differentiation/diff.py), [Regularization](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/image/regularization/regularization.py), [Image-to-Graph](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/image/img2graph.py)
- **Optimization**: [Gradient Descent](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/optimization/minimization/gradient_descent/gradient_descent.py), [Newton's Method](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/optimization/minimization/newton_method/newton.py), [Barrier Method](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/optimization/minimization/barrier_method/barrier.py), [Proximal Gradient Descent](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/optimization/minimization/proximal_gradient_descent/proximal_gradient.py), [ISTA](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/optimization/minimization/ista/ista.py), [Projected Gradient Descent](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/optimization/minimization/projected_gradient_descent/projected_gradient.py), [Newton-Raphson (Root Finding)](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/optimization/root_finding/newton_raphson/newton_raphson.py)
- **Deep Learning**:
    - **Models**: [Autoencoders (AE)](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/deep_learning/models/ae.py), [Variational Autoencoders (VAE)](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/deep_learning/models/vae.py)
    - **Trainers**: [Standard](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/deep_learning/train/StandardTrainer.py), [Supervised](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/deep_learning/train/SupervisedTrainer.py), [GAN](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/deep_learning/train/GANTrainer.py), [Distillation](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/deep_learning/train/DistillationTrainer.py)
    - **Losses**: [Beta-VAE](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/deep_learning/loss/BetaVAELoss.py), [Wasserstein GAN](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/deep_learning/loss/WassersteinGANLoss.py)
- **Sorting**: [Heap Sort](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/sorting/heapsort.py)
- **Mathematical Tools**: [Polynomial Operations](https://github.com/Vinschers/algorithms/tree/main/src/vicentin/misc/polynomial.py)

---

## **Installation**

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/your-username/vicentin.git
cd vicentin
```

### **2️⃣ Set Up a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate
```

### **3️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## **Usage**

### Heap data structure

```python
# Example: Using the heap data structure
from vicentin.data_structures.heap import Heap

heap = Heap()
heap.insert(5)
heap.insert(2)
heap.insert(8)

print(heap.extract_min())  # Output: 2
```

### Newton's Method

```python
import torch
from vicentin.optimization.minimization import newton_method

def objective(x):
    return torch.sum(x**2)

x0 = torch.tensor([10.0, 10.0])
A = torch.tensor([[1.0, 1.0]])
b = torch.tensor([1.0])

# Backend (Torch) is automatically detected from x0
x_opt = newton_method(objective, x0, equality=(A, b))
print(f"Optimal solution: {x_opt}")
```

---

## **Pre-commit Setup**

This repository uses `pre-commit` to enforce coding standards, automatic formatting and automatic version bumping before commits.

### **1️⃣ Install `pre-commit`**

```bash
pip install pre-commit
```

### **2️⃣ Install Hooks**

```bash
pre-commit install
```

### 3️⃣ Use [`commitizen`](https://commitizen-tools.github.io/commitizen/) to commit

```bash
cz commit
```

---

## **License**

This project is licensed under the [MIT License](LICENSE).
