# **Vicentin**

*A Python package showcasing algorithms, data structures, and mathematical concepts.*

[![CI](https://github.com/Vinschers/algorithms/actions/workflows/publish.yml/badge.svg)](https://github.com/Vinschers/algorithms/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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
- **Dynamic Programming**: Knapsack, Matrix Chain Multiplication, Rod Cutting
- **Graph Algorithms**: Minimum Spanning Tree (MST), Shortest Path
- **Image Processing**: Optical Flow, Differentiation, Regularization
- **Optimization**: Gradient Descent
- **Sorting**: Heap Sort
- **Mathematical Tools**: PCA, Kernel PCA, Polynomial Operations

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

Import the package and use its functionalities in Python scripts.

```python
import vicentin

# Example: Using the heap data structure
from vicentin.data_structures.heap import Heap

heap = Heap()
heap.insert(5)
heap.insert(2)
heap.insert(8)

print(heap.extract_min())  # Output: 2
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
