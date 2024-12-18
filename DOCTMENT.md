# Kala_Quantum Full Module Documentation

**Kala_Quantum** is a hybrid quantum-classical framework designed to handle astronomical computations, such as predicting the Sun and Moon's positions using Julian Date calculations. It introduces quantum-inspired techniques for operations, training, and simulations.

This document provides a complete overview of the **codebase**, including the structure, components, and how each module works.

---

## **1. Overview of the Project**
Kala_Quantum is designed to:
- Calculate **Sun** and **Moon** positions using positional astronomy formulas.
- Leverage quantum-inspired mechanisms for data training and predictions.
- Simulate quantum states and gates for advanced operations.
- Train models to process large-scale data over **9000 years** (or longer).
- Store predictions in structured JSON format for further use.

---

## **2. Directory Structure**
```
Kala_Quantum/
├── __init__.py             # Initialize the package
├── datasets.py             # Dataset preparation for training
├── models.py               # Classical and hybrid quantum-classical models
├── quantum_core.py         # Core quantum state and gates implementation
├── quantum_layer.py        # Quantum layers for model integration
├── tokenizer.py            # Tokenization utility for code and text data
├── train.py                # Training utilities for the models
├── tests/
│   ├── demo.py             # Script to compute Sun and Moon positions
│   └── main.py             # Script to train the hybrid model
└── build/                  # Build artifacts for packaging
```

---

## **3. Module Breakdown**

### 3.1 **`quantum_core.py`**
This module implements the **quantum state** and essential gates like Hadamard, Pauli-X, and CNOT.

#### Key Components:
- **`QuantumState` Class**: Represents the quantum state for N qubits.
    - **State Initialization**: Initializes the quantum state to |0⟩.
    - **Gate Application**: Apply single-qubit and multi-qubit gates.
    - **Measurement**: Simulates quantum measurements based on probabilities.
- **Quantum Gates**:
    - `hadamard()`: Creates a Hadamard gate for superposition.
    - `pauli_x()`: Implements a Pauli-X gate (bit-flip gate).
    - `cnot()`: Implements a 2-qubit controlled NOT gate.

#### Example Usage:
```python
from Kala_Quantum.quantum_core import QuantumState, hadamard

# Initialize quantum state
qs = QuantumState(num_qubits=2)
qs.apply_gate(hadamard(), qubit=0)
qs.measure()
```

---

### 3.2 **`tokenizer.py`**
This module provides a simple tokenizer for preparing datasets.

#### Key Components:
- **`SimpleTokenizer` Class**:
    - Builds a vocabulary from code or text samples.
    - Tokenizes inputs into numerical representations.
    - Decodes numerical tokens back to readable strings.

#### Example Usage:
```python
from Kala_Quantum.tokenizer import SimpleTokenizer

# Initialize and build tokenizer
tokenizer = SimpleTokenizer()
tokenizer.build_vocab(["def example(): print('Hello')"], vocab_size=1000)

# Tokenize code
tokens = tokenizer("print('Hello')")
print(tokens)
```

---

### 3.3 **`datasets.py`**
This module prepares datasets for training models.

#### Key Components:
- **`CodeDataset` Class**:
    - Loads a JSON file containing code snippets.
    - Tokenizes and prepares inputs for training.
    - Returns tensors for PyTorch models.

#### Example Dataset:
```json
[
  {
    "topic": "Functions",
    "subtopics": [
      {"name": "Basic Function", "code": "def greet(name): print(f'Hello, {name}')"}
    ]
  }
]
```

---

### 3.4 **`models.py`**
This module defines the classical and hybrid models.

#### Key Components:
- **`CodeLanguageModel` Class**:
    - A classical LSTM-based model for code prediction.
- **`HybridModel` Class**:
    - Extends the classical model by integrating quantum outputs using a **QuantumLayer**.

#### Example:
```python
from Kala_Quantum.models import CodeLanguageModel, HybridModel

# Initialize classical and hybrid models
classical_model = CodeLanguageModel(vocab_size=1000)
hybrid_model = HybridModel(classical_model, num_qubits=4)
```

---

### 3.5 **`train.py`**
This module provides the **training loop** for both classical and hybrid models.

#### Key Function:
- **`train_model` Function**:
    - Loads the dataset and tokenizer.
    - Trains either a classical model or a hybrid model.
    - Saves the trained model.

#### Example:
```python
from Kala_Quantum.train import train_model
from Kala_Quantum.tokenizer import SimpleTokenizer

# Train the model
train_model("dataset.json", tokenizer, classical_model, num_qubits=4, epochs=10)
```

---

### 3.6 **`tests/demo.py`**
This script calculates **Sun** and **Moon** positions using Julian Date calculations and normalizes the output.

#### Key Steps:
1. **Julian Date Calculation**:
   - `julian_date()` computes the Julian date.
2. **Sun Position**:
   - `sun_position()` computes the Sun's longitude.
3. **Moon Position**:
   - `moon_position()` computes the Moon's longitude and latitude.
4. **Quantum Encoding**:
   - Data is encoded using the **QuantumState** class.
5. **Training and Storage**:
   - Outputs are stored in `sun_moon_positions.json`.

#### Example:
```python
python Kala_Quantum/tests/demo.py
```
---

## **4. Key Algorithms and Formulas**

### Julian Date Formula:
\[ JD = \text{Integer part of } 365.25 \times (\text{Year} + 4716) + 30.6001 \times (\text{Month} + 1) + \text{Day} \]

### Sun Position:
\[
\lambda_{\text{sun}} = L + 1.915 \cdot \sin(g) + 0.020 \cdot \sin(2g)
\]

### Moon Position:
\[
\lambda_{\text{moon}} = L + 6.289 \cdot \sin(M), \quad \beta_{\text{moon}} = 5.128 \cdot \sin(F)
\]

---

## **5. JSON Output Format**
The results are stored in a JSON file:

```json
[
  {
    "julian_date": 2451545.00001,
    "sun": {
      "longitude": 280.5,
      "latitude": 0,
      "normalized": [0.779, 0.5]
    },
    "moon": {
      "longitude": 134.7,
      "latitude": 5.1,
      "normalized": [0.374, 0.528]
    }
  }
]
```

---

## **6. How It Works**
1. Calculate Julian Date for each second over the required timeframe.
2. Predict Sun and Moon positions using classical astronomy formulas.
3. Normalize data for quantum processing.
4. Encode normalized data into quantum states and apply gates.
5. Simulate training and store measurements.
6. Save outputs to JSON for further analysis.

---

## **7. How to Run**
- Train the model:
    ```bash
    python Kala_Quantum/tests/main.py
    ```
- Predict Sun and Moon positions:
    ```bash
    python Kala_Quantum/tests/demo.py
    ```

---

## **8. Conclusion**
**Kala_Quantum** demonstrates the power of combining classical astronomy and quantum-inspired computation. It predicts Sun and Moon positions efficiently and integrates quantum state simulation for training and encoding large-scale data.

---

## **9. Future Scope**
- Add real quantum hardware integration.
- Extend support for other astronomical bodies.
- Improve quantum encoding for higher precision.

---

## **10. References**
- Julian Date Calculation: https://en.wikipedia.org/wiki/Julian_day
- Positional Astronomy Formulas
- Quantum Mechanics Basics

# Quantum Gates and Operations

This document provides an overview of quantum gates implemented in the `QuantumState` class. Quantum gates are fundamental building blocks of quantum circuits. Below, you’ll find definitions and graphical representations of these gates.

---

## 1. Single-Qubit Gates

### Hadamard Gate (H)
The Hadamard gate creates superposition:
\[
H = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
\]

- **Effect:** Transforms \(|0\rangle \rightarrow \frac{|0\rangle + |1\rangle}{\sqrt{2}}\), and \(|1\rangle \rightarrow \frac{|0\rangle - |1\rangle}{\sqrt{2}}\).

---

### Pauli Gates

1. **Pauli-X Gate (NOT Gate)**:
\[
X = 
\begin{bmatrix}
0 & 1 \\
1 & 0
\end{bmatrix}
\]
- **Effect:** Flips \(|0\rangle \leftrightarrow |1\rangle\).

2. **Pauli-Y Gate**:
\[
Y = 
\begin{bmatrix}
0 & -i \\
i & 0
\end{bmatrix}
\]
- **Effect:** Combines a bit-flip and a phase-flip.

3. **Pauli-Z Gate**:
\[
Z = 
\begin{bmatrix}
1 & 0 \\
0 & -1
\end{bmatrix}
\]
- **Effect:** Flips the phase of \(|1\rangle\).

---

### Phase and T Gates

1. **Phase Gate (S)**:
\[
S = 
\begin{bmatrix}
1 & 0 \\
0 & i
\end{bmatrix}
\]

2. **T Gate**:
\[
T = 
\begin{bmatrix}
1 & 0 \\
0 & e^{i\pi/4}
\end{bmatrix}
\]

---

## 2. Multi-Qubit Gates

### CNOT Gate (Controlled-X)
\[
CNOT = 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 \\
0 & 0 & 1 & 0
\end{bmatrix}
\]

- **Effect:** Flips the target qubit if the control qubit is \(|1\rangle\).

---

### Swap Gate
\[
SWAP = 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\]

- **Effect:** Swaps the states of two qubits.

---

### Toffoli Gate (CCNOT)
\[
TOFFOLI = 
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
\end{bmatrix}
\]

- **Effect:** A NOT operation on the target qubit if both control qubits are \(|1\rangle\).

---

### Fredkin Gate (CSWAP)
\[
FREDKIN = 
\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
\]

- **Effect:** Swaps the two target qubits if the control qubit is \(|1\rangle\).

---

## Visualization

- Single-qubit gates operate on a single state vector.
- Multi-qubit gates act on combined Hilbert spaces using the Kronecker product.

This README explains the mathematical definitions and matrix representations of gates. For examples of usage, see the provided code.


---

## **11. Contact**
For questions, suggestions, or contributions:
- **N V R K SAI KAMESH YADAVALLI**: saikamesh.y@gmail.com
- **GitHub**: [https://github.com/Kalasaikamesh944/Kala_Quantum](https://github.com/Kalasaikamesh944/Kala_Quantum)
