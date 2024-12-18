# Kala_Quantum

**Kala_Quantum** is designed for quantum operations and training, enabling advanced computations and simulations for astronomical data such as the Sun and Moon's longitude and latitude. It integrates classical positional astronomy formulas with quantum computation concepts, allowing precise predictions and data processing over extensive timeframes (e.g., 9,000 years).

## Features
- **Julian Date Calculation**: Compute precise Julian dates for astronomical calculations.
- **Sun and Moon Positioning**: Calculate solar and lunar positions (longitude and latitude).
- **Quantum State Simulation**: Encode and process positional data using quantum states and gates.
- **Quantum Training**: Simulate training processes using hybrid quantum-classical models.
- **Large-scale Data Generation**: Predict Sun and Moon positions for every second over thousands of years.
- **JSON Data Storage**: Save results in a structured JSON format for easy retrieval and analysis.

---

## Installation

### Prerequisites
- **Python 3.8+**
- **NumPy**
- **TQDM** (for progress tracking)

### Install Dependencies
To install dependencies, run:

```bash
pip install numpy tqdm
```

### Clone the Repository
```bash
git clone https://github.com/Kalasaikamesh944/Kala_Quantum.git
cd Kala_Quantum
```

---

## Directory Structure
```
Kala_Quantum/
├── __init__.py
├── datasets.py           # Data handling modules
├── models.py             # Quantum and classical model definitions
├── quantum_core.py       # Quantum state and gate implementations
├── quantum_layer.py      # Quantum-inspired layers for ML
├── tokenizer.py          # Tokenization logic for code and data
├── train.py              # Training utilities
├── tests/
│   ├── demo.py           # Example script for Sun & Moon prediction
│   └── main.py           # Model training script
└── build/                # Build-related files
```

---

## Usage

### 1. Training Quantum Models
Train the model to predict the Sun and Moon's positions for 9,000 years with second-level precision.

```bash
python Kala_Quantum/tests/demo.py
```

### 2. Output
The training results (Sun and Moon positions) are saved in a JSON file:

```
sun_moon_positions.json
```

Sample JSON structure:
```json
[
  {
    "julian_date": 2451545.00001,
    "sun": {
      "longitude": 280.5,
      "latitude": 0,
      "normalized": [0.779, 0.5],
      "measurement": 1
    },
    "moon": {
      "longitude": 134.7,
      "latitude": 5.1,
      "normalized": [0.374, 0.528],
      "measurement": 0
    }
  },
  ...
]
```

### 3. Sun and Moon Position Calculation
You can compute Sun and Moon positions manually using:

```python
from Kala_Quantum.quantum_core import QuantumState, hadamard
from demo import julian_date, sun_position, moon_position

# Example: Calculate positions for January 1, 2024
jd = julian_date(2024, 1, 1)
sun_long, sun_lat = sun_position(jd)
moon_long, moon_lat = moon_position(jd)

print(f"Sun Longitude: {sun_long}, Latitude: {sun_lat}")
print(f"Moon Longitude: {moon_long}, Latitude: {moon_lat}")
```

---

## Key Formulas

### Julian Date
The Julian Date is calculated using:
\[ JD = \text{Integer part of } 365.25 \times (\text{Year} + 4716) + 30.6001 \times (\text{Month} + 1) + \text{Day} + \text{Adjustments} \]

### Sun Position
The Sun's longitude is calculated as:
\[
\lambda_{\text{sun}} = L + 1.915 \cdot \sin(g) + 0.020 \cdot \sin(2g)
\]
Where:
- \( L \): Mean longitude of the Sun
- \( g \): Mean anomaly of the Sun

### Moon Position
The Moon's longitude and latitude are calculated using:
\[
\lambda_{\text{moon}} = L + 6.289 \cdot \sin(M)\
\beta_{\text{moon}} = 5.128 \cdot \sin(F)
\]
Where:
- \( M \): Moon's mean anomaly
- \( F \): Moon's argument of latitude

---

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push to the branch: `git push origin feature/your-feature`.
5. Open a pull request.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements
- Inspired by classical astronomy and quantum computation techniques.
- Thanks to the open-source community for supporting tools like NumPy and TQDM.

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

## Contact
For questions or suggestions, contact:
- **N V R K SAI KAMESH YADAVALLI**: saikamesh.y@gmail.com
- **Project Repository**: [GitHub](https://github.com/Kalasaikamesh944/Kala_Quantum)
