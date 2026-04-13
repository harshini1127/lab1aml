markdown
# Machine Learning Lab Programs

## 📋 Overview

This repository contains implementations of four fundamental machine learning algorithms from scratch using Python.

| # | Algorithm | Type | Dataset |
|---|-----------|------|---------|
| 1 | Find-S | Concept Learning | Tennis (Enjoy Sports) |
| 2 | Candidate Elimination | Version Space Learning | Enjoy Sports |
| 3 | Decision Tree (ID3 with Gain Ratio) | Classification | Tennis |
| 4 | Neural Network | Deep Learning | XOR Problem |

## 📁 Project Structure

```
ml-lab-programs/
│
├── data/
│   ├── tennis.csv # Dataset for Find-S & Decision Tree
│   ├── enjoy_sports.csv # Dataset for Candidate Elimination
│   └── dataset.csv # XOR dataset for Neural Network
│
├── src/
    ├── find_s.py # Find-S algorithm
    ├── candidate_elimination.py # Candidate Elimination
    ├── decision_tree.py # Decision Tree (ID3)
    └── neural_network.py # Neural Network


```

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ml-lab-programs.git
cd ml-lab-programs

# Install dependencies
pip install -r requirements.txt
```

## 📊 Datasets

### tennis.csv (Find-S & Decision Tree)

```csv
Outlook,Temperature,Humidity,Wind,PlayTennis
Sunny,Hot,High,Weak,No
Sunny,Hot,High,Strong,No
Overcast,Hot,High,Weak,Yes
Rain,Mild,High,Weak,Yes
Rain,Cool,Normal,Weak,Yes
Rain,Cool,Normal,Strong,No
Overcast,Cool,Normal,Strong,Yes
Sunny,Mild,High,Weak,No
Sunny,Cool,Normal,Weak,Yes
Rain,Mild,Normal,Weak,Yes
Sunny,Mild,Normal,Strong,Yes
Overcast,Mild,High,Strong,Yes
Overcast,Hot,Normal,Weak,Yes
Rain,Mild,High,Strong,No
```

### enjoy_sports.csv (Candidate Elimination)

```csv
Sky,AirTemp,Humidity,Wind,Water,Forecast,EnjoySport
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Sunny,Warm,High,Strong,Warm,Same,Yes
Rainy,Cold,High,Strong,Warm,Change,No
Sunny,Warm,High,Strong,Cool,Change,Yes
Sunny,Warm,Normal,Strong,Warm,Same,Yes
Rainy,Cold,Normal,Light,Warm,Same,No
Sunny,Warm,Normal,Light,Warm,Same,Yes
Rainy,Cold,Normal,Light,Warm,Change,No
Sunny,Cold,Normal,Light,Warm,Same,No
Sunny,Warm,Normal,Light,Warm,Same,Yes
```

### dataset.csv (Neural Network - XOR)

```csv
x1,x2,y
0,0,0
0,1,1
1,0,1
1,1,0
```
## 💻 Running the Programs

```bash
# Program 1: Find-S
python src/find_s.py

# Program 2: Candidate Elimination
python src/candidate_elimination.py

# Program 3: Decision Tree
python src/decision_tree.py

# Program 4: Neural Network
python src/neural_network.py
```

## 📝 Algorithm Explanations

### 1. Find-S Algorithm

- Initializes hypothesis with first positive example
- Generalizes hypothesis when positive examples differ
- Ignores negative examples
- Time Complexity: O(n)

### 2. Candidate Elimination Algorithm

- Maintains S (most specific) and G (most general) boundaries
- Positive examples: generalize S, prune G
- Negative examples: specialize G
- Time Complexity: O(2^n) worst case

### 3. Decision Tree (ID3 with Gain Ratio)

**Mathematical Formulas:**

```
Entropy(S) = -Σ pᵢ log₂(pᵢ)
Gain(S, A) = Entropy(S) - Σ (|Sᵥ|/|S|) × Entropy(Sᵥ)
GainRatio(S, A) = Gain(S, A) / IntrinsicValue(S, A)
```

### 4. Neural Network (XOR Problem)

**Architecture:**

- Input Layer: 2 neurons
- Hidden Layer: 8 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

**Formulas:**

```
Forward: ŷ = σ(ReLU(X·W¹ + b¹)·W² + b²)
Loss: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

## 📈 Algorithm Comparison
Feature	Find-S	Candidate Elimination	Decision Tree	Neural Network
Output	Single hypothesis	Version space	Tree structure	Weights & biases
Handles negatives	No	     Yes       Yes     	     Yes
Interpretability	High	Medium	  Very High	     Low
Overfitting risk    Low	     Low	   High	        Moderate
Time Complexity	    O(n)	O(2ⁿ)	 O(m·n·log n)	O(epochs·n·h)
Non-linear learning	 No	     No	        Yes           Yes

## 🔧 Requirements

```txt
numpy==1.21.0
pandas==1.3.0
```

## 📚 Theoretical Concepts
Entropy: Measures impurity in data (0 = pure, 1 = maximum impurity)

Information Gain: Reduction in entropy after splitting

Gain Ratio: Information gain normalized by intrinsic value

Backpropagation: Algorithm for training neural networks using gradient descent

## 🐛 Troubleshooting
Issue	Solution
FileNotFoundError	Ensure CSV files are in correct directory
ImportError	Run pip install -r requirements.txt
Convergence issues	Adjust learning rate or increase epochs



