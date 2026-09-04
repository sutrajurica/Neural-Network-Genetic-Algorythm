A repository featuring a Streamlit web application and a Neural Network optimized using a Genetic Algorithm.

---

## 🌐 Gazelle Web Application

- **Live Web App:** [gazelle.streamlit.app](https://gazelle.streamlit.app/)
- **Sample Dataset:** [Kaggle - Running Log Insight](https://www.kaggle.com/datasets/jeffreybraun/running-log-insight)

---

## 🧠 Neural Network (Genetic Algorithm Optimization)

> **Note:** All credits for test files, specifications, and assignment instructions go to the **Faculty of Electrical Engineering and Computing (FER)**.

### Example Run

```bash
python solution.py --train sine_train.txt --test sine_test.txt --nn 5s --popsize 10 --elitism 1 --p 0.1 --K 0.1 --iter 10000

| Parameter | Argument | Description |
| :--- | :--- | :--- |
| **Training Data** | `--train` | Path to the training dataset file |
| **Testing Data** | `--test` | Path to the testing dataset file |
| **NN Architecture** | `--nn` | Neural network architecture definition (e.g., `5s`) |
| **Population Size** | `--popsize` | Population size for the genetic algorithm |
| **Elitism** | `--elitism` | Number of elite individuals preserved per generation |
| **Mutation Rate** | `--p` | Mutation probability for each chromosome element |
| **Mutation StdDev** | `--K` | Standard deviation of Gaussian noise applied during mutation |
| **Iterations** | `--iter` | Number of genetic algorithm iterations/generations |
