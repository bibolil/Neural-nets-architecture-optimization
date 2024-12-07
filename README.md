# Genetic Algorithm for Optimizing Neural Network Architecture

This project uses a **genetic algorithm (GA)** to optimize the architecture of neural networks by deciding which layers to freeze or train. It is inspired by **transfer learning**, where pre-trained models (e.g., **BERT**, **ResNet**) are fine-tuned for specific tasks by freezing certain layers.

---

## Key Idea

When fine-tuning large models like **BERT**, selecting which layers to freeze is critical for balancing performance and efficiency. Instead of manually choosing, this algorithm automates the process:

- **Genes**: Represent each layer as `1` (frozen) or `0` (trainable).
- **Fitness**: Measures the model's validation accuracy with a specific layer configuration.
- **Evolution**: Through **selection**, **crossover**, and **mutation**, the algorithm finds the best setup over multiple generations.

---

## How It Works

1. **Initialize Population**: Start with random layer configurations.
2. **Evaluate Fitness**: Train a model with each configuration and calculate accuracy.
3. **Selection**: Pick the best-performing configurations (chromosomes) to reproduce.
4. **Crossover**: Combine configurations from two parents to create new ones.
5. **Mutation**: Randomly flip layer states (`1` ↔ `0`) for diversity.
6. **Evolve**: Repeat for a fixed number of generations to find the optimal setup.

---

## Why Use It?

- **Automates Layer Freezing**: No need to guess which layers to train or freeze.
- **Improves Performance**: Systematically finds the best configuration for the task.
- **Works Across Models**: Applicable to models like **BERT**, **GPT**, **ResNet**, or custom architectures.

---

## Example Use Cases

1. **Natural Language Processing**:
   - Fine-tune pre-trained models like **BERT** for tasks like sentiment analysis or text classification.
2. **Computer Vision**:
   - Optimize layer freezing in **ResNet** for image classification.
3. **Custom Architectures**:
   - Combine convolutional and dense layers and let the GA decide the best freezing strategy.

---

## Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/genetic-algorithm-layer-freezing.git
   cd genetic-algorithm-layer-freezing
2. Clone the repository:
   ```bash
   pip install -r requirements.txt
3. Run the main program:
   ```bash
   python3 main.py
4. Output example:
   ```bash
   === Generation 1 ===
    Fittest chromosome: [0, 1, 0] | Fitness: 0.85
    
    === Generation 5 ===
    Fittest chromosome: [1, 0, 0] | Fitness: 0.92
