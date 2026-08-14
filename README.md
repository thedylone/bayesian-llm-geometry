# Quantifying the Geometry of Concepts: A Two-Stage Bayesian Framework

This codebase provides the tools to systematically evaluate and extract the geometric structure (linear vs. circular) of representations in Large Language Models (LLMs). It includes the mathematical derivations and implementations for our Bayesian Model Comparison (BMC) framework and the Sparse Circular Autoencoder (SCAE).

## 📁 Repository Structure

- `src/`: Core Python modules.
  - `bmc/`: Implementations of the linear and circular generative models, Expectation-Maximization (EM) algorithm, and Laplace approximation.
  - `sae/`: Architecture and training loops for the standard SAE and our novel SCAE.
  - `data/`: Utilities for synthetic data generation and LLM activation extraction, loading, and preprocessing.
  - `vis/`: Visualization utilities for plotting manifolds, training curves, and evaluation metrics.
- `scripts/`: Executable pipeline scripts for end-to-end reproduction of the paper's experiments.
  - `config.py`: Centralized configuration file for hyperparameters and paths.
- `figures/`: Output directory where generated plots and evaluation metrics are saved.

---

## 💻 Compute Resources & Requirements

To fully reproduce the experiments detailed in the paper, the following compute resources are recommended:

- **Hardware:**
  - 1x NVIDIA A100 or V100 GPU (Minimum 16GB VRAM) for extracting Llama-3-8B-Instruct activations and training the autoencoders.
  - Standard multi-core CPU (e.g., 8+ cores, 16GB+ RAM) for the BMC

- **Execution Time:**
  - **Data Extraction:** ~15 minutes.
  - **BMC Fitting:** ~1 minute.
  - **SAE/SCAE Training:** ~2 minutes per concept (K=10 features, 5000 epochs).

- **Storage:** < 200 MB of disk space for saving data, models, and figures. 5 GB for Python packages and dependencies.

---

## ⚙️ Installation

1. Clone this repository and navigate to the root directory:

```bash
git clone https://github.com/thedylone/bayesian-llm-geometry.git
cd bayesian-llm-geometry

```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
pip install -e .

```

**Note on Llama 3:** To run the real LLM experiments, you must request access to the Llama-3-8B-Instruct weights via HuggingFace and provide your HuggingFace token. You must agree to the Meta Llama 3 Community License Agreement.

---

## 🚀 Running the Framework Pipeline

The pipeline is split into logical steps housed within the `./scripts` directory. Execute them in the following order from the repository root.

### Step 1: Data Preparation

You can run the framework on synthetic data or extract actual activations from Llama 3.

`01_extract_activations.py` extracts the activations from the LLM and saves them in a structured format for downstream analysis.

`02_generate_synthetic.py` generates synthetic datasets with known linear and circular structures for validation.

`02a_generate_synthetic_multiple.py` generates multiple synthetic datasets at once.

`03_process_activations.py` processes the extracted activations by subtracting template means (if available) and PCA reducing the data.


### Step 2: Bayesian Model Comparison (BMC)

Fit the linear and circular generative models using the EM algorithm and compute the Bayes Factor via Laplace approximation.

```bash
python scripts/04_run_bmc.py
python scripts/05_plot_bmc.py

```

`04a_run_bmc_synthetic_multiple.py` runs BMC on multiple synthetic datasets at once, but only saves the Bayes Factor metrics, not the fitted manifolds, for getting the mean and standard deviation of the Bayes Factor across multiple runs. (Not meant for plotting.)


### Step 3: Train Sparse Autoencoders

Train both the standard SAE and the novel Sparse Circular Autoencoder (SCAE) on the datasets.

```bash
python scripts/06_train_sae.py
python scripts/07_plot_sae.py

```

`06a_train_sae_synthetic_multiple.py` trains SAE and SCAE on multiple synthetic datasets at once, but only saves the evaluation metrics, not the weights, for getting the mean and standard deviation of the evaluation metrics across multiple runs. (Not meant for plotting.)

---

## 📊 Reproducibility

The hyperparameters for all experiments are centralized in `scripts/config.py`. The results in the paper can be reproduced using the provided hyperparameters.

## 📝 License & Acknowledgements

- The code in this repository is released under the **MIT License**.
- Usage of the Llama 3 model is subject to the **Meta Llama 3 Community License Agreement**.
