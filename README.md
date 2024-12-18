# M1 Coursework
### Keying Song (ks2146)

In this coursework, an inference pipeline for calculating the sum of two handwritten MNIST digits is built,
evaluated and analysed.

## Declaration
No auto-generation tools were used in this coursework except for generation of BibTeX references.

## Project Structure
The main structure of the package `add_mnist` is like:
```
.
├── add_mnist/
│   ├── build_dataset.py      # module for dataset construction
│   ├── compare_models.py     # module for random forest and SVC and comparison
│   ├── linear_classifier.py  # module for two kinds of linear classifier
│   ├── nn_model.py           # module for NN model's building and training
│   ├── save_path.py          # module for path arrangement
│   └── tsne.py               # module for t-SNE on input and embedding layer
|
├── data/                     # the folder to save the dataset *combined_mnist*
├── model/                    # the folder to save the best model parameters
├── result/                   # the folder to save the result figures
|
├── pyproject.toml            # config file
├── README.md                 # readme file
└── main.ipynb                # the main file to answer the questions
```

## Installation

1. Clone the repository:
```bash
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m1_coursework/ks2146.git
```

2. Install:
```bash
pip install -e .
```

3. Use:
After installing, all the classes and functions in package `add_mnist` can be imported and used anywhere on your own machine.
```python
from add_mnist import NewDataset, SavePath, NNModel, CompareModels, load_data, combined_classifier, separate_classifier, Tsne
```

## Usage

The main workflow is demonstrated in `main.ipynb`. The five sections in it address each of the five questions in the coursework.


## Dependencies
- numpy>=1.20
- tensorflow>=2.10
- scikit-learn>=1.0
- matplotlib>=3.5
- optuna>=3.0
