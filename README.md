# Deep Learning Project – Open Set Recognition  
This repository contains my final project for the Deep Learning course. The project explores **Open Set Recognition (OSR)** using PyTorch. The goal is to train a classifier capable of recognising digits from the MNIST dataset while detecting out-of-distribution (OOD) samples from CIFAR ‑10 as unknown.  
  
## 📝 Project Report  
The full project report is provided in `DL_final_project.pdf`. It includes detailed description of the problem, methodology, experiments, results, and conclusions. The PDF also contains figures and diagrams illustrating the dataset distribution, model architecture, and training curves. Since GitHub's preview may not show images properly, please open the PDF directly to view them.  
  
## 🤔 Summary  
- **Open Set Recognition:** Unlike standard classifiers, OSR models must not only classify known classes correctly but also identify unknown samples. In this project, the known classes are digits **0–9** from the MNIST dataset, and unknown samples are random classes from CIFAR ‑10.  
- **Combined Dataset:** `project_utils.py` defines a `CombinedDataset` class that merges MNIST and OOD datasets. It returns label `10` for OOD samples so the model can learn to separate unknown examples.  
- **Evaluation Metrics:** The `eval_model` function measures accuracy on MNIST, OOD sets, and combined accuracy. This helps assess how well the model distinguishes between known and unknown classes.  
  
## 📁 Repository Contents  
| Path | Description |  
| --- | --- |  
| `DL_Final_Project.ipynb` | Jupyter notebook containing data loading, model definition, training loops, and evaluation. Open this notebook to reproduce the experiments. |  
| `DL_final_project.pdf` | Project report with explanations, figures, and results. |  
| `project_utils.py` | Utility functions and classes, including the `CombinedDataset` and evaluation function for OSR. |  
| `osr_model_epoch_30.pth` | Saved PyTorch model weights after training for 30 epochs. Load this file to evaluate the pre‑trained model. |  
| `.DS_Store` | macOS directory metadata file (can be ignored). |
## 🛠️ How to Run  
1. **Clone** the repository:  
   ```bash  
   git clone https://github.com/MatanGoldfarB/DeepLearning-Project  
   cd DeepLearning-Project  
   ```  
2. **Install dependencies** (assuming Python 3.8+ and pip):  
   ```bash  
   pip install -r requirements.txt  # if a requirements file exists  
   # or manually install torch, torchvision, numpy, matplotlib, etc.  
   ```  
3. **Run the notebook**:  
   - Open `DL_Final_Project.ipynb` in Jupyter Notebook or VS Code.  
   - Execute the cells to load the dataset, train the model, and evaluate OSR performance.  
   - Alternatively, skip training by loading the pre‑trained weights from `osr_model_epoch_30.pth` using the notebook code.  
4. **Evaluate the pre‑trained model**:  
   - Use the functions in `project_utils.py` to load the saved model and call `eval_model` on MNIST and OOD samples.  
  
## 💡 What This Project Demonstrates  
- Implementing open set recognition using PyTorch.  
- Combining multiple datasets and creating custom PyTorch `Dataset` classes.  
- Training convolutional neural networks and evaluating them on known and unknown classes.  
- Using metrics to compare performance on in‑distribution and out‑of‑distribution data.  
- Documenting experiments in Jupyter notebooks and a comprehensive PDF report.  
  
## 📍 Future Work  
- Experiment with different model architectures (e.g., ResNet, WideResNet) to improve OSR performance.  
- Incorporate other OOD datasets to test generalization.  
- Implement advanced OSR techniques such as energy‑based models or outlier exposure.  
- Add charts or images from the PDF report into the README once GitHub allows embedding; currently, please refer to the PDF for visualizations.  
  
## 💌 Contact  
If you're a recruiter or researcher interested in this project, feel free to reach out.  

- **Email:** matangoldfarb1@gmail.com  
- **GitHub:** [MatanGoldfarB](https://github.com/MatanGoldfarB)
