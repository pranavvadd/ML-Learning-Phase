# Machine Learning Learning — Future AI Engineer Pathway

Welcome to my **Machine Learning Learning Repository**, created as part of the  
**[Udacity x AWS – Introducing Generative AI]** course on the **Future AI Engineer Pathway**.

This repo is not a showcase of original project builds, but a **space for reflection and exploration** as I work through foundational models. I'm documenting how different machine learning techniques work, how I visualized their behavior, and what I’ve learned about the tools and libraries used to implement them.

---

## Why This Repo?

Before building full-scale AI applications, I'm using this phase to:

- Practice reading and running ML code
- Understand supervised vs. unsupervised learning
- Interpret outputs, performance metrics, and visualizations
- Develop a habit of thoughtful documentation
- Prepare myself to apply these tools in real-world scenarios

---

## Projects

### 1. Energy Efficiency Prediction (Regression)

- **Goal:** Predict a building’s energy efficiency from architectural features.
- **Model:** `RandomForestRegressor`
- **What I Learned:**
  - `seaborn.pairplot()` helped me observe that no single feature had a perfectly linear correlation with energy efficiency.
  - Visualizing predictions against actual values showed most points clustered around the diagonal — indicating a good fit.
  - I learned how to evaluate regression models using **Mean Squared Error (MSE)** and understood that lower values mean better performance.

---

### 2. Consumer Purchase Prediction (Neural Network - Binary Classification)

- **Goal:** Predict whether a user made a purchase based on visit duration and pages visited.
- **Model:** Simple neural network using Keras
- **What I Learned:**
  - Built a basic feedforward neural network with `Dense` layers and a sigmoid activation for binary classification.
  - Understood the use of `binary_crossentropy` as a loss function for yes/no outcomes.
  - Observed how well the model trained on simple logic-based synthetic data (purchase = 1 if combined features exceed 1).
  - Gained exposure to Keras’s `.compile()`, `.fit()`, and `.evaluate()` pipeline.

---

### 3. Customer Churn Classification (Decision Tree)

- **Goal:** Predict if a telecom customer will churn based on age, service calls, and billing.
- **Model:** `DecisionTreeClassifier`
- **What I Learned:**
  - Visualized the decision process using `sklearn.tree.plot_tree()` — I could see the exact splits and rules the model used.
  - Identified the limits of synthetic, patterned data (e.g., repeating churn labels) and how it can falsely inflate accuracy.
  - Appreciated the **interpretability** of decision trees and how they can be used to explain decisions to stakeholders.

---

### 4. Vehicle Clustering (Unsupervised Learning)

- **Goal:** Group vehicles into clusters based on physical features like horsepower and weight.
- **Model:** `KMeans`
- **What I Learned:**
  - No labels were used — the model had to find structure from the feature space alone.
  - I visualized the clusters using a scatterplot with colors for each assigned cluster.
  - Learned that unsupervised models like KMeans are useful when classification labels aren’t available, but we still want to identify groups or anomalies.

---

## Libraries I Learned About

| Library | Purpose | What I Learned |
|--------|---------|----------------|
| `pandas` | DataFrame manipulation | I used it to organize data into tabular form and slice features/labels for model training. |
| `numpy` | Numeric computing | Used to generate synthetic data and perform quick mathematical operations like slicing, summing, and reshaping arrays. |
| `matplotlib` | Plotting library | Learned to plot scatter plots and customize axes and titles to understand model predictions vs actuals. |
| `seaborn` | Statistical visualization | Learned to quickly create multi-variable plots (`pairplot`) to explore feature relationships. |
| `sklearn` (scikit-learn) | ML models and tools | Used `RandomForestRegressor`, `DecisionTreeClassifier`, `KMeans`, `train_test_split`, and evaluation tools like `accuracy_score` and `mean_squared_error`. Learned about supervised/unsupervised learning, model fitting, prediction, and evaluation. |
| `tensorflow.keras` | Deep learning | Used to build and train a neural network with `Dense` layers. Learned about activation functions, loss metrics, optimizers, and how to structure input/output layers. |
| `warnings` | Suppressing warnings | Used to avoid cluttering the notebook output with training logs during learning. |

---

## Core Concepts Reinforced

| Concept | Description |
|--------|-------------|
| Supervised Learning | Learning with labeled data (classification + regression) |
| Unsupervised Learning | Learning from unlabeled data (clustering) |
| Model Evaluation | Used accuracy for classification, and MSE for regression |
| Neural Networks | Built and trained a basic Keras model for binary output |
| Decision Trees | Understood decision paths and model interpretability |
| Clustering | Visualized and interpreted KMeans clusters |
| Data Visualization | Used pairplots, scatterplots, and decision trees to explain behavior |

---

## What’s Next

- Build original projects using real-world datasets (e.g., from Kaggle or UCI)
- Apply more advanced evaluation techniques like AUC-ROC, precision/recall
- Compare multiple models on the same dataset
- Dive into feature engineering and model interpretability tools like SHAP
- Begin experimenting with generative AI tools (e.g., text generation, summarization)

---

## Note

The code in this repository is based on guided examples provided in the **Udacity x AWS “Introducing Generative AI”** course. All reflections, explanations, and observations are my own. This repository serves as a personal record of my growth in the early stages of my AI journey.

---

Thanks for visiting!  
Feel free to browse the project folders and follow my learning journey as I move from guided exploration to original builds.
