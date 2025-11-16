# Phishing Email Detection

A study exploring how Machine Learning (ML) and Deep Learning (DL) can help detect phishing emails.

## Project Background

Phishing, first identified in 1996 through attacks on America On-Line (AOL) accounts, is a long-standing form of social engineering. Attackers ("phishers") use deceptive messages ("baits") to steal personal information from victims ("fish"). The term **phishing** is a historical evolution of hacker slang inherited from "Phone Phreaking".

Since then, phishing has become one of the most pervasive cyber threats. Attackers constantly adapt, crafting highly personalized and context-aware emails that bypass traditional rule-based filters.

Modern approaches increasingly rely on **ML and DL models**, which can detect subtle linguistic patterns, semantic structures, or behavioral anomalies that traditional filters miss.

This project investigates how these techniques can improve phishing detection, the challenges they face, and the trade-offs between **accuracy, adaptability, and interpretability**.

## Research Problem

Despite progress in ML and DL, a core challenge remains:

### **How effective are ML and DL approaches at detecting phishing emails, and what trade-offs exist between accuracy, adaptability, and interpretability in real-world deployment?**

## State of the Art

### **Traditional ML approaches**

Models such as **SVM**, **Random Forest**, or **Decision Trees** show strong performance using engineered textual features (lexical, structural, semantic). Their main limitation lies in:

- manual feature engineering
- reduced adaptability to new phishing strategies

### **Deep Learning approaches**

Using CNNs, RNNs, LSTMs, or Transformers, DL models automatically extract hierarchical features and better generalize across message variations. However, they require:

- large labeled datasets
- intensive computational resources
- improved interpretability

### **Hybrid solutions**

The emerging trend favors combining ML and DL to balance **performance, robustness, and explainability**.

## Dataset

The study uses a dataset compiled by **Advaith S. Rao**, merging 3 sources:

- The **"Enron emails dataset"** containing emails recovered from the Enron company upon its bankrupcy.
- A **"Phishing emails dataset"** containing phishing emails.
- And a **"Social engineering dataset"** containing more phishing emails.


It contains ~450k emails and 32 columns, making it appropriate for large-scale ML experimentation.

Dataset link:
[DATASET ON KAGGLE](https://www.kaggle.com/datasets/advaithsrao/enron-fraud-email-dataset/data)

## Sources

### **Papers**

- *Phishing Email Detection Using Natural Language Processing Techniques: A Literature Survey*
  [https://www.sciencedirect.com/science/article/pii/S1877050921011741](https://www.sciencedirect.com/science/article/pii/S1877050921011741)

- *Deep Learning for Phishing Detection: Taxonomy, Current Challenges and Future Directions*
  [https://ieeexplore.ieee.org/abstract/document/9716113](https://ieeexplore.ieee.org/abstract/document/9716113)

## How to Run the Project

- Clone the repo using `git clone` and navigate to the root of the project.
- After making sure you had python installed, run
    `python -m venv .venv`
to create a venv and activate the venv.
    - `source .venv/bin/activate` on linux
    - `.venv/script/activate.ps1` on windows
- Once in your venv, run 
    `pip install -r ./src/requirements.txt` 
to install the dependencies
- Finally, run the app using 
    `python ./src`
