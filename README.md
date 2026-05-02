# Toxicity Detection in Online Comments: A Comparative Analysis of Deep Learning Architectures

Online forums and social media are increasingly plagued by toxic comments, encompassing threats, insults, and hate speech that foster hostile and unsafe digital environments. To address this challenge, this project conducts a comparative analysis of multiple deep learning architectures to determine the most effective, efficient, and unbiased methods for toxicity classification.

### Problem Statement and Proposed Approach

Traditional machine learning approaches to toxicity classification often struggle with accuracy and can exhibit algorithmic bias against specific demographic groups. Furthermore, while contemporary sentence semantic classification methods offer high accuracy, they often come with massive computational costs.

Rather than relying on a single approach, this project systematically benchmarks a spectrum of neural network architectures. By evaluating everything from lightweight foundational models to complex Transformers and hybrid Recurrent/Convolutional networks, this project aims to identify exactly how different architectures handle the nuance of multi-label toxicity, and where the optimal balance of computational efficiency and classification accuracy lies.

### Architectures Explored

To establish a rigorous comparative experimental setting, this project implements and evaluates the following six architectures:

* **MLP (Multilayer Perceptron):** Serving as the foundational deep learning baseline to evaluate raw, non-sequential pattern recognition.
* **CNN (Convolutional Neural Network):** Deployed to test the efficacy of extracting local text features and isolated toxic n-grams.
* **C-LSTM (Convolutional Long Short-Term Memory):** A hybrid approach testing whether combining local feature extraction with sequential memory improves upon the pure CNN.
* **CNN-BiGRU:** A lower-complexity alternative that uses CNNs for keyword extraction while relying on Bidirectional Gated Recurrent Units (BiGRUs) to capture long-range dependencies and underlying semantics.
* **MCBiGRU (Multichannel CNN-BiGRU):** An advanced, multi-pathway extension of the CNN-BiGRU designed to capture even deeper semantic nuances.
* **BERT (Bidirectional Encoder Representations from Transformers):** A pre-trained large language model used to establish the performance ceiling for deep semantic understanding and context.

### Dataset and Evaluation Metrics

The models will be rigorously evaluated against the Wikipedia Toxic Comment Corpus. This dataset contains over 200,000 comments, each labeled by human raters to identify specific toxic behaviors. The dataset classifies toxicity into six distinct categories:
* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

To accurately and fairly assess these vastly different models, the project will rely on standard toxicity detection metrics:
* **Accuracy:** The percentage of comments correctly classified as either toxic or non-toxic.
* **Recall:** The percentage of actual toxic comments that the system successfully identifies.
* **F1-Score:** The weighted average of accuracy and recall, providing a balanced measure of a model's real-world robustness.
* **AUROC (Area Under the Receiver Operating Characteristic):** Utilized to evaluate the ranking capability and confidence calibration of the models across all thresholds.

### Foundational Literature

The theoretical framework for the models explored in this project is grounded in several core papers, with HuggingFace's `transformers` library powering the BERT implementation:

*   Risch, J., & Krestel, R. "Toxic Comment Detection in Online Discussions." *Springer*.
*   Khieu, K., & Narwal, N. "CS224N: Detecting and Classifying Toxic Comments." *Stanford University*.
*   Kim, Y. (2014). "Convolutional Neural Networks for Sentence Classification." *EMNLP*.
*   Zhang, D., Tian, L., Hong, M., Han, F., Ren, Y., & Chen, Y. (2018). "Combining Convolution Neural Network and Bidirectional Gated Recurrent Unit for Sentence Semantic Classification." *IEEE Access*.
*   Ashok Kumar J., Abirami S., Tina Esther Trueman, & Cambria, E. (2021). "Comment toxicity detection via a multichannel convolutional bidirectional gated recurrent unit." *Neurocomputing*.
*   Xiaoyan, L., Raga, R. C., & Xuemei, S. (2022). "GloVe-CNN-BiLSTM Model for Sentiment Analysis on Text Reviews." *Journal of Sensors*.
*   Li, H., Mao, W., & Liu, H. "Toxic Comment Detection and Classification." *Stanford University*.

### Experimental Methodology and Performance Benchmarking

This project is structured as an empirical comparative analysis of deep learning architectures. By systematically training and evaluating these models, the 
project aims to quantify:
* **Architectural Performance:** How foundational networks (like MLP and CNN) perform against increasingly complex, sequential, and attention-based models (like MCBiGRU and BERT) across the six toxicity classes.
* **The "Compute Tax" vs. Accuracy Trade-off:** A direct comparison of the computational cost (training time, VRAM usage, etc) of deep models against the 
  rapid execution of shallower models, determining whether marginal performance gains justify the increased computational overhead.
* **Hyperparameter Sensitivity:** An analysis of how each architecture responds to specific hyperparameter tuning (e.g., learning rates, dropout, layer depth, and threshold adjustments) and how those variables influence the delicate balance between Precision and Recall.el toxicity (e.g., subtle threats vs. explicit obscenity) in modern social media discourse.
