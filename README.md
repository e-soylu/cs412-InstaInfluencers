# cs412-InstaInfluencers
CS412 Term Project | InstaInfluencers



## Project Overview:
This project aims to classify Instagram account content types and predict the number of likes on posts using machine learning models. The goal is to improve classification accuracy and reduce regression errors through three rounds of experimentation. Each round explores different models and techniques, ultimately concluding that Support Vector Machine (SVM) yields the best performance.

## Project Structure:
The project is divided into three rounds:
1. **Round 1: SVM Classification**
   - In this round, the Support Vector Machine (SVM) model was used to classify Instagram content and predict likes. SVM was chosen for its ability to handle high-dimensional data and efficiently create linear decision boundaries.
   
2. **Round 2: Ensemble Learning**
   - In this round, Random Forest (an ensemble learning method) was tested to improve classification accuracy. Ensemble methods help reduce overfitting by combining multiple weak learners into a stronger model. Additionally, models such as Logistic Regression, Naïve Bayes, and Multi-Layer Perceptron were also utilized in this phase.
   
3. **Round 3: Exploring Other Models**
   - In the final round, various models like Gradient Boosting were explored to see if they could improve classification accuracy or reduce regression error. Despite some improvements, SVM was still the most effective model.

## Key Models and Parameters:
1. **SVM (Support Vector Machine):**
   - **Kernel:** Linear kernel was used in Round 1 to classify Instagram content. The linear kernel was chosen for its simplicity and efficiency in linearly separable problems.
   
2. **Random Forest:**
   - **n_estimators:** 100 decision trees were used to form the Random Forest. This value was chosen for a balance between computational efficiency and model performance.
   - **random_state:** 42, ensuring reproducibility of the results
  
3. **Logistic Regression**
   - Solver: 'liblinear,' suitable for smaller datasets and binary classification tasks.
   - Penalty: L2 regularization to reduce overfitting.
   - C: Default value (1.0), balancing regularization strength and accuracy.

4. **Naïve Bayes**
   - Type: Gaussian Naïve Bayes, assuming normally distributed features.
   - Advantage: Efficiently handles high-dimensional datasets.

5. **Multi-Layer Perceptron (MLP)**
   - Hidden Layers: 1 hidden layer with 100 neurons, capturing non-linear patterns.
   - Activation: ReLU, preventing vanishing gradient issues and enabling effective training.
   - Optimizer: Adam, leveraging adaptive learning rates and momentum for faster convergence.
   - random_state: 42, ensuring consistent and reproducible results.



## SMOTE Usage:
SMOTE (Synthetic Minority Over-sampling Technique) was applied to handle class imbalance in the training data. SMOTE generates synthetic samples for underrepresented classes to ensure the model does not become biased toward the majority class.

## Performance Results:
- **Round 1 (SVM):** 
  - Classification Accuracy: 77.43%
  - Regression Error: 727.46
  
- **Round 2 (Random Forest):**
  - Classification Accuracy: 84.14%
  - Regression Error: 727.47
  
- **Round 3 (SVM):**
  - Similar results to Round 2 in terms of classification, but SVM was chosen as the final model due to its overall performance.

## Libraries Used:
- `scikit-learn`: For machine learning models.
- `numpy`: For numerical operations.
- `pandas`: For data manipulation.
- `imblearn`: For SMOTE implementation.


## Running the Project:
To run the project, execute the Jupyter Notebooks in the following order:
1. `round1-SVM Based.ipynb`
2. `round2-ensemble Based_sent.ipynb`
3. `round3-SVM Based.ipynb`

These notebooks include all the necessary steps for data preprocessing, model training, and evaluation.

## Conclusion:
This project demonstrates the application of machine learning models in classifying Instagram content and predicting like counts. Through three rounds of experimentation, we found that SVM provided the best results for both classification and regression tasks. The project also highlights the importance of addressing class imbalance using techniques like SMOTE to improve model performance.

## Future Work:
Future improvements could include hyperparameter tuning to further optimize model performance or applying deep learning techniques for more complex problems in content classification and prediction.
