# Multilingual Sentiment Analysis Using LSTM & Naive Bayes

## 📌 Project Overview
This project performs **sentiment analysis** on text data using **LSTM (Long Short-Term Memory) networks** and **Naive Bayes** classifiers. The goal is to classify text into positive or negative sentiment using machine learning and deep learning techniques.

## 🚀 Technologies Used
- **Python**
- **TensorFlow/Keras** (for LSTM model)
- **Scikit-learn** (for Naive Bayes classifier)
- **NLTK & SpaCy** (for text preprocessing)
- **Pandas & NumPy** (for data handling)
- **Matplotlib & Seaborn** (for visualization)

## 📂 Project Structure
```
📁 Sentiment_Analysis
│── 📄 sentiment_analysis.ipynb  # Jupyter Notebook with code
│── 📄 README.md                 # Project Documentation
│── 📄 requirements.txt          # Dependencies list
|-- 📄 sentiment_analysis.py     # Main file
```

## 🔧 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use 'venv\Scripts\activate'
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Models Used
### 1️⃣ **Naive Bayes Classifier**
- Uses **TF-IDF** vectorization to transform text data.
- Trained using **Multinomial Naive Bayes** with hyperparameter tuning via GridSearchCV.
- Evaluated using **confusion matrix and classification report**.

### 2️⃣ **LSTM Neural Network**
- Uses an **Embedding layer** to represent words as vectors.
- LSTM layer captures **sequential dependencies** in text.
- Dropout layer prevents overfitting.
- Uses **sigmoid activation** in the final Dense layer for binary classification.

## 🛠️ How to Run the Models
### Run Naive Bayes Classifier:
```python
python naive_bayes_sentiment.py
```

### Run LSTM Model:
```python
python lstm_sentiment.py
```

## 📊 Results & Evaluation
- **Naive Bayes Model Accuracy:** X% (Replace with actual accuracy)
- **LSTM Model Accuracy:** Y% (Replace with actual accuracy)
- Model performance is analyzed using **confusion matrix, F1-score, and accuracy**.

## 📌 Future Improvements
- Train on a larger dataset for better generalization.
- Experiment with **Bidirectional LSTMs** for improved sentiment prediction.
- Implement **attention mechanisms** to improve word importance detection.
