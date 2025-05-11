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
   git clone https://github.com/Virenbhalgamiya/Multilingual-Sentiment-analysis-dashboard.git
   cd Multilingual-Sentiment-analysis-dashboard
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
4. Run
   ```bash
   streamlit run sentiment_analysis.py
   ```

## 📊 Models Used
### 1️⃣ **Naive Bayes Classifier**
- Uses **TF-IDF** vectorization to transform text data.
- Trained using **Multinomial Naive Bayes** with hyperparameter tuning via GridSearchCV.
- Evaluated using **confusion matrix and classification report**.



## 📌 Future Improvements
- Train on a larger dataset for better generalization.
- Experiment with **Bidirectional LSTMs** for improved sentiment prediction.
- Implement **attention mechanisms** to improve word importance detection.
