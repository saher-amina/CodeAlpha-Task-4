# CodeAlpha-Task-4
CodeAlpha Internship Task 4-Sentiment Analysis 
## Sentiment Analysis using Python & NLP

### 📌 Project Description
This project is part of **CodeAlpha Internship Task 4**.  
The goal of this project is to perform **Sentiment Analysis** on textual data using Python and Natural Language Processing.  
Sentiment Analysis helps to identify whether a given text expresses **Positive, Negative, or Neutral** emotions.  
This is widely used in product reviews, social media analysis, and customer feedback.

---

### 🎯 Objectives
- Load and analyze the dataset `sentiment_dataset.csv`
- Clean and preprocess text data
- Extract features using TF-IDF Vectorizer
- Train a Machine Learning model for sentiment classification
- Evaluate model performance
- Generate a detailed report with visualizations

---

### 📂 Files in this Repository
| File Name | Description |
| --- | --- |
| `sentiment_analysis.py` | Main Python script. Contains data loading, preprocessing, model training and prediction |
| `sentiment_dataset.csv` | Dataset used for training and testing the sentiment model |
| `Sentiment_Analysis_Report.pdf` | Detailed report with graphs, results and analysis |
| `README.md` | Project documentation |
| `README_1.md` | Additional notes |

---

### 🛠️ Technologies, Objects and Rules Used
| Object / Library | Rule & Usage |
| --- | --- |
| **Python 3.x** | Main language. Rule: Version 3.8+ recommended |
| **pandas** | `pd.read_csv()` object. Rule: Used to load CSV dataset into DataFrame |
| **numpy** | `np.array()` object. Rule: Used for numerical operations |
| **nltk** | `word_tokenize()`, `stopwords` object. Rule: Used to tokenize text and remove stopwords |
| **re** | `re.sub()` function. Rule: Used to remove punctuation, numbers and special characters |
| **scikit-learn** | `TfidfVectorizer`, `train_test_split`, `LogisticRegression` objects. Rule: TF-IDF converts text to vectors. Model is trained on 80% data and tested on 20% |
| **matplotlib / seaborn** | `plt.figure()`, `sns.heatmap()` objects. Rule: Used to plot accuracy and confusion matrix |
| **joblib** | `joblib.dump()` function. Rule: Used to save the trained model |

#### **Key Rules Followed in Code:**
1.  **Data Rule**: Dataset must have at least 2 columns: `text` and `sentiment`
2.  **Preprocessing Rule**: All text converted to lowercase. Stopwords and punctuation removed
3.  **Model Rule**: Data is split into Train 80% and Test 20% before training
4.  **Evaluation Rule**: Model evaluated using Accuracy, Precision, Recall, F1-Score

---

### ⚙️ Installation
Run the following command to install all required libraries:
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn joblib
