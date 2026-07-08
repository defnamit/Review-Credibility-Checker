# 🛡️ Review Credibility Checker

It is a Machine Learning-powered Flask web application that classifies product reviews as **Fake**, **Suspicious**, or **Genuine**. The system combines Natural Language Processing (NLP) with review metadata to help users identify trustworthy online reviews.

---

## 🚀 Features

- 🔍 Detects Fake, Suspicious, and Genuine reviews
- 🤖 Machine Learning classification using Random Forest
- 📝 TF-IDF text vectorization
- 📊 Real-time prediction through a Flask web interface
- ⚡ Simple and responsive UI
- 📈 Feature engineering for improved prediction accuracy

---

## 🛠️ Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS

---

## 📂 Project Structure

```
TrustNet/
│
├── app.py
├── dataset.csv
├── requirements.txt
│
├── templates/
│   ├── index.html
│   └── base.html
│
├── .idea/
│
└── README.md
```

---

## 🧠 Machine Learning Pipeline

The application performs the following steps:

1. Receive user review
2. Extract review features
3. Convert review text using TF-IDF
4. Scale numerical features
5. Predict review authenticity using Random Forest
6. Display prediction to the user

---

## 📋 Features Used for Prediction

- Review Text
- Rating
- Account Age
- Verified Purchase
- Review Length
- Capital Letter Ratio
- Exclamation Count
- Repeated Phrases
- Profile Photo Availability

---

## 🎯 Prediction Classes

| Class | Meaning |
|--------|----------|
| 🟢 Genuine | Review appears authentic |
| 🟡 Suspicious | Review has suspicious patterns |
| 🔴 Fake | Review is likely fake |

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/trustnet.git
```

Move into the project

```bash
cd trustnet
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python feedback.py
```

Open your browser

```
http://127.0.0.1:5000
```

---

## 📸 Screenshots

Add screenshots of

- Home Page
- Prediction Result
- Fake Review Detection
- Genuine Review Detection

---

## 🔮 Future Improvements

- Deep Learning (BERT)
- User Authentication
- Review History
- Explainable AI (Prediction Reasons)
- Confidence Score
- Database Integration
- REST API
- Dark Mode
- Deployment on Cloud

---

## 👨‍💻 Author

**Namit Singh**

AI & Machine Learning Student

---

## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
