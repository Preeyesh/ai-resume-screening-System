# AI Resume Screening System

A machine learning web app that reads a resume and predicts the most suitable job category from 16+ tech domains.

## Features

- Upload a PDF resume or paste plain text
- Predicts job category with confidence scores (top 3)
- Trained on 60+ resumes across 16 categories
- Ensemble model (SVM + Logistic Regression + Random Forest)
- Clean UI built with Streamlit

## Tech Stack

- **Frontend** — Streamlit
- **NLP** — NLTK (tokenization, lemmatization, stopword removal)
- **ML** — scikit-learn (TF-IDF, LinearSVC, LogisticRegression, RandomForest, VotingClassifier)
- **PDF Parsing** — PyPDF2
- **Data** — pandas, numpy
- **Visualization** — matplotlib, seaborn

## Categories Supported

Data Science, Web Development, Cybersecurity, DevOps, Mobile Development,
Artificial Intelligence, Database Engineering, Embedded Systems, Game Development,
UI/UX Design, Blockchain, QA & Testing, Product Management, Systems Programming,
AR/VR Development, Developer Relations

## Setup

```bash
git clone https://github.com/your-username/ai-resume-screening.git
cd ai-resume-screening

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Train the Model

Run the notebook first to generate the `.pkl` files:

```bash
jupyter notebook AI_Resume_Screening_System.ipynb
```

This saves:
- `tfidf_vectorizer.pkl`
- `resume_classifier.pkl`
- `label_encoder.pkl`

## Run the App

```bash
streamlit run app.py
```

## Project Structure

```
ai-resume-screening/
├── app.py
├── AI_Resume_Screening_System.ipynb
├── requirements.txt
├── .gitignore
├── README.md
├── tfidf_vectorizer.pkl
├── resume_classifier.pkl
└── label_encoder.pkl
```

## Made with love by Preeyesh Joshi