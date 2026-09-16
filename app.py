import os
import io
import re
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import nltk
import PyPDF2

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet"]:
    nltk.download(pkg, quiet=True)


st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


@st.cache_resource
def load_models():

    paths = {
        "tfidf": os.path.join(
            BASE_DIR,
            "tfidf_vectorizer.pkl"
        ),
        "model": os.path.join(
            BASE_DIR,
            "resume_classifier.pkl"
        ),
        "encoder": os.path.join(
            BASE_DIR,
            "label_encoder.pkl"
        )
    }

    missing_files = [
        path
        for path in paths.values()
        if not os.path.exists(path)
    ]

    if missing_files:
        st.error(
            "Missing model files:\n\n"
            + "\n".join(missing_files)
        )
        st.stop()

    try:

        with open(
            paths["tfidf"],
            "rb"
        ) as f:
            tfidf = pickle.load(f)

        with open(
            paths["model"],
            "rb"
        ) as f:
            classifier = pickle.load(f)

        with open(
            paths["encoder"],
            "rb"
        ) as f:
            label_encoder = pickle.load(f)

        return (
            tfidf,
            classifier,
            label_encoder
        )

    except Exception as e:

        st.error(
            f"Error loading model files: {e}"
        )
        st.stop()


tfidf, classifier, label_encoder = load_models()


lemmatizer = WordNetLemmatizer()

stop_words = set(
    stopwords.words("english")
)


TECH_KEEP = {
    "c",
    "cpp",
    "csharp",
    "r",
    "go",
    "rust",
    "php",
    "sql",
    "bash",
    "perl",
    "lua",
    "dart",
    "swift",
    "java",
    "javascript",
    "typescript",

    "ai",
    "ml",
    "dl",
    "nlp",
    "llm",
    "rag",
    "cv",
    "cnn",
    "rnn",
    "lstm",
    "gru",
    "gan",
    "gpt",
    "bert",
    "ner",
    "ocr",
    "asr",
    "t5",

    "db",
    "nosql",
    "etl",
    "olap",
    "oltp",
    "bi",
    "eda",
    "csv",
    "json",
    "xml",
    "api",

    "aws",
    "gcp",
    "azure",
    "ec2",
    "s3",
    "iam",
    "eks",
    "ecs",
    "vpc",
    "cdn",
    "dns",
    "vm",

    "git",
    "svn",
    "ci",
    "cd",
    "cicd",
    "devops",
    "docker",
    "k8s",
    "kubernetes",
    "jenkins",
    "helm",
    "ansible",
    "terraform",

    "html",
    "css",
    "js",
    "ts",
    "rest",
    "http",
    "https",
    "url",
    "uri",
    "dom",
    "ajax",
    "seo",

    "ui",
    "ux",
    "oop",
    "orm",
    "sdk",
    "ide",
    "npm",
    "pip",
    "pypi",
    "cli",
    "gui",

    "mysql",
    "dbms",
    "rdbms",
    "mongodb",
    "redis",
    "neo4j",
    "cassandra",
    "dynamodb",

    "os",
    "cpu",
    "gpu",
    "ram",
    "rom",
    "ssd",
    "tcp",
    "udp",
    "ip",
    "ssh",
    "ftp",
    "vpn",
    "lan",
    "wan",

    "dsa",
    "ds",
    "algo",
    "mvc",
    "mvp",
    "mvvm",
    "solid",
    "dry",
    "kiss",

    "ssl",
    "tls",
    "jwt",
    "oauth",
    "sso",
    "mfa",
    "2fa",

    "ios",
    "apk",

    "qa",
    "uat",
    "po",
    "pm",
    "saas",
    "paas",
    "iaas"
}


def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    replacements = {
        "c++": " cpp ",
        "c#": " csharp ",
        ".net": " dotnet ",
        "node.js": " nodejs ",
        "next.js": " nextjs ",
        "react.js": " reactjs ",
        "vue.js": " vuejs ",
        "web3.js": " web3js ",
        "scikit-learn": " scikitlearn ",
        "github actions": " githubactions ",
        "machine-learning": " machinelearning ",
        "deep-learning": " deeplearning ",
        "computer-vision": " computervision ",
        "natural-language-processing": " naturallanguageprocessing "
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new
        )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = word_tokenize(text)

    words = [
        word
        for word in words
        if (
            word not in stop_words
            or word in TECH_KEEP
        )
        and (
            len(word) > 1
            or word in TECH_KEEP
        )
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)


def read_pdf(uploaded_file):

    try:

        pdf_bytes = uploaded_file.read()

        reader = PyPDF2.PdfReader(
            io.BytesIO(pdf_bytes)
        )

        text = " ".join(
            page.extract_text() or ""
            for page in reader.pages
        )

        return text

    except Exception as e:

        st.error(
            f"PDF reading error: {e}"
        )

        return ""


def read_txt(uploaded_file):

    try:

        return uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

    except Exception as e:

        st.error(
            f"Text file reading error: {e}"
        )

        return ""


def get_match_score(
    job_desc,
    resume_text
):

    job_clean = clean_text(
        job_desc
    )

    resume_clean = clean_text(
        resume_text
    )

    if not job_clean or not resume_clean:
        return 0.0

    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform(
        [
            job_clean,
            resume_clean
        ]
    )

    score = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return round(
        score * 100,
        2
    )


def predict_top3(
    resume_text,
    top_n=3
):

    cleaned = clean_text(
        resume_text
    )

    if not cleaned:
        return []

    vector = tfidf.transform(
        [cleaned]
    )

    if hasattr(
        classifier,
        "predict_proba"
    ):

        probs = classifier.predict_proba(
            vector
        )[0]

        top_n = min(
            top_n,
            len(probs)
        )

        top_idx = np.argsort(
            probs
        )[::-1][:top_n]

        return [
            (
                label_encoder.classes_[i],
                round(
                    probs[i] * 100,
                    1
                )
            )
            for i in top_idx
        ]

    prediction = classifier.predict(
        vector
    )[0]

    category = (
        label_encoder.inverse_transform(
            [prediction]
        )[0]
    )

    return [
        (
            category,
            100.0
        )
    ]


def verdict_label(score):

    if score >= 40:
        return (
            "Strong Match",
            "green"
        )

    if score >= 20:
        return (
            "Possible Match",
            "orange"
        )

    return (
        "Weak Match",
        "red"
    )


with st.sidebar:

    st.title(
        "📄 AI Resume Screener"
    )

    st.markdown(
        "Built with **NLTK** + "
        "**Scikit-learn** + "
        "**Streamlit**"
    )

    st.markdown("---")

    st.markdown(
        "**How to use**"
    )

    st.markdown(
        "1. Go to **Screen Resumes** tab\n"
        "2. Paste a job description\n"
        "3. Upload one or more PDF/TXT resumes\n"
        "4. Click **Screen Resumes**"
    )

    st.markdown("---")

    st.markdown(
        "**Supported Categories**"
    )

    categories = [
        "Data Science",
        "Web Development",
        "Cybersecurity",
        "DevOps",
        "Mobile Development",
        "Artificial Intelligence",
        "Database Engineering",
        "Embedded Systems",
        "Game Development",
        "UI/UX Design",
        "Blockchain",
        "QA & Testing",
        "Product Management",
        "Systems Programming",
        "AR/VR Development",
        "Developer Relations"
    ]

    for category in categories:

        st.markdown(
            f"• {category}"
        )

    st.markdown("---")

    st.markdown(
        ":red[*Made with love by* "
        "**Sarthak Jain**]"
    )


st.title(
    "AI Resume Screening System"
)

st.markdown(
    "Match resumes to job descriptions "
    "and classify candidates instantly."
)

st.markdown("---")


tab1, tab2 = st.tabs(
    [
        "📋 Screen Resumes",
        "🎯 Classify Resume"
    ]
)


with tab1:

    col1, col2 = st.columns(
        2
    )

    with col1:

        st.subheader(
            "Job Description"
        )

        job_desc = st.text_area(
            "Paste the job description here:",
            height=280,
            placeholder=(
                "Enter job requirements, "
                "required skills, experience..."
            )
        )

    with col2:

        st.subheader(
            "Upload Resumes"
        )

        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files "
            "(multiple allowed)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if uploaded_files:

            st.success(
                f"{len(uploaded_files)} "
                f"file(s) uploaded"
            )

    if st.button(
        "Screen Resumes",
        type="primary",
        use_container_width=True
    ):

        if not job_desc.strip():

            st.error(
                "Please enter a job description."
            )

        elif not uploaded_files:

            st.error(
                "Please upload at least one resume."
            )

        else:

            results = []

            with st.spinner(
                "Analyzing resumes..."
            ):

                for file in uploaded_files:

                    try:

                        if file.name.lower().endswith(
                            ".pdf"
                        ):

                            resume_text = read_pdf(
                                file
                            )

                        else:

                            resume_text = read_txt(
                                file
                            )

                        if not resume_text.strip():

                            st.warning(
                                f"Could not extract "
                                f"text from "
                                f"{file.name}."
                            )

                            continue

                        score = get_match_score(
                            job_desc,
                            resume_text
                        )

                        top3 = predict_top3(
                            resume_text
                        )

                        if not top3:

                            continue

                        label, _ = verdict_label(
                            score
                        )

                        results.append(
                            {
                                "Candidate":
                                    file.name.rsplit(
                                        ".",
                                        1
                                    )[0],

                                "Match Score":
                                    score,

                                "Top Category":
                                    top3[0][0],

                                "Confidence":
                                    f"{top3[0][1]}%",

                                "Verdict":
                                    label
                            }
                        )

                    except Exception as e:

                        st.warning(
                            f"Error processing "
                            f"{file.name}: {e}"
                        )

            if results:

                st.markdown("---")

                st.subheader(
                    "Screening Results"
                )

                results_df = pd.DataFrame(
                    results
                )

                results_df = (
                    results_df
                    .sort_values(
                        "Match Score",
                        ascending=False
                    )
                    .reset_index(
                        drop=True
                    )
                )

                results_df.insert(
                    0,
                    "Rank",
                    range(
                        1,
                        len(results_df) + 1
                    )
                )

                display_df = (
                    results_df.copy()
                )

                display_df[
                    "Match Score"
                ] = display_df[
                    "Match Score"
                ].apply(
                    lambda x: f"{x}%"
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

                best = results_df.iloc[0]

                st.success(
                    f"🏆 Best Candidate: "
                    f"**{best['Candidate']}** — "
                    f"{best['Match Score']}% match"
                )

                st.markdown("---")

                st.subheader(
                    "Score Breakdown"
                )

                chart_df = pd.DataFrame(
                    {
                        "Candidate":
                            results_df[
                                "Candidate"
                            ],

                        "Score":
                            results_df[
                                "Match Score"
                            ]
                    }
                )

                chart_df = (
                    chart_df
                    .sort_values(
                        "Score",
                        ascending=False
                    )
                )

                st.bar_chart(
                    chart_df.set_index(
                        "Candidate"
                    )
                )

            else:

                st.warning(
                    "No valid resumes could "
                    "be analyzed."
                )


with tab2:

    resume_input = st.text_area(
        "Paste resume text here:",
        height=300,
        placeholder=(
            "Paste your full resume "
            "content here..."
        )
    )

    col_a, col_b = st.columns(
        [1, 3]
    )

    with col_a:

        classify_btn = st.button(
            "Classify Resume",
            type="primary",
            use_container_width=True
        )

    if classify_btn:

        if not resume_input.strip():

            st.error(
                "Please paste a resume."
            )

        else:

            with st.spinner(
                "Classifying..."
            ):

                top3 = predict_top3(
                    resume_input
                )

            if top3:

                st.markdown("---")

                st.subheader(
                    "Prediction Results"
                )

                columns = st.columns(
                    len(top3)
                )

                medals = [
                    "🥇",
                    "🥈",
                    "🥉"
                ]

                for i, (
                    category,
                    confidence
                ) in enumerate(top3):

                    with columns[i]:

                        st.metric(
                            f"{medals[i]} #{i + 1} Match",
                            category,
                            f"{confidence}% confidence"
                        )

                st.markdown("---")

                st.info(
                    f"This resume best matches "
                    f"a **{top3[0][0]}** role "
                    f"with **{top3[0][1]}%** confidence."
                )

                st.subheader(
                    "Confidence Breakdown"
                )

                confidence_df = pd.DataFrame(
                    top3,
                    columns=[
                        "Category",
                        "Confidence"
                    ]
                )

                st.bar_chart(
                    confidence_df.set_index(
                        "Category"
                    )
                )

            else:

                st.error(
                    "Could not classify the resume."
                )