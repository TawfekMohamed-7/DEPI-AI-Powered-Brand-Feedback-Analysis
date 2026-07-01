# DEPI-AI-Powered-Brand-Feedback-Analysis

A comprehensive **Reddit-based brand sentiment analysis and recommendation system** built with **Streamlit**, **Python**, and a modular ML pipeline. The project collects Reddit posts and comments, cleans and preprocesses text, performs sentiment analysis, and generates brand-level ranking and insights through an interactive dashboard.

---

## Key Features
- **Reddit Data Collection:** Scrapes posts and comments for Egyptian brands and related keywords.
- **Sentiment Analysis Pipeline:** Cleans text, extracts features, and predicts sentiment using a trained ML model.
- **Brand Recommendation Engine:** Ranks companies based on sentiment quality, engagement, and confidence.
- **Interactive Streamlit Dashboard:** Displays metrics, rankings, logs, and workflow status in a modern UI.
- **MLOps Workflow View:** Tracks the end-to-end pipeline from data collection to deployment.
- **Egypt-Focused Brand Intelligence:** Supports telecom, banking, fintech, ecommerce, delivery, and more.

---

## Project Architecture
The system is organized into modular components:

1. **Data Collection**
   - Scrapes Reddit posts and comments.
   - Filters Egypt-relevant content.
   - Stores raw and processed outputs.

2. **Preprocessing & Feature Engineering**
   - Cleans text and normalizes content.
   - Generates TF-IDF features for classification.

3. **Model Training**
   - Trains a sentiment classifier using SVM.
   - Saves the model and vectorizer as `.pkl` artifacts.

4. **Prediction & Recommendation**
   - Predicts sentiment for brand-related content.
   - Produces recommendation scores and ranking tables.

5. **Streamlit Dashboard**
   - Provides a visual interface for analysis and monitoring.
   - Shows pipeline stages, metrics, and brand comparisons.

---

## Repository Structure

| File/Folder | Description |
| :--- | :--- |
| `app/` | Streamlit application folder. |
| `app/streamlit_app.py` | Main dashboard application. |
| `src/` | Core pipeline and utility scripts. |
| `src/scraper.py` | Collects Reddit posts and comments. |
| `src/train_model.py` | Trains the sentiment classification model. |
| `src/predict_reddit.py` | Runs sentiment prediction on collected data. |
| `src/recommendation.py` | Builds brand profiles and ranking logic. |
| `src/run_pipeline.py` | Orchestrates the full pipeline. |
| `models/` | Saved ML artifacts such as vectorizer and classifier. |
| `data/` | Raw, labeled, and processed datasets. |
| `notebooks/` | Jupyter notebooks for data collection and model training. |
| `requirements.txt` | Python dependencies. |

---

## Pipeline Workflow
1. **Collect Data** from Reddit using brand-specific search queries.
2. **Filter Relevant Content** using Egypt-focused keywords and brand aliases.
3. **Clean Text** by removing URLs, punctuation, emojis, and Arabic diacritics.
4. **Train Model** on labeled sentiment data.
5. **Predict Sentiment** for new brand mentions.
6. **Rank Companies** using positive/negative feedback and confidence.
7. **Visualize Results** in the Streamlit dashboard.

---

## Supported Brand Categories
- Telecom
- Fintech
- Banking
- Ecommerce
- Delivery
- Transport
- Real Estate
- Healthcare
- Classifieds
- Other

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/brand-feedback-analyzer.git
cd brand-feedback-analyzer
```

### 2. Create a Virtual Environment
```bash
python -m venv .venv
```

### 3. Activate the Environment
**Windows**
```bash
.venv\Scripts\activate
```

**Linux / macOS**
```bash
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Full Pipeline
```bash
python src/run_pipeline.py
```

### Train the Model
```bash
python src/train_model.py
```

### Run Predictions
```bash
python src/predict_reddit.py
```

### Launch the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

---

## Data Flow
- `data/raw/` → raw scraped data.
- `data/labeled/` → labeled training data.
- `data/processed/` → cleaned and model-ready data.
- `models/` → trained model and vectorizer artifacts.

---

## Model Details
The project uses:
- **TF-IDF Vectorizer** for text feature extraction.
- **SVM Classifier** for sentiment classification.
- **Recommendation Scoring** based on sentiment distribution and record volume.

---

## Configuration
To enable Reddit access, set your credentials in:
```toml
.streamlit/secrets.toml
```

Example:
```toml
[reddit]
clientid = "YOUR_CLIENT_ID"
clientsecret = "YOUR_CLIENT_SECRET"
```

You can also use environment variables:
- `REDDITCLIENTID`
- `REDDITCLIENTSECRET`

---

## Example Output
The dashboard provides:
- Sentiment metrics
- Brand ranking tables
- Model status overview
- Data pipeline stages
- Deployment monitoring panel

---

## Keywords
`Sentiment Analysis` `Reddit Scraping` `Machine Learning` `SVM` `TF-IDF` `Streamlit` `MLOps` `Brand Intelligence` `Text Classification`
