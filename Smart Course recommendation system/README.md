# Smart Course Recommendation System

Smart Course Recommendation System is a Flask web application that recommends Udemy courses using NLP-based similarity search and AI-assisted guidance.

## Brief Project Info

This project helps learners discover relevant online courses in two ways:

- Keyword search with filters (level, price, rating) using TF-IDF + cosine similarity.
- Conversational AI advisor that collects a learner profile and returns personalized recommendations with reasons and a suggested learning path.
- Intelligent chatbot tutor for follow-up course and learning questions.

It uses `udemy.csv` as the main data source and renders results through a multi-page Flask UI.

## Features

- TF-IDF recommendation engine over cleaned course metadata (`course_title`, `subject`, `level`)
- Filtered search by:
  - Course level
  - Maximum price
  - Minimum rating
- AI Learning Advisor flow:
  - Collects goal, level, learning objective, available time, and preference
  - Produces top recommendations
  - Adds AI-generated explanation per course
  - Generates a concise learning path
- Intelligent Chatbot Tutor:
  - Answers learning questions
  - Suggests relevant courses with links
  - Returns markdown-formatted responses rendered as HTML
- Responsive frontend pages using Flask templates

## Tech Stack

- Backend: Python, Flask
- Data: Pandas
- NLP/ML: `neattext`, `scikit-learn` (`TfidfVectorizer`, cosine similarity)
- LLM: Google Generative AI (Gemini)
- Frontend: HTML/CSS (Jinja templates)

## Project Structure

```text
.
|- app.py
|- recommendation_system.py
|- udemy.csv
|- UdemyCleanedTitle.csv
|- templates/
|  |- index.html
|  |- results.html
|  |- advisor.html
|  |- ai_results.html
|  |- tutor.html
|- .env
```

Note: `nav.md` and `surrounding.md` are intentionally not part of this README scope.

## How It Works

1. Dataset is loaded from `udemy.csv`.
2. Course titles are cleaned (stopword and punctuation removal).
3. Combined course text is vectorized with TF-IDF.
4. User query is vectorized and compared with cosine similarity.
5. Top matching courses are returned and optionally filtered.
6. AI features (advisor/tutor) generate explanations and guidance using Gemini.

## Setup

### 1. Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install flask pandas scikit-learn neattext python-dotenv google-generativeai markdown
```

### 3. Configure environment variables

Create or update `.env`:

```env
SECRET_KEY=your_flask_secret_key
GEMINI_API_KEY=your_gemini_api_key
```

Important: `app.py` currently contains a hardcoded Gemini key. For safety, replace it with:

```python
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```

## Run the App

```powershell
python app.py
```

Open: `http://127.0.0.1:5000`

## Main Routes

- `/` Home page with search form and navigation
- `/search` Returns similarity-based recommendations
- `/advisor` AI profile chat and personalized course suggestions
- `/tutor` Intelligent tutor chat for learning support

## Notes

- Prices are converted in code using an exchange multiplier (`EGP -> NGN`).
- AI response quality depends on prompt quality and API availability.
- Large CSV processing happens at startup.

## Future Improvements

- Move secrets fully to environment variables
- Add `requirements.txt`
- Add unit/integration tests
- Add caching for repeated recommendation queries
- Add pagination and ranking explainability metrics

