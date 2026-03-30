Markdown
<div align="center">

███╗   ███╗ ██████╗ ██╗   ██╗██╗███████╗    ██████╗ ███████╗ ██████╗
████╗ ████║██╔═══██╗██║   ██║██║██╔════╝    ██╔══██╗██╔════╝██╔════╝
██╔████╔██║██║   ██║██║   ██║██║█████╗      ██████╔╝█████╗  ██║

██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██║██╔══╝      ██╔══██╗██╔══╝  ██║

██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ██║███████╗    ██║  ██║███████╗╚██████╗
╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝


### 🎬 Movie Recommendation System (Content-Based)

## 📌 Overview

This project implements a **Content-Based Movie Recommendation System** using Natural Language Processing (NLP). By analyzing metadata such as titles, cast, and crew, the engine identifies mathematical similarities between movies to suggest the best matches for a user's favorite film.

> 🎯 **Goal:** Provide personalized movie suggestions by calculating feature vectors and cosine similarity scores across a dataset of 4,803 movies.

---

## 📊 Dataset

The system processes a comprehensive movie dataset with the following key attributes:

| Feature | Description |
|---|---|
| `movie_id` | Unique identifier for each film |
| `title` | The official name of the movie |
| `cast` | Stringified list of lead actors/characters |
| `crew` | Stringified list of directors and production staff |
| `combined_features` | ⭐ **Processed Feature Vector** (Title + Cast + Crew) |

- **Total Records:** 4,803 Movies
- **Technique:** TF-IDF Vectorization & Cosine Similarity

---

## 🔧 Project Workflow

Data Loading ──► Preprocessing ──► Vectorization ──► Similarity Matrix ──► User Input ──► Recommendations


### 1️⃣ Data Preprocessing
- **Feature Selection:** Focused on `title`, `cast`, and `crew` to capture the "essence" of a movie.
- **Null Handling:** Verified there are zero null values in the combined text features to ensure matrix integrity.
- **Text Concatenation:** Merged selected features into a single string per movie to create a rich text profile.

### 2️⃣ NLP & Vectorization
- Used **TfidfVectorizer** (Term Frequency-Inverse Document Frequency) to convert raw text into numerical feature vectors.
- Resulted in a sparse matrix of shape `(4803, 419819)`, capturing unique keywords and their importance.

### 3️⃣ Similarity Computation
- Calculated **Cosine Similarity** to measure the cosine of the angle between two vectors.
- Created a similarity square matrix of `(4803, 4803)` where each cell represents the likeness between two films.

### 4️⃣ User Interaction Logic
- Integrated `difflib.get_close_matches` to handle user typos (e.g., "iron man" vs "Ironman").
- Ranked the similarity scores in descending order to fetch the Top 30 most relevant movies.

---

## 📈 Technical Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation and matrix operations |
| `scikit-learn` | TF-IDF Vectorization & Cosine Similarity scores |
| `difflib` | Handling user input string matching |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy pandas scikit-learn
Run the System
Bash
# Clone the repository
git clone [https://github.com/your-username/movie-recommendation-system.git](https://github.com/your-username/movie-recommendation-system.git)
cd movie-recommendation-system

# Run the notebook or script
# Input your favorite movie when prompted!
📁 Project Structure
movie-recommendation/
│
├── 📓 Movie_Recommendation.ipynb   # Analysis & Model code
├── 📄 movies.csv                  # Dataset file
└── 📘 README.md                   # Project documentation
