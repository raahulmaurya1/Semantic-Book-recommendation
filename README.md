
# 🌍📖 World’s First Emotion + Semantics-Based Book Recommendation System

> _“Not just what you want to read, but how you want to feel.”_  
> _Redefining the future of reading using AI, emotions, and intent._

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Streamlit-red)
---

## 🚀 About the Project

Welcome to the **world’s first truly intelligent Book Recommendation System**—one that understands **how you feel**, what you're **curious about**, and the **vibe** of your perfect book.

Unlike traditional platforms that recommend books based only on ratings or sales, this system taps into the **power of emotions, semantic meaning, and genre** to personalize your literary journey.

We built this to **change the lifestyle of reading**—making discovery more emotional, intuitive, and transformative than ever before.

---

## 🔍 Key Highlights

### 🧠 AI-Powered Semantic Search  
Using **Gemini AI embeddings**, the system deeply understands the **meaning** behind your input—whether it’s a feeling, phrase, or story idea.

### 🎭 Emotion-Aware Recommendations  
Books are filtered by **emotional tone** (Happy, Sad, Suspenseful, Angry, Surprising, etc.) using **Hugging Face sentiment models**—so you find the book that matches your current state of mind.

### 🧩 Genre & Category Control  
Easily select from genres like Fiction, Mystery, Sci-Fi, Romance, etc., to tailor results using **category-based metadata filtering**.

### 🖥️ Interactive Frontend (Gradio Dashboard)  
A **real-time, browser-based interface** built with **Gradio**, allowing users to describe what they want and immediately view custom recommendations.

---


## 📸 Screenshot

<img src="https://github.com/raahulmaurya1/Semantic-Book-recommendation/blob/e08b766e8bdda2271cd275f86acc576583db228b/Picture1.png" width="500" height="500"/>

---

## 🧪 Technical Stack

| Layer                | Technology Used                                 |
|----------------------|--------------------------------------------------|
| 🧠 Semantic Embeddings | [Gemini AI](https://deepmind.google/technologies/gemini/) |
| 📖 Emotion Detection  | [Hugging Face Transformers](https://huggingface.co/) |
| 🔗 Semantic Retrieval | [LangChain](https://www.langchain.com/)          |
| ⚙️ Backend API        | [FastAPI](https://fastapi.tiangolo.com/)         |
| 💻 UI Layer           | [Gradio](https://www.gradio.app/)                |
| 📊 Data Processing    | Python (Pandas, NumPy)                           |

---

## 📂 Project Structure

```
📁 book-recommendation-system/
├── app.py                     # Main backend logic (FastAPI + Gradio)
├── .env                       # Environment variables (API keys)
├── requirements.txt           # Python dependencies
├── 📁 data/
│   ├── books_with_emotions.csv       # Metadata with emotional scores
│   ├── books_with_categories.csv     # Metadata with genres/categories
│   └── tagged_description.txt        # Preprocessed text for embedding
├── 📁 notebooks/
│   ├── data_exploration.ipynb        # EDA & cleaning steps
│   ├── sentiment-analysis.ipynb      # Hugging Face classification
│   └── vector-search.ipynb           # LangChain vector search logic
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/book-recommendation-system.git
cd book-recommendation-system
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a `.env` file in the root directory:
```env
GOOGLE_API_KEY=your_google_api_key
HUGGINGFACE_API_KEY=your_hugging_face_api_key
```

### 4️⃣ Run the App
```bash
python app.py
```

### 5️⃣ Access the System
Open the URL provided by Gradio (typically `http://localhost:7860/`) in your browser and start exploring.

---

## 📊 Datasets Used

- **books_with_emotions.csv** – Emotionally scored book metadata  
- **books_with_categories.csv** – Genre and category-tagged books  
- **tagged_description.txt** – Textual input for vectorization

---

## 🔬 Example Use Cases

| Input Example | Result |
|---------------|--------|
| _“I feel heartbroken, I need something inspiring.”_ | Shows motivational and hopeful stories |
| _“Give me a suspenseful sci-fi novel.”_ | Pulls thrilling futuristic plots |
| _“Happy romance in a magical world.”_ | Recommends uplifting fantasy love stories |

---

## 🤝 Contribution

Contributions are welcome!  
Open an issue or submit a PR to improve emotion detection, expand the dataset, or optimize performance.

---

## 📜 License

Distributed under the **MIT License**.  
See [LICENSE](LICENSE) for more information.

---

## 🙏 Acknowledgments

- [LangChain](https://www.langchain.com/) – For semantic retrieval  
- [Hugging Face](https://huggingface.co/) – For emotion detection  
- [Gradio](https://www.gradio.app/) – For beautiful, fast prototyping  
- [Gemini AI](https://deepmind.google/technologies/gemini/) – For next-gen embeddings

---

## 🌐 Final Thought

> This is more than a tool. It’s a companion for readers.  
> A system that listens to your emotions, understands your thoughts, and recommends books that **resonate**.

Be part of the reading revolution.

## 📬 Contact

Created by Rahul Maurya 
📧 Email: raahulmaurya2@gmail.com  
🔗 GitHub: @raahulmaurya1

