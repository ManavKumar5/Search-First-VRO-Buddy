🛍️ Search-First VRO Buddy                                     
🔹 Objective
Search-First VRO Buddy is a prototype AI shopping assistant that helps users discover products from a catalog using search-based discovery only.
The assistant supports keyword queries (e.g., “blue sneakers under 5000”) and filter-based searches (category, price), making product discovery faster and smarter.
________________________________________
🔹 Features
🔍 Search-Based Discovery
•	Keyword queries using vector embeddings.
•	Filters for category and price range.
🤝 Multi-Modal Interaction
•	Accepts text input queries.
•	Displays products with image, name, description, category, and price.
🎯 Personalization (Optional)
•	Tracks user clicks and category preferences.
•	Adjusts ranking dynamically based on interaction history.
________________________________________
🔹 Tech Stack
•	Language: Python 
•	Libraries:
o	sentence-transformers → for embeddings
o	faiss → for vector similarity search
o	scikit-learn → for TF-IDF & cosine similarity
o	pandas, numpy → for data handling
o	streamlit → for interactive web UI
•	Dataset: Custom product catalog (catalog.csv) with:
o	id, name, description, category, price, image
________________________________________
🔹 How It Works
1️⃣ Catalog Preparation
•	Product catalog stored in CSV file.
•	Each entry contains product details + image URL.
2️⃣ Index Building
•	Generate embeddings using SentenceTransformer.
•	Store and index in FAISS for fast similarity search.
•	Use TF-IDF for keyword-based relevance.
3️⃣ Search Process
•	Query → Embedding similarity + TF-IDF score.
•	Scores normalized and combined (weighted).
•	Apply filters (category, price).
•	Apply personalization boosts (if enabled).
•	Return Top-k matching results.
4️⃣ User Interface
•	Built with Streamlit (app.py).
•	Sidebar filters for category & price range.
•	Product cards with image + details.
•	Like button ❤️ → stores user preferences for future ranking boosts.
________________________________________
🔹 Deliverables
✅ Working prototype (Streamlit web app)
✅ Sample product catalog (20–50 items)
✅ Clean, modular, reproducible codebase
✅ README.md (this file)
________________________________________
🔹 Limitations
⚠️ Small catalog size (20–50 products only)
⚠️ No integration with real e-commerce APIs
⚠️ Personalization is basic (click-based boost only)
⚠️ Embeddings are pre-trained (not fine-tuned on shopping data)
________________________________________
🔹 Potential Next Steps
🚀 Expand catalog with real product datasets
🚀 Enhance personalization (session history, collaborative filtering, ML models)
🚀 Deploy on cloud platforms (Hugging Face Spaces, Streamlit Cloud)
🚀 Support multi-modal input (image + text search)
