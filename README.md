# Restaurant-Recommender-System
A Neural Recommender System powered by TensorFlow/Keras. Uses Deep Learning and Entity Embeddings to predict restaurant ratings and rank dining options based on location, cuisine, and budget constraints, featuring smart fallback logic for zero-result queries.
# 🍔 Deep Learning Restaurant Recommender

A restaurant recommendation engine built with **TensorFlow/Keras**. 

Most tutorials just use simple cosine similarity or matrix factorization. I wanted to try something different: using **Deep Learning (Neural Collaborative Filtering)** to learn "latent" features about cities and cuisines. Ideally, this helps the model understand that *Italian* and *Pizza* are related, or that certain cities have similar pricing structures, without me having to hard-code those rules.

## 🤔 Why Deep Learning?

In a standard content-based system, if you like "Pizza," the system looks for the word "Pizza." 

By using **Embedding Layers** in Keras, this model converts high-dimensional categorical data (like City names or Cuisine types) into dense, low-dimensional vectors. This allows the model to:
1.  **Generalize better:** It learns relationships between cuisines during training.
2.  **Handle Sparsity:** It works well even when the data for a specific city/cuisine combo is a bit thin.
3.  **Capture Non-Linearity:** The Dense layers allow it to learn complex patterns (e.g., "High ratings in City A usually mean higher prices, but not in City B").

## ⚙️ How It Works

The architecture is a Multi-Input Neural Network:

1.  **Inputs:** It takes in the **Cuisine**, **City**, and **Price** (normalized).
2.  **Embeddings:** * Cuisines and Cities get passed through their own Embedding layers (dimension=8).
    * This turns a single ID (like `42`) into a vector of 8 numbers representing that item's characteristics.
3.  **The "Brain":**
    * These vectors are flattened and concatenated with the price.
    * They pass through two Dense layers (32 neurons -> 16 neurons) with ReLU activation.
4.  **Output:** A single neuron predicts the "Score" (approximating the rating).

### The "Smart" Fallback System
One specific issue I handled in the code is the **"Empty Result" problem**. If a user asks for *Sushi* in a city that has zero Sushi places, most basic recommenders crash or return nothing.

My `recommend_top_restaurants` function has a waterfall logic:
1.  Tries to find an exact match (Cuisine + City + Price + Rating).
2.  If that fails, it drops the Cuisine filter and looks for *anything* good in that city.
3.  If that fails, it relaxes the price/rating constraints.
4.  As a last resort, it just shows the highest-rated places in the city regardless of criteria.

## 🛠️ Tech Stack

* **Python 3.8+**
* **TensorFlow / Keras** (Model Architecture)
* **Pandas & NumPy** (Data wrangling)
* **Scikit-Learn** (LabelEncoding and MinMaxScaling)

## 🚀 How to Run It

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/your-username/restaurant-recommender.git](https://github.com/your-username/restaurant-recommender.git)
    ```

2.  **Install requirements:**
    ```bash
    pip install pandas numpy tensorflow scikit-learn
    ```

3.  **Check your data:**
    Make sure `enhanced_zomato_dataset_clean.csv` is in the root folder. (The script will error out if it's missing).

4.  **Run the script:**
    ```bash
    python main.py
    ```

5.  **Follow the prompts:**
    The script runs in the terminal. It will ask for your city, cuisine preference, and budget, and then generate the top 5 distinct recommendations.

## 🔮 Future Improvements

If I were to take this further, here is what I'd add:
* **User History:** Right now it's purely content-based. I'd like to add User IDs to make it a true Hybrid Filtering system.
* **NLP on Reviews:** Instead of just using the numerical rating, I could feed the text reviews into a BERT layer to get a "Sentiment Score" as an extra input.
* **Save/Load Model:** Currently, it retrains every time you run the script. I should add `model.save()` functionality to persist the trained weights.

---
