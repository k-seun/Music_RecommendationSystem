# 🎵 Music Recommendation System  
**Machine Learning Term Project – Team 3**

---

## 🧭 Business Objective  
Build a **hybrid music recommendation system** that generates personalized playlists using both:
- **User listening history** (collaborative signals)  
- **Music features** such as genre, artist, and mood (content signals)

---

## ⚙️ Filtering Methods  
### 1️⃣ Model-based Collaborative Filtering  
- Learns from **implicit user-item interactions** (play history)
- Predicts user preferences for unseen tracks

### 2️⃣ Content-based Filtering  
- Uses **music metadata and audio features** (e.g., genre, artist, tempo)
- Recommends tracks similar to those a user already likes

### → Hybrid Filtering

---

## 📂 Dataset  

### 🏋️ Training Dataset  
**Million Song Dataset + Spotify + Last.fm**  
📎 [Dataset Link](https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm)  

**Files:**
- `User Listening History.csv`: user–track listening data  
- `Music Info.csv`: metadata and audio features  

This dataset merges and reconstructs MSD, Spotify, and Last.fm.

---

### 🧪 Test Dataset  
**Million Song Dataset Challenge**  
📎 [Dataset Link](https://www.kaggle.com/competitions/msdchallenge/overview)  

**Files:**
- `train_triplets.txt`: user–track listening data
- Convert this dataset to use for testing and name it as `User Listening History.csv`

---

## 🧹 Data Preprocessing  

### 🔸 For Model-based Collaborative Filtering
1. **Feature Selection** – Select essential columns: `user_id`, `track_id`  
2. **Implicit Feedback Transformation** – Set `playcount = 1` to consider all interactions with a listening history as “preferred”  
3. **Noise Reduction** – Filter users with ≥10 interactions and tracks with ≥20 listeners  
4. **Label Encoding** – Map string IDs to integer indices  
5. **Sparse Matrix Conversion** – Transform (user, item, playcount) into a **user–item sparse matrix**

---

### 🔸 For Content-based Filtering
1. **Data Alignment** 
2. **Missing Value Imputation** – Replace:
   - Numeric NULL → `0`
   - Categorical NULL → `""` (empty string)
3. **Numeric Feature Scaling** – Apply `StandardScaler`  
4. **Text Feature Vectorization** – Use `TfidfVectorizer` on textual columns (e.g., genre, artist)  
5. **Feature Combination** – Concatenate scaled numeric and text vectors into a **final content feature vector**

---

## 🧠 Modeling  

### 1️⃣ Alternating Least Squares (ALS)
- **Used for:** Model-based Collaborative Filtering  

Since explicit ratings are unavailable, the **playcount** serves as **implicit feedback**.  
The model learns latent user and item representations such that:

`R ≈ X × Yᵀ`

Where:  
- **R** = user–item interaction matrix  
- **X** = user latent factor matrix (n_users × k)  
- **Y** = item latent factor matrix (n_items × k)  
- **k** = number of latent factors  

The **inner product (X · Yᵀ)** represents predicted preference.

**Model Training Example:**
```python
model_cf = implicit.als.AlternatingLeastSquares(
    factors=64,         # Number of latent factors (embedding dimensions)
    regularization=0.1, # L2 regularization to prevent overfitting
    iterations=20,      # Number of training iterations
    alpha=20,           # Confidence weight for implicit feedback 
    random_state=42,
    use_gpu=False       
)
model_cf.fit(item_user_matrix, show_progress=True)
```

### 2️⃣ Clustering (MiniBatch K-Means)
- **Used for:** Content-based Filtering  
- **Process:**
  1. Combine `numeric_features` and `text_features` into `song_content_vectors`
  2. Use **`silhouette_score`** to find the best number of clusters `k`
  3. Determine the cluster of a user's favorite song:
     ```python
     target_cluster = song_clusters_cb[sample_item_idx]
     similar_indices = np.where(song_clusters_cb == target_cluster)[0]
     ```
  4. Recommend another song from the **same cluster** as the user’s favorite song

---

### 🎧 Recommendation Types

#### 1. Content-based Filtering
- Solves the **Cold Start problem**
- Can recommend songs to **new users** (no listening history required)
- Recommends songs with **similar musical characteristics** (e.g., genre, tempo, energy)

#### 2. Collaborative Filtering
- Based on the **user’s listening history**
- The **ALS model** (Alternating Least Squares) finds user preference patterns  
- Calculates **preference scores** for all songs using a **dot product**
- Recommends **Top N songs**, excluding songs the user has already heard

#### 3. Hybrid Filtering (CF + CB)
- Check the **cluster distribution** of Top N songs recommended by CF
- Choose **one song per cluster** to recommend various genres
- Provides **personalized playlists** combining user preferences and content similarity

---

## 📊 Evaluation Metrics

| **Metric**     | **Meaning** |
|-----------------|-------------|
| **Precision@K** | Accuracy of recommendations |
| **MAP@K**       | Average of the average precision for the top K recommendation results |
| **NDCG@K**      | Evaluates the relevance of rankings |
| **AUC**         | Measures ranking quality |

---

## 🧩 Architecture Description
<img width="500" height="550" alt="Image" src="https://github.com/user-attachments/assets/67238cc9-9627-49b2-9db1-7cd4c8f5af21" />

---

## 🏋️ Training Results

```

======================================================================
Hybrid Recommendation System
======================================================================

[Step 1] Loading Data
✓ Successfully load 'User Listening History.csv' and 'Music Info.csv' 

======================================================================
(A) Training Collaborative Filtering (CF) Model using ALS
======================================================================

[Step A-2] CF Data Preprocessing
  - Number of Original interactions: 9,711,301
  - Number of Interactions after filtering: 7,017,360

[Step A-3] Splitting Train/Test Data (for evaluation)
Splitting Train/Test: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 290898/290898 [06:48<00:00, 712.27it/s]
✓ Evaluation data split: Train 5,498,688 / Test 1,518,672

[Step A-4] CF Index Mapping (based on full dataset)
✓ CF mapping: 290,898 users × 20,855 items

[Step A-5] Training CF (ALS) Model (using full dataset)
Starting CF model training (full dataset)...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:28<00:00,  1.42s/it]
✓ CF model training complete
  - User vectors (CF): (290898, 64)
  - Song vectors (CF): (20855, 64)

[Step A-6] Defining CF Manual Recommendation Function

======================================================================
(B) Training Content-Based (CB) Model using K-Means Clustering
======================================================================

[Step B-5] CB Data Preprocessing
✓ CB data alignment complete. Shape: (20855, 21)

[Step B-6] Creating CB Vector Space Model
  - Numeric features to be used: ['year', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
✓ CB vector space model creation complete
  - Content vectors (CB) Shape: (20855, 113)

[Step B-7] Training CB (K-Means) Model
  - Finding optimal K using silhouette score...
CB Elbow: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.17it/s] 
✓ Optimal K for content-based clustering: 2
✓ CB model training complete
  - Song clusters (CB) Shape: (20855,)

======================================================================
(C) Hybrid Recommendation (Demo)
======================================================================

--- (A) Cold Start Recommendation Demo ---
  - Reference song: It's About Time (ID: TRAAAED128E0783FAB)
  - Feature cluster: Cluster 1

  - [Content-Based Recommendations (Same Cluster)]
    1. Keep It Loose, Keep It Tight (ID: TRERUDZ128F1465259)
    2. Sailor's Lament (ID: TRTLXMN128F92D15A4)
    3. Melodia Africana III (ID: TRBZYEE128F92EF884)
    4. Sear Me MCMXCIII (ID: TRPCONI128F42AD92D)
    5. Uschi's Groove (ID: TRIDCZE12903CAC5E0)

--- (B) Recommendation Diversity Enhancement Demo ---
  - Reference user: c39a86c34d7ae4e6aeeabb3e0aa85a701ecbdb91

  - [Cluster Distribution in CF Top 20 Recommendations]
    - Cluster 0: 14 songs
    - Cluster 1: 6 songs

  - [Hybrid Recommendations (Diversity Enhanced)]
    1. Rock And Roll All Nite (Cluster: 0)
    2. Love Comes Tumbling (Cluster: 1)

======================================================================
(D) Final Recommendation Function for Specified User (Demo)
======================================================================

--- 1. Starting recommendations for user '4a692ea4153c4a279799ac53bf8401d39be596d7' ---
✓ User Index: 83831
✓ Excluded 18 songs already listened to by '4a692ea4153c4a279799ac53bf8401d39be596d7'

--- 2. Top 10 Recommendations for '4a692ea4153c4a279799ac53bf8401d39be596d7'  ---
   1. DVNO (ID: TRNPKRK128F429831C) (Score: 0.9006)
   2. Moar Ghosts 'n' Stuff (ID: TRIDQJA12903CE29CC) (Score: 0.8059)
   3. One Minute to Midnight (ID: TRPXIWX128F429831F) (Score: 0.7341)
   4. New Born (ID: TRGVSMR128F42B58E7) (Score: 0.6267)
   5. Plug In Baby (ID: TRULNMQ128F92E1FDA) (Score: 0.6252)
   6. Propane Nightmares (ID: TRSOHMN128F429388A) (Score: 0.6229)
   7. Warp 1.9 (feat. Steve Aoki) (ID: TRVJXHW128F933E2E2) (Score: 0.6114)
   8. Doperide (ID: TRVKJMI128F1490EBA) (Score: 0.6108)
   9. & Down (ID: TRERPOK128F4284833) (Score: 0.5886)
  10. Unnatural Selection (ID: TRRLQIP12903CB78F7) (Score: 0.5803)

[Step E-0] Train/Test Consistency Check
  - Train matrix: 290898 x 20855
  - Out-of-range indices removed: 0
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [00:29<00:00,  1.46s/it]
✓ Model training complete
item_user_matrix_train.shape = (20855, 290898)
user_item_matrix_test.shape  = (290898, 20855)

Starting Top-10 recommendation performance evaluation...

--- Final Evaluation Results ---
Precision@10: 0.1776
MAP@10:       0.0946
NDCG@10:      0.1597
AUC@10:       0.5891

======================================================================
(F) Data Visualization
======================================================================
[Step F-1] CB Cluster Visualization (PCA)
  - Reducing high-dimensional content vectors to 2D to visualize cluster distribution.
✓ Saved as 'visualization_cb_clusters.png'

[Step F-2] CF Song Vector Visualization (PCA) (Colored by CB Clusters)
  - Examining the relationship between CF latent vectors (behavior-based) and CB clusters (feature-based).
✓ Saved as 'visualization_cf_vectors_hybrid.png'

[Step F-3] Model Performance Metrics Visualization
  - Displaying evaluation metrics calculated in section (E) as a bar chart.
✓ Saved as 'visualization_evaluation_metrics.png'

======================================================================
All processes complete: Model training, Demo, Performance evaluation, Visualization
======================================================================

```
## 🏋️ Training Output Plots

### 🎯 Content-Based (CB) Cluster 2D Visualization (PCA)
<img width="500" height="400" alt="Image" src="https://github.com/user-attachments/assets/d410c5df-f978-4c15-b8f1-a1a49068bf75" />

---
### 🤝 Collaborative Filtering (CF) Song Vector 2D Visualization (Colored by CB Clusters)
<img width="500" height="400" alt="Image" src="https://github.com/user-attachments/assets/1ce4f58f-ab37-4615-a9c4-824c7ceade5f" />

---
### 📈 CF Model Performance Evaluation (Top-10 Recommendations)
<img width="500" height="300" alt="Image" src="https://github.com/user-attachments/assets/21f8e062-8434-45b2-8471-c567be324796" />

---

## 🧪 Test Results

```

======================================================================
Hybrid Recommendation System
======================================================================        

[Step 1] Loading Data
✓ Successfully load 'User Listening History.csv' and 'Music Info.csv

======================================================================        
(A) Training Collaborative Filtering (CF) Model using ALS
======================================================================        

[Step A-2] CF Data Preprocessing
  - Number of Original interactions: 48,373,586
  - Number of Interactions after filtering: 46,887,362

[Step A-3] Splitting Train/Test Data (for evaluation)
Splitting Train/Test: 100%|████████| 1019287/1019287 [12:12<00:00, 1391.85it/s]
✓ Evaluation data split: Train 37,104,864 / Test 9,782,498

[Step A-4] CF Index Mapping (based on full dataset)
✓ CF mapping: 1,019,287 users × 161,173 items

[Step A-5] Training CF (ALS) Model (using full dataset)
Starting CF model training (full dataset)...
100%|████████████████████████████████████████| 20/20 [12:41<00:00, 38.08s/it]
✓ CF model training complete
  - User vectors (CF): (1019287, 64)
  - Song vectors (CF): (161173, 64)

[Step A-6] Defining CF Manual Recommendation Function

======================================================================        
(B) Training Content-Based (CB) Model using K-Means Clustering
======================================================================        

[Step B-5] CB Data Preprocessing
✓ CB data alignment complete. Shape: (161173, 21)

[Step B-6] Creating CB Vector Space Model
- Numeric features to be used: ['year', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
'tags' column has no valid text, excluding text features.
✓ CB vector space model creation complete
  - Content vectors (CB) Shape: (161173, 13)

[Step B-7] Training CB (K-Means) Model
  - Finding optimal K using silhouette score...
CB Elbow: 100%|████████████████████████████████| 7/7 [00:00<00:00, 15.45it/s] 
No valid silhouette scores obtained. Using safe default K=3
✓ CB model training complete
  - Song clusters (CB) Shape: (161173,)

======================================================================        
(C) Hybrid Recommendation (Demo)
======================================================================        

--- (A) Cold Start Recommendation Demo ---
  - Reference song: nan (ID: SOAAAFI12A6D4F9C66)
  - Feature Cluster: Cluster 0

  - [Content-Based Recommendations (Same Cluster)]
    1. nan (ID: SOJGLPC12A8C13B77E)
    2. nan (ID: SOYDVHE12AF72A058E)
    3. nan (ID: SOCBOZX12A58A7A93A)
    4. nan (ID: SOOTWNU12AB0180A68)
    5. nan (ID: SOYJUAE12A67021CBC)

--- (B) Recommendation Diversity Enhancement Demo ---
  - Reference user: d9949a154634ad275fcff2d7ce9bbfbb06668d9a

  - [Cluster Distribution in CF Top 20 Recommendations]
    - Cluster 0: 20 songs

  - [Hybrid Recommendations (Diversity Enhanced)]
    1. nan (Cluster: 0)

======================================================================
(D) Final Recommendation Function for Specified User (Demo)
======================================================================

--- 1. Starting recommendations for user '21088384fa5aa0ab4c11d0ef940a43b176844a1a' ---
✓ User Index: 130794
✓ Excluded 111 songs already listened to by '21088384fa5aa0ab4c11d0ef940a43b176844a1a'

--- 2. Top 10 Recommendations for '21088384fa5aa0ab4c11d0ef940a43b176844a1a' ---
   1. Unknown Title (ID: SOFSPAT12A8C145F53) (Score: 1.0971)
   2. Unknown Title (ID: SOVUBST12AB018C9A4) (Score: 1.0024)
   3. Unknown Title (ID: SOJTIXE12AB018C99E) (Score: 0.9916)
   4. Unknown Title (ID: SOTEPSZ12AB018C99D) (Score: 0.9842)
   5. Unknown Title (ID: SOEKVCJ12AB0185E18) (Score: 0.9508)
   6. Unknown Title (ID: SOPBXPQ12AB01887E2) (Score: 0.9492)
   7. Unknown Title (ID: SOKTJMZ12AB018C9A0) (Score: 0.9452)
   8. Unknown Title (ID: SOGECRB12AB018C9AB) (Score: 0.9223)
   9. Unknown Title (ID: SOMVMVF12AB018C9A6) (Score: 0.9169)
  10. Unknown Title (ID: SOPXBSU12AB018917D) (Score: 0.8971)

[Step E-0] Train/Test Consistency Check
  - Train matrix: 1019287 x 161173
  - Out-of-range indices removed: 0
100%|████████████████████████████████████████| 20/20 [12:26<00:00, 37.32s/it]
✓ Model training complete
item_user_matrix_train.shape = (161173, 1019287)
user_item_matrix_test.shape  = (1019287, 161173)

Starting Top-10 recommendation performance evaluation...

--- Final Evaluation Results ---
Precision@10: 0.1084
MAP@10:       0.0568
NDCG@10:      0.1060
AUC@10:       0.5481

======================================================================
(F) Data Visualization
======================================================================

[Step F-1] CB Cluster Visualization (PCA)
  - Reducing high-dimensional content vectors to 2D to visualize cluster distribution.
✓ Saved as 'visualization_cb_clusters.png'

[Step F-2] CF Song Vector Visualization (PCA) (Colored by CB Clusters)
  - Examining the relationship between CF latent vectors (behavior-based) and CB clusters (feature-based).
✓ Saved as 'visualization_cf_vectors_hybrid.png'

[Step F-3] Model Performance Metrics Visualization
  - Displaying evaluation metrics calculated in section (E) as a bar chart.
✓ Saved as 'visualization_evaluation_metrics.png'

======================================================================
All processes complete: Model training, Demo, Performance evaluation, Visualization
======================================================================

```

## 🧪 Test Output Plots

### 🎯 Content-Based (CB) Cluster 2D Visualization (PCA)
<img width="500" height="400" alt="Image" src="https://github.com/user-attachments/assets/3d79d21f-c15a-44f6-b26d-59c607400f7a" />

---

### 🤝 Collaborative Filtering (CF) Song Vector 2D Visualization (Colored by CB Clusters)
<img width="500" height="400" alt="Image" src="https://github.com/user-attachments/assets/9bfdcb69-cb30-4695-96ce-db8bd86cd948" />

---

### 📈 CF Model Performance Evaluation (Top-10 Recommendations)
<img width="500" height="300" alt="Image" src="https://github.com/user-attachments/assets/6db0ee5c-febf-42a9-9aa2-d8c0b039e0e3" />
