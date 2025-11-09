import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import warnings
from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from implicit.evaluation import ranking_metrics_at_k 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
plt.rcParams['axes.unicode_minus'] = False 
warnings.filterwarnings('ignore')

print("="*70)
print("Hybrid Recommendation System")
print("="*70)

# ============================================================
# 1. Setting and data loading
# ============================================================
print("\n[Step 1] Loading Data")

# Configuration 1: File names
CF_FILE = 'User Listening History.csv'  
CB_FILE = 'Music Info.csv'              

# Configuration 2: Column names for CF data
CF_USER_COL = 'user_id'  
CF_ITEM_COL = 'track_id' 
# If play count is not available, we assume a value of 1 for implicit feedback

# Configuration 3: Column names for CB data
CB_ITEM_COL = 'track_id' 
CB_TAGS_COL = 'tags'     # Genre/tag column

try:
    df_cf = pd.read_csv(CF_FILE)
    df_cb_all = pd.read_csv(CB_FILE)
    print(f"✓ Successfully load '{CF_FILE}' and '{CB_FILE}' ")
except FileNotFoundError:
    print(f"Error: Cannot find file '{CF_FILE}' or '{CB_FILE}'.")
    print("Please check if the file names are correct and files are in the same directory as the code.")
    exit()

# ============================================================
# A. Collaborative Filtering (CF) Model Training
# ============================================================
print("\n" + "="*70)
print("(A) Training Collaborative Filtering (CF) Model using ALS")
print("="*70)

# --- 2. CF Data Preprocessing ---
print("\n[Step A-2] CF Data Preprocessing")

# Standardize column names for internal processing
df_cf = df_cf[[CF_USER_COL, CF_ITEM_COL]].copy()
df_cf.columns = ['user_id', 'item_id']

# # Set play count to 1 (implicit feedback)
df_cf['play_count'] = 1

print(f"  - Number of Original interactions: {len(df_cf):,}")

# Filter data to improve CF model quality
min_user_interactions = 10 # Minimum number of songs a user must have listened to
min_song_interactions = 20 # Minimum number of times a song must have been played

# Calculate the number of occurrences for each unique value in a column
def get_counts(df, col):
    return df.groupby(col).size()

user_counts = get_counts(df_cf, 'user_id')
song_counts = get_counts(df_cf, 'item_id')

# Identify active users and popular songs based on thresholds
active_users = user_counts[user_counts >= min_user_interactions].index
popular_songs = song_counts[song_counts >= min_song_interactions].index

# Filter the dataset to include only active users and popular songs
df_cf_filtered = df_cf[
    df_cf['user_id'].isin(active_users) &
    df_cf['item_id'].isin(popular_songs)
]

print(f"  - Number of Interactions after filtering: {len(df_cf_filtered):,}")

# --- A-3. Split Train/Test Data for Performance Evaluation ---
print("\n[Step A-3] Splitting Train/Test Data (for evaluation)")

def split_train_test_by_user(df, test_size=0.2, random_state=42):
# Split listening history into train/test sets on a per-user basis
    train_list = []
    test_list = []
    
    # Iterate through each user's interaction group
    for _, group in tqdm(df.groupby('user_id'), desc="Splitting Train/Test"):
        if len(group) < 5: # Users with very few interactions go entirely to training
            train_list.append(group)
        else:
            # Split each user's data 80:20
            train_part, test_part = train_test_split(group, test_size=test_size, random_state=random_state)
            train_list.append(train_part)
            test_list.append(test_part)
            
    df_train = pd.concat(train_list)
    df_test = pd.concat(test_list)
    return df_train, df_test

# Create training/test datasets for evaluation
df_train_eval, df_test_eval = split_train_test_by_user(df_cf_filtered, test_size=0.2)
print(f"✓ Evaluation data split: Train {len(df_train_eval):,} / Test {len(df_test_eval):,}")


# --- 3. CF Index Mapping and Sparse Matrix Creation ---
print("\n[Step A-4] CF Index Mapping (based on full dataset)")

unique_users = sorted(df_cf_filtered['user_id'].unique())
unique_items = sorted(df_cf_filtered['item_id'].unique()) # item_id corresponds to track_id

# Create bidirectional mappings between IDs and indices
user_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}
item_to_idx = {sid: idx for idx, sid in enumerate(unique_items)}
idx_to_item = {idx: sid for sid, idx in item_to_idx.items()}

# Apply index mappings to evaluation datasets
df_train_eval['user_idx'] = df_train_eval['user_id'].map(user_to_idx)
df_train_eval['item_idx'] = df_train_eval['item_id'].map(item_to_idx)
df_test_eval['user_idx'] = df_test_eval['user_id'].map(user_to_idx)
df_test_eval['item_idx'] = df_test_eval['item_id'].map(item_to_idx)

# Apply index mappings to full training data
df_cf_filtered['user_idx'] = df_cf_filtered['user_id'].map(user_to_idx)
df_cf_filtered['item_idx'] = df_cf_filtered['item_id'].map(item_to_idx)

n_users = len(unique_users)
n_items = len(unique_items)

print(f"✓ CF mapping: {n_users:,} users × {n_items:,} items")

# Create sparse matrix from full dataset for final model training
user_item_matrix = csr_matrix(
    (df_cf_filtered['play_count'].values,
     (df_cf_filtered['user_idx'].values, df_cf_filtered['item_idx'].values)),
    shape=(n_users, n_items),
    dtype=np.float32
)

# Create sparse matrices for evaluation 
user_item_matrix_train = csr_matrix(
    (df_train_eval['play_count'].values,
     (df_train_eval['user_idx'].values, df_train_eval['item_idx'].values)),
    shape=(n_users, n_items),
    dtype=np.float32
)
user_item_matrix_test = csr_matrix(
    (df_test_eval['play_count'].values,
     (df_test_eval['user_idx'].values, df_test_eval['item_idx'].values)),
    shape=(n_users, n_items),
    dtype=np.float32
)

# --- 4. CF Model (ALS) Training ---
print("\n[Step A-5] Training CF (ALS) Model (using full dataset)")
# Train the model on full dataset
# The implicit library expects an (item, user) matrix format
item_user_matrix = user_item_matrix.T.tocsr()

# 1. Initialize the ALS model
model_cf = implicit.als.AlternatingLeastSquares(
    factors=64,         # Number of latent factors (embedding dimensions)
    regularization=0.1, # L2 regularization to prevent overfitting
    iterations=20,      # Number of training iterations
    alpha=20,           # Confidence weight for implicit feedback 
    random_state=42,
    use_gpu=False       # Set to True if GPU is available
)

# 2. Train the model
print("Starting CF model training (full dataset)...")
model_cf.fit(item_user_matrix, show_progress=True)
print("✓ CF model training complete")

# 3. Extract trained embeddings
song_vectors_cf = model_cf.user_factors  # (n_items, factors)
user_vectors_cf = model_cf.item_factors  # (n_users, factors)

# 4. Display results
print(f"  - User vectors (CF): {user_vectors_cf.shape}")
print(f"  - Song vectors (CF): {song_vectors_cf.shape}")

# ============================================================
# A-5. Define Manual Recommendation Function
# ============================================================
print("\n[Step A-6] Defining CF Manual Recommendation Function")
def recommend_for_user(user_idx, user_vector, all_song_vectors, user_item_matrix, N=20):
    # Calculate predicted scores: dot product of song vectors with user vector
    scores = all_song_vectors @ user_vector  
    # Exclude songs the user has already listened to
    liked_items = user_item_matrix[user_idx].indices
    scores[liked_items] = -np.inf # Set to negative infinity to exclude from recommendations
    # Find top N songs efficiently using partial sorting
    top_indices = np.argpartition(scores, -N)[-N:]
    top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
    top_scores = scores[top_indices]
    return top_indices, top_scores
    
# ============================================================
# B. Content-Based (CB) Model Training 
# ============================================================
print("\n" + "="*70)
print("(B) Training Content-Based (CB) Model using K-Means Clustering")
print("="*70)

# --- 5. CB Data Preprocessing ---
print("\n[Step B-5] CB Data Preprocessing")
# Filter CB data to only include items that appear in CF data
cf_item_ids = set(unique_items)
df_cb = df_cb_all[df_cb_all[CB_ITEM_COL].isin(cf_item_ids)].copy()
# Map track IDs to indices for alignment with CF data
df_cb['item_idx'] = df_cb[CB_ITEM_COL].map(item_to_idx)
df_cb = df_cb.sort_values('item_idx').set_index('item_idx')
# Create aligned DataFrame with all item indices (filling missing entries)
df_cb_aligned = pd.DataFrame(index=range(n_items))
df_cb_aligned = df_cb_aligned.join(df_cb)
print(f"✓ CB data alignment complete. Shape: {df_cb_aligned.shape}")

# --- 6. CB Feature Engineering (Vector Space Model) ---
print("\n[Step B-6] Creating CB Vector Space Model")
# Define feature sets for content-based representation
numeric_features = ['year', 'duration_ms', 'danceability', 'energy', 
                    'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
                    'instrumentalness', 'liveness', 'valence', 'tempo']
text_features = CB_TAGS_COL
# Filter to only use available numeric features
valid_numeric_features = [f for f in numeric_features if f in df_cb_aligned.columns]
print(f"  - Numeric features to be used: {valid_numeric_features}")
# Fill missing values
df_cb_aligned[valid_numeric_features] = df_cb_aligned[valid_numeric_features].fillna(0)
df_cb_aligned[text_features] = df_cb_aligned[text_features].fillna("").astype(str)

# Check if text column has valid tokens (not just empty or special characters)
has_text_tokens = df_cb_aligned[text_features]\
    .str.replace(r"\W+", " ", regex=True)\
    .str.strip()\
    .str.len()\
    .gt(0)\
    .any()

# Create preprocessing pipeline for numeric features (standardization)
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Build list of transformers based on available features
transformers = []
if valid_numeric_features:
    transformers.append(('num', numeric_transformer, valid_numeric_features))

if has_text_tokens:
    # Create TF-IDF vectorizer for text features (tags/genres)
    text_transformer = Pipeline(steps=[('tfidf', TfidfVectorizer(max_features=100))])
    transformers.append(('text', text_transformer, text_features))
else:
    print("'tags' column has no valid text, excluding text features.")

if not transformers:
    raise ValueError("No CB features available: both numeric and text features are empty.")

# Combine transformers into a single preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder='drop' 
)

# Transform CB data into feature vectors
song_content_vectors = preprocessor.fit_transform(df_cb_aligned)
print("✓ CB vector space model creation complete")
print(f"  - Content vectors (CB) Shape: {song_content_vectors.shape}")

# --- 7. CB Model (K-Means) Training ---
print("\n[Step B-7] Training CB (K-Means) Model")
silhouettes = []

# Check number of samples
n_rows = song_content_vectors.shape[0]

if n_rows < 2:
    print("Fewer than 2 songs, skipping K-Means. Setting all to single cluster.")
    song_clusters_cb = np.zeros(n_rows, dtype=int)
else:
    # Memory optimization: Sample subset of data for finding optimal K
    sample_size = min(2000, n_rows)
    sample_indices = np.random.choice(n_rows, sample_size, replace=False)
    subset = song_content_vectors[sample_indices]
    try:
        subset_dense = subset.toarray()  # Sparse to dense conversion
    except AttributeError:
        subset_dense = np.asarray(subset)  # Already dense

    # Determine valid K candidates (must be less than sample size and total items)
    effective_sample = subset_dense.shape[0]
    k_candidates = [k for k in range(2, 16, 2) if k < effective_sample and k < n_rows]
    if not k_candidates:
        # If sample is too small, use safe default
        k_candidates = [min(3, max(2, n_rows - 1))]

    print("  - Finding optimal K using silhouette score...")
    valid_k = []
    for k in tqdm(k_candidates, desc="CB Elbow"):
        try:
            kmeans_temp = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans_temp.fit_predict(subset_dense)
            # Silhouette score requires at least 2 clusters
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(subset_dense, labels)
            silhouettes.append(score)
            valid_k.append(k)
        except Exception as e:
            # Skip if this K value fails
            continue

    if silhouettes:
        # Choose K with highest silhouette score
        optimal_k_cb = valid_k[int(np.argmax(silhouettes))]
        print(f"✓ Optimal K for content-based clustering: {optimal_k_cb}")
    else:
        optimal_k_cb = min(3, max(2, n_rows - 1))
        print(f"No valid silhouette scores obtained. Using safe default K={optimal_k_cb}")

    # Train final K-Means model with optimal K on full dataset
    kmeans_cb = MiniBatchKMeans(n_clusters=optimal_k_cb, random_state=42, n_init=10)
    song_clusters_cb = kmeans_cb.fit_predict(song_content_vectors)
    print("✓ CB model training complete")
    print(f"  - Song clusters (CB) Shape: {song_clusters_cb.shape}") # (n_items,)

# ============================================================
# C. Hybrid Recommendation Demo
# ============================================================
print("\n" + "="*70)
print("(C) Hybrid Recommendation (Demo)")
print("="*70)

# --- 8. Hybrid Approach Application ---
# (A) Cold Start Problem: Recommend for new user based on single song selection
try:
    # Select a sample song with valid features
    sample_item_idx = df_cb_aligned.dropna(subset=valid_numeric_features).index[0]
    sample_item_id = idx_to_item[sample_item_idx]
    print(f"\n--- (A) Cold Start Recommendation Demo ---")
    # Find cluster of selected song
    print(f"  - Reference song: {df_cb_aligned.loc[sample_item_idx]['name']} (ID: {sample_item_id})")
    target_cluster = song_clusters_cb[sample_item_idx]
    print(f"  - Feature cluster: Cluster {target_cluster}")
    # Find all songs in the same cluster (similar content features)
    similar_song_indices = np.where(song_clusters_cb == target_cluster)[0]
    similar_song_indices = [idx for idx in similar_song_indices if idx != sample_item_idx]
    print("\n  - [Content-Based Recommendations (Same Cluster)]")
    # Display 5 random recommendations from the same cluster
    for i, idx in enumerate(np.random.choice(similar_song_indices, 5, replace=False)):
        item_name = df_cb_aligned.loc[idx]['name']
        print(f"    {i+1}. {item_name} (ID: {idx_to_item[idx]})")
except Exception as e:
    print(f"  - (A) Cold Start demo failed: {e}")

# (B) Recommendation Diversity: Enhance variety for existing user
try:
    sample_user_idx = np.random.choice(n_users)
    print(f"\n--- (B) Recommendation Diversity Enhancement Demo ---")
    print(f"  - Reference user: {unique_users[sample_user_idx]}")

    # Get CF-based recommendations (top 20)
    rec_items, rec_scores = recommend_for_user(
        sample_user_idx, 
        user_vectors_cf[sample_user_idx],  
        song_vectors_cf,                   
        user_item_matrix, # Full matrix
        N=20
    )
    
    print("\n  - [Cluster Distribution in CF Top 20 Recommendations]")
    # Analyze cluster distribution of CF recommendations
    rec_clusters = song_clusters_cb[rec_items]
    cluster_dist = Counter(rec_clusters)
    for cid, count in cluster_dist.most_common():
        print(f"    - Cluster {cid}: {count} songs")
        
    # Create diverse playlist by selecting one song from each cluster
    final_playlist = []
    seen_clusters = set()
    for item_idx in rec_items: 
        cluster = song_clusters_cb[item_idx]
        if cluster not in seen_clusters:
            final_playlist.append(item_idx)
            seen_clusters.add(cluster)
            
    print("\n  - [Hybrid Recommendations (Diversity Enhanced)]")
    # Display diversified recommendations (one from each cluster)
    for i, idx in enumerate(final_playlist[:10]): 
        item_name = df_cb_aligned.loc[idx]['name']
        cluster = song_clusters_cb[idx]
        print(f"    {i+1}. {item_name} (Cluster: {cluster})")
except Exception as e:
    print(f"  - (B) Diversity enhancement demo failed: {e}")
    
# ============================================================
# D. Final Recommendation Function (CF Model)
# ============================================================
print("\n" + "="*70)
print("(D) Final Recommendation Function for Specified User (Demo)")
print("="*70)

def get_recommendations_for_user_id(user_id_str, N=10):
    print(f"\n--- 1. Starting recommendations for user '{user_id_str}' ---")
    # Validate user exists in training data
    if user_id_str not in user_to_idx:
        print(f"Error: '{user_id_str}'is not a trained user.")
        return
    user_idx = user_to_idx[user_id_str]
    print(f"✓ User Index: {user_idx}")
    # Get user's latent factor vector
    user_vector = user_vectors_cf[user_idx]
    # Calculate predicted scores for all songs
    scores = song_vectors_cf @ user_vector
    # Exclude songs the user has already listened to
    liked_items_indices = user_item_matrix[user_idx].indices 
    scores[liked_items_indices] = -np.inf
    print(f"✓ Excluded {len(liked_items_indices)} songs already listened to by '{user_id_str}'")
    # Find top N recommendations using efficient partial sorting
    top_indices = np.argpartition(scores, -N)[-N:]
    top_indices = top_indices[np.argsort(scores[top_indices])][::-1]
    top_scores = scores[top_indices]
    print(f"\n--- 2. Top {N} Recommendations for '{user_id_str}'  ---")
    recommendations = []
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        track_id = idx_to_item[idx]
        song_name = df_cb_aligned.loc[idx].get('name', 'Unknown Title')
        if pd.isna(song_name): song_name = 'Unknown Title'
        print(f"  {i+1:2d}. {song_name} (ID: {track_id}) (Score: {score:.4f})")
        recommendations.append((track_id, song_name, score))
    return recommendations

# --- Test the recommendation function ---
random_test_user_id = unique_users[np.random.randint(0, n_users)]
get_recommendations_for_user_id(random_test_user_id, N=10)

# ============================================================
# E. CF Model Performance Evaluation 
# ============================================================
from scipy.sparse import csr_matrix
from implicit.evaluation import ranking_metrics_at_k

print("\n[Step E-0] Train/Test Consistency Check")

n_users_train, n_items_train = user_item_matrix_train.shape
print(f"  - Train matrix: {n_users_train} x {n_items_train}")

# Ensure indices are within valid range by clipping to matrix dimensions
df_test_eval['user_idx'] = df_test_eval['user_idx'].clip(upper=n_users_train - 1)
df_test_eval['item_idx'] = df_test_eval['item_idx'].clip(upper=n_items_train - 1)

# Remove any out-of-range indices
valid_test = df_test_eval[
    (df_test_eval['user_idx'] < n_users_train) &
    (df_test_eval['item_idx'] < n_items_train)
].copy()

print(f"  - Out-of-range indices removed: {len(df_test_eval) - len(valid_test)}")

# Create test matrix with same dimensions as training matrix
user_item_matrix_test = csr_matrix(
    (valid_test['play_count'].values,
     (valid_test['user_idx'].values, valid_test['item_idx'].values)),
    shape=(n_users_train, n_items_train),
    dtype=np.float32
)

# # Initialize and train ALS model for evaluation
model_eval = implicit.als.AlternatingLeastSquares(
    factors=64,
    regularization=0.1,
    iterations=20,
    alpha=20,
    random_state=42,
    use_gpu=False
)

# Transpose to (item, user) format as required by implicit library
item_user_matrix_train = user_item_matrix_train.T.tocsr()
model_eval.fit(item_user_matrix_train, show_progress=True)
print("✓ Model training complete")

# Handle embedding direction mismatch for evaluation
# The implicit library's factor naming can be confusing; swap temporarily for evaluation
_orig_user_factors = model_eval.user_factors
_orig_item_factors = model_eval.item_factors
model_eval.user_factors = _orig_item_factors
model_eval.item_factors = _orig_user_factors

# Verify matrix dimensions
print(f"item_user_matrix_train.shape = {item_user_matrix_train.shape}")
print(f"user_item_matrix_test.shape  = {user_item_matrix_test.shape}")

# Auto-correct dimension mismatch if detected
if item_user_matrix_train.shape[0] != user_item_matrix_test.shape[1]:
    print("Dimension mismatch detected — applying transpose to test matrix")
    user_item_matrix_test = user_item_matrix_test.T.tocsr()

# Perform evaluation at K
K = 10
print(f"\nStarting Top-{K} recommendation performance evaluation...")

try:
    metrics = ranking_metrics_at_k(
        model_eval,
        user_item_matrix_train, # Training set (user, item) format
        user_item_matrix_test,  # Test set (user, item) format
        K=K,
        show_progress=False
    )
    # Restore original embeddings after evaluation
    model_eval.user_factors = _orig_user_factors
    model_eval.item_factors = _orig_item_factors

    # Extract metric values
    p_at_k = metrics['precision']
    map_at_k = metrics['map']     # Mean Average Precision
    ndcg_at_k = metrics['ndcg']   # Normalized Discounted Cumulative Gain
    auc_at_k = metrics['auc']     # Area Under Curve
 
    eval_scores = {
        'Precision': p_at_k,
        'MAP': map_at_k,
        'NDCG': ndcg_at_k,
        'AUC': auc_at_k
    }
except Exception as e:
    # Fallback: Manual evaluation if library function fails
    model_eval.user_factors = _orig_user_factors
    model_eval.item_factors = _orig_item_factors
    print(f"ranking_metrics_at_k failed, performing fallback evaluation: {e}")

    # Calculate precision@k
    def _precision_at_k(recs, truth_set, k):
        if k == 0: return 0.0
        hits = sum((item in truth_set) for item in recs[:k])
        return hits / k

    # Calculate average precision@k
    def _avg_precision_at_k(recs, truth_set, k):
        if k == 0: return 0.0
        hits = 0
        score = 0.0
        limit = min(k, len(recs))
        for i in range(limit):
            if recs[i] in truth_set:
                hits += 1
                score += hits / (i + 1)
        denom = min(k, len(truth_set)) if len(truth_set) > 0 else 1
        return score / denom

    # Calculate NDCG@k
    def _ndcg_at_k(recs, truth_set, k):
        def _dcg(xs):
            dcg = 0.0
            for i, is_rel in enumerate(xs, start=1):
                if is_rel:
                    dcg += 1.0 / np.log2(i + 1)
            return dcg
        rels = [(item in truth_set) for item in recs[:k]]
        dcg = _dcg(rels)
        ideal_rels = [1] * min(len(truth_set), k)
        idcg = _dcg(ideal_rels) if ideal_rels else 1.0
        return dcg / idcg if idcg > 0 else 0.0

    # Sample users who have test interactions
    n_users_eval = user_item_matrix_test.shape[0]
    test_indptr = user_item_matrix_test.indptr
    users_with_test = np.where((test_indptr[1:] - test_indptr[:-1]) > 0)[0]
    if len(users_with_test) == 0:
        print("No test interactions found, skipping evaluation.")
        eval_scores = {'Precision': 0.0, 'MAP': 0.0, 'NDCG': 0.0, 'AUC': 0.0}
    else:
        # Limit evaluation to 5000 users for computational efficiency
        max_users = min(5000, len(users_with_test))
        sample_users = np.random.choice(users_with_test, size=max_users, replace=False)
        precisions = []
        maps = []
        ndcgs = []
        aucs = []
        NEG_SAMPLES = 200 # Number of negative samples for AUC calculation
        for u in tqdm(sample_users, desc=f"Fallback Eval@{K}"):
            # Use original preserved embeddings
            user_embeddings = _orig_item_factors  # Actual user embeddings
            item_embeddings = _orig_user_factors  # Actual item embeddings

            if u >= user_embeddings.shape[0]:
                continue

            # Get test items for this user (ground truth)
            test_items = set(user_item_matrix_test[u].indices.tolist())
            if not test_items:
                continue
            # Manual recommendation: calculate embedding scores and exclude training 
            user_vec = user_embeddings[u]
            scores = item_embeddings @ user_vec
            liked_train = user_item_matrix_train[u].indices

            if len(liked_train) > 0:
                scores[liked_train] = -np.inf
            # Extract top-K recommendations
            k_eff = min(K, scores.shape[0])
            if k_eff <= 0:
                continue
            top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
            top_idx = top_idx[np.argsort(scores[top_idx])][::-1]
            rec_items = list(top_idx[:K])
            # Calculate metrics            
            precisions.append(_precision_at_k(rec_items, test_items, K))
            maps.append(_avg_precision_at_k(rec_items, test_items, K))
            ndcgs.append(_ndcg_at_k(rec_items, test_items, K))

            # AUC calculation (sampling-based): compare positive vs negative item scores
            pos_idx = np.array(list(test_items), dtype=int)
            if pos_idx.size == 0:
                continue
            pos_scores = scores[pos_idx]

            # Negative candidates: sample from items not in training or test
            excluded = set(liked_train.tolist())
            excluded.update(test_items)
            n_items_total = item_embeddings.shape[0]
            neg_idx = []

            # Simple rejection sampling for negative items
            while len(neg_idx) < NEG_SAMPLES:
                draw = np.random.randint(0, n_items_total, size=NEG_SAMPLES)
                for i in draw:
                    if i in excluded:
                        continue
                    neg_idx.append(i)
                    if len(neg_idx) >= NEG_SAMPLES:
                        break

            neg_idx = np.array(neg_idx[:NEG_SAMPLES], dtype=int)
            neg_scores = scores[neg_idx]
            if neg_scores.size == 0:
                continue
            # Efficient AUC calculation: sort negative scores then use binary search
            neg_sorted = np.sort(neg_scores)
            wins_total = 0.0

            for ps in pos_scores:
                left = np.searchsorted(neg_sorted, ps, side='left')
                right = np.searchsorted(neg_sorted, ps, side='right')
                ties = right - left
                wins_total += left + 0.5 * ties
            total_pairs = pos_scores.size * neg_sorted.size
            if total_pairs > 0:
                aucs.append(wins_total / total_pairs)

        # Calculate average metrics across all evaluated users
        p_at_k = float(np.mean(precisions)) if precisions else 0.0
        map_at_k = float(np.mean(maps)) if maps else 0.0
        ndcg_at_k = float(np.mean(ndcgs)) if ndcgs else 0.0
        auc_at_k = float(np.mean(aucs)) if aucs else 0.0
        eval_scores = {
            'Precision': p_at_k,
            'MAP': map_at_k,
            'NDCG': ndcg_at_k,
            'AUC': auc_at_k
        }

# Display evaluation results
print("\n--- Final Evaluation Results ---")
print(f"Precision@{K}: {p_at_k:.4f}")
print(f"MAP@{K}:       {map_at_k:.4f}") 
print(f"NDCG@{K}:      {ndcg_at_k:.4f}")
print(f"AUC@{K}:       {auc_at_k:.4f}")

# ============================================================
# F. Data Visualization
# ============================================================
print("\n" + "="*70)
print("(F) Data Visualization")
print("="*70)

# --- F-1. Content-Based (CB) Cluster Visualization (PCA) ---
print("[Step F-1] CB Cluster Visualization (PCA)")
print("  - Reducing high-dimensional content vectors to 2D to visualize cluster distribution.")

# Apply PCA for dimensionality reduction to 2D
pca_cb = PCA(n_components=2, random_state=42)
# song_content_vectors (from B-6) 
_X_cb = song_content_vectors

# Handle both sparse and dense matrices
try:
    _X_cb_dense = _X_cb.toarray() # Convert sparse to dense
except AttributeError:
    _X_cb_dense = np.asarray(_X_cb) # Already dense
vectors_2d_cb = pca_cb.fit_transform(_X_cb_dense) 

# Create DataFrame for visualization
df_viz_cb = pd.DataFrame({
    'PC1': vectors_2d_cb[:, 0],
    'PC2': vectors_2d_cb[:, 1],
    'cluster': song_clusters_cb # (from B-7)
})

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_viz_cb, 
    x='PC1', 
    y='PC2', 
    hue='cluster', 
    palette='Set2', 
    alpha=0.6, 
    s=30,
    legend='full'
)
plt.title('Content-Based (CB) Cluster 2D Visualization (PCA)', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.legend(title='Cluster ID', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.savefig('visualization_cb_clusters.png')
print("✓ Saved as 'visualization_cb_clusters.png'")


# --- F-2. Collaborative Filtering (CF) Song Vector Visualization (PCA) ---
print("\n[Step F-2] CF Song Vector Visualization (PCA) (Colored by CB Clusters)")
print("  - Examining the relationship between CF latent vectors (behavior-based) and CB clusters (feature-based).")

# Apply PCA for dimensionality reduction to 2D
pca_cf = PCA(n_components=2, random_state=42)
# song_vectors_cf (from A-4)
vectors_2d_cf = pca_cf.fit_transform(song_vectors_cf)

# Create DataFrame (Hybrid: CF vectors colored by CB clusters)
df_viz_cf_hybrid = pd.DataFrame({
    'PC1': vectors_2d_cf[:, 0],
    'PC2': vectors_2d_cf[:, 1],
    'cluster (from CB)': song_clusters_cb # (from B-7)
})

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_viz_cf_hybrid, 
    x='PC1', 
    y='PC2', 
    hue='cluster (from CB)', 
    palette='Set2', 
    alpha=0.6, 
    s=30,
    legend='full'
)
plt.title('Collaborative Filtering (CF) Song Vector 2D Visualization (Colored by CB Clusters)', fontsize=16)
plt.xlabel('PCA Component 1', fontsize=12)
plt.ylabel('PCA Component 2', fontsize=12)
plt.legend(title='CB Cluster ID', loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('visualization_cf_vectors_hybrid.png')
print("✓ Saved as 'visualization_cf_vectors_hybrid.png'")


# --- F-3. Model Performance Visualization---
print("\n[Step F-3] Model Performance Metrics Visualization")
print("  - Displaying evaluation metrics calculated in section (E) as a bar chart.")

df_scores = pd.DataFrame(
    list(eval_scores.items()), 
    columns=['Metric', 'Score']
)

plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=df_scores, 
    x='Metric', 
    y='Score', 
    palette='viridis'
)
# Annotate bars with exact values
for p in barplot.patches:
    barplot.annotate(f'{p.get_height():.4f}', 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha = 'center', va = 'center', 
                     xytext = (0, 9), 
                     textcoords = 'offset points')

plt.title(f'CF Model Performance Evaluation (Top-{K} Recommendations)', fontsize=16)
plt.ylim(0, max(eval_scores.values()) * 1.2) 
plt.xlabel('Evaluation Metric', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.savefig('visualization_evaluation_metrics.png')
print("✓ Saved as 'visualization_evaluation_metrics.png'")

print("\n" + "="*70)
print("All processes complete: Model training, Demo, Performance evaluation, Visualization")
print("="*70)    