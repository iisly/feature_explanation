import json
import pandas as pd
import numpy as np
import umap
import hdbscan
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# 설정
INPUT_FILE = '/content/drive/MyDrive/data_v2.json'
OUTPUT_FILE = 'd3_data_v2.json'
MODEL_NAME = 'all-MiniLM-L6-v2'

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['features'] if isinstance(data, dict) and 'features' in data else data

def get_score(explanation):
    """
    fuzz, detection, embedding 세 가지 지표를 조합하여 최적의 점수를 계산합니다.
    """
    s = explanation.get('scores', {})
    if not isinstance(s, dict):
        return 0

    fuz_score = s.get('fuzz', 0)
    det_score = s.get('detection', 0)
    emb_score = s.get('embedding', 0)

    # 가중치
    final_score = (fuz_score * 0.5) + (det_score * 0.3) + (emb_score * 0.2)

    return final_score

def process():
    features = load_data(INPUT_FILE)
    model = SentenceTransformer(MODEL_NAME)

    processed_data = []
    all_combined_texts = []

    for item in features:
        feature_id = item.get('feature_id', -1)
        explanations = item.get('explanations', [])

        if not explanations:
            continue

        # 점수 순으로 정렬하여 Best Explanation 선정
        sorted_exps = sorted(explanations, key=get_score, reverse=True)
        best_exp = sorted_exps[0]

        display_text = best_exp.get('text', '')
        best_model = best_exp.get('llm_explainer', '')
        best_score = get_score(best_exp)

        enriched_explanations = []
        combined_text_list = []

        for expl in explanations:
            text = expl.get('text', '')
            combined_text_list.append(text)
            vector = model.encode(text).tolist()
            enriched_explanations.append({
                'llm_explainer': expl.get('llm_explainer', ''),
                'text': text,
                'scores': expl.get('scores', {}),
                'vector': vector
            })

        all_combined_texts.append(" ".join(combined_text_list))

        processed_data.append({
            'feature_id': feature_id,
            'display_text': display_text,
            'source_model': best_model,
            'source_score': best_score,
            'all_explanations': enriched_explanations,
            'group_tag': ''
        })

    df = pd.DataFrame(processed_data)

    print("UMAP Reduction...")
    embeddings = model.encode(all_combined_texts, show_progress_bar=True)

    
    # UMAP 기존 코드
    # reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine', random_state=42)
    # 수정본
    reducer = umap.UMAP(
        n_neighbors=10, 
        n_components=2, # 2차원
        min_dist=0.0, 
        metric='cosine', 
        random_state=42,
        n_jobs=-1  
    )
    embedding_2d = reducer.fit_transform(embeddings)

    df['x'] = embedding_2d[:, 0]
    df['y'] = embedding_2d[:, 1]

    print("HDBSCAN Clustering...")
    
    # HDBSCAN 기존 코드
    # clusterer = hdbscan.HDBSCAN(min_cluster_size=3, cluster_selection_method='leaf')
    # 수정본
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=7,
        min_samples=1,
        cluster_selection_epsilon=0.1,
        cluster_selection_method='leaf',
        core_dist_n_jobs=-1 
    )

    df['cluster_id'] = clusterer.fit_predict(embeddings)

    print("Generating Cluster Tags...")
    df['temp_text'] = all_combined_texts
    docs_per_cluster = df[df['cluster_id'] != -1].groupby(['cluster_id'], as_index=False).agg({'temp_text': ' '.join})

    vectorizer = CountVectorizer(stop_words='english', max_features=1)
    cluster_tags = {}

    for _, row in docs_per_cluster.iterrows():
        cid = row['cluster_id']
        try:
            vec = vectorizer.fit([row['temp_text']])
            tag = list(vec.vocabulary_.keys())[0].capitalize()
        except:
            tag = f"Group {cid}"
        cluster_tags[cid] = tag

    df['group_tag'] = df['cluster_id'].map(cluster_tags).fillna("Misc")

    # ==============클러스터 통계================

    # 1. 노이즈(-1) 개수와 실제 클러스터 개수 계산
    n_noise = len(df[df['cluster_id'] == -1])
    n_clusters = len(df['cluster_id'].unique()) - (1 if n_noise > 0 else 0)

    # 2. 클러스터별 태그, 개수 출력
    print("-" * 50)
    print(f"• Total Feature: {len(df)}")
    print(f"• Clusters : {n_clusters}")
    print(f"• Noise : {n_noise} (Misc)")
    print("-" * 50)
    print(f"{'ID':<5} | {'Tag (Label)':<20} | {'Count':<5}")
    print("-" * 50)
    stats = df.groupby(['cluster_id', 'group_tag']).size().reset_index(name='count')
    for _, row in stats.iterrows():
        c_id = row['cluster_id']
        tag = row['group_tag']
        count = row['count']
        
        # 노이즈(-1)는 'Misc'로 표시되거나 별도 표기
        id_str = str(c_id) if c_id != -1 else "Noise"
        print(f"{id_str:<5} | {tag:<20} | {count:<5}")
    # ==========================================

    del df['temp_text']

    final_output = { "data": df.to_dict(orient='records') }
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False)

    print(f"Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process()