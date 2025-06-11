import os
import pickle
import numpy as np
import pandas as pd
import webbrowser
import networkx as nx
import random

from sentence_transformers import SentenceTransformer
from pyvis.network import Network
from bertopic import BERTopic

# --- Configuration ---
MEMORY_DIR = "sira_memory_v2"
MEMORY_METADATA_PATH = os.path.join(MEMORY_DIR, "sira_memory_metadata.pkl")
CSV_OUTPUT_PATH = os.path.join(MEMORY_DIR, "sira_memories.csv")
HTML_OUTPUT_PATH = "sira_memory_graph_v2.html"

# --- Analysis & Visualization Parameters ---
MODEL_NAME = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.50 # Lowered slightly to help form communities
MIN_TOPIC_SIZE = 3 # A topic must contain at least this many memories

def load_memories():
    """Loads the pickled memory metadata from disk."""
    if not os.path.exists(MEMORY_METADATA_PATH):
        print(f"Error: Memory file not found at '{MEMORY_METADATA_PATH}'")
        return None
    with open(MEMORY_METADATA_PATH, 'rb') as f:
        memories = pickle.load(f)
    print(f"✅ Successfully loaded {len(memories)} memories from pickle.")
    return memories

def save_memories_to_csv(memories):
    """Converts the list of memory dictionaries to a CSV file."""
    if not memories: return False
    df = pd.DataFrame(memories)
    df.to_csv(CSV_OUTPUT_PATH, index_label="memory_id")
    print(f"✅ Memories successfully saved to '{CSV_OUTPUT_PATH}'")
    return True

def create_enhanced_graph_from_csv():
    """Reads memories, performs topic/community analysis, and creates an enhanced network graph."""
    # --- Step 1: Load Data ---
    print("\n--- Reading data from CSV ---")
    try:
        df = pd.read_csv(CSV_OUTPUT_PATH)
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{CSV_OUTPUT_PATH}'.")
        return None
    
    memory_texts = df['text'].astype(str).tolist()

    # --- Step 2: Topic Modeling with BERTopic ---
    print("\n--- Step 1 of 4: Finding topics with BERTopic ---")
    # Using a pre-computed embedding model is faster
    embedding_model = SentenceTransformer(MODEL_NAME)
    embeddings = embedding_model.encode(memory_texts, show_progress_bar=True)

    topic_model = BERTopic(verbose=False, min_topic_size=MIN_TOPIC_SIZE, embedding_model=MODEL_NAME)
    df['topic'] = topic_model.fit_transform(memory_texts, embeddings)[0]
    
    topic_info = topic_model.get_topic_info()
    print(f"✅ Found {len(topic_info) - 1} distinct topics.")

    # --- Step 3: Build Graph and Detect Communities ---
    print("\n--- Step 2 of 4: Building graph structure & calculating similarity ---")
    similarity_matrix = np.dot(embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True), (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)).T)

    print("\n--- Step 3 of 4: Detecting communities with Louvain algorithm ---")
    # Create a NetworkX graph to run community detection
    G = nx.Graph()
    for i in range(len(df)):
        G.add_node(i)

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
                G.add_edge(i, j, weight=similarity_matrix[i, j])

    communities = nx.community.louvain_communities(G, weight='weight', seed=42)
    df['community'] = -1
    for i, comm in enumerate(communities):
        for node_id in comm:
            df.loc[node_id, 'community'] = i
    print(f"✅ Detected {len(communities)} communities (thought clusters).")

    # --- Step 4: Create Interactive Visualization with PyVis ---
    print("\n--- Step 4 of 4: Creating final interactive visualization ---")
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white", notebook=False)

    # Generate a color for each community
    community_colors = {i: f"#{random.randint(0, 0xFFFFFF):06x}" for i in range(len(communities))}
    community_colors[-1] = "#cccccc" # Color for nodes not in a community

    for idx, row in df.iterrows():
        node_id = int(row['memory_id'])
        topic_id = int(row['topic'])
        community_id = int(row['community'])
        
        # Get a readable name for the topic
        topic_name = topic_model.get_topic(topic_id)
        topic_label = f"{topic_id}: {'_'.join([word for word, _ in topic_name][:4])}" if topic_name else f"{topic_id}: N/A"

        node_title = (f"<b>Memory ID:</b> {node_id}<br>"
                    f"<b>Community:</b> {community_id}<br>"
                    f"<b>Topic:</b> {topic_label}<br>"
                    f"<b>Importance:</b> {row['importance']:.2f}<br>"
                    f"<hr><b>Text:</b> {row['text']}")
        
        net.add_node(node_id,
                     label=topic_label,
                     title=node_title,
                     size=10 + (row['importance'] * 2),
                     color=community_colors[community_id])

    # Add edges from the NetworkX graph
    for i, j in G.edges():
        net.add_edge(i, j)

    print(f"✅ Added {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to the final graph.")
    
    net.set_options("""
    var options = {
      "nodes": {"borderWidth": 2, "font": {"size": 14, "face": "arial", "color": "#ffffff"}},
      "edges": {"color": {"inherit": false, "opacity": 0.4}, "smooth": false},
      "physics": {"forceAtlas2Based": {"gravitationalConstant": -70, "centralGravity": 0.015, "springLength": 200, "avoidOverlap": 0.5}, "minVelocity": 0.75, "solver": "forceAtlas2Based"}
    }
    """)
    net.write_html(HTML_OUTPUT_PATH)
    return HTML_OUTPUT_PATH

def main():
    """Main function to run the conversion and visualization process."""
    print("--- Sira's Brain Advanced Visualization Tool ---")
    
    memories = load_memories()
    if memories and save_memories_to_csv(memories):
        output_file = create_enhanced_graph_from_csv()
        if output_file:
            print("\n--- ✅ Success! ---")
            print(f"Interactive graph saved as '{output_file}'")
            try:
                webbrowser.open('file://' + os.path.realpath(output_file))
                print(f"Attempting to open '{output_file}' in your web browser.")
            except Exception as e:
                print(f"Could not automatically open the file. Please open it manually. Error: {e}")

if __name__ == "__main__":
    main()