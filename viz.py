import os
import pickle
import random
import webbrowser
from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from bertopic import BERTopic
from pyvis.network import Network
from sentence_transformers import SentenceTransformer

# --- Configuration ---
class Config:
    """Groups all configuration parameters for easy access and modification."""
    # --- File I/O ---
    MEMORY_DIR = "sira_memory_v2"
    MEMORY_METADATA_PATH = os.path.join(MEMORY_DIR, "sira_memory_metadata.pkl")
    CSV_OUTPUT_PATH = os.path.join(MEMORY_DIR, "sira_memories.csv")
    HTML_OUTPUT_PATH = "sira_memory_graph_v2.html"
    IMAGE_OUTPUT_PATH = "sira_memory_graph_v2.png" # Path for the static image

    # --- Analysis & Visualization ---
    MODEL_NAME = 'all-MiniLM-L6-v2'
    SIMILARITY_THRESHOLD = 0.50
    MIN_TOPIC_SIZE = 3
    RANDOM_SEED = 42

# --- Utility Functions ---
def print_header(title: str) -> None:
    """Prints a styled header to the console."""
    print("\n" + "="*80)
    print(f"🧠 Sira's Brain Explorer: {title}")
    print("="*80)

def print_status(message: str, emoji: str = "⚙️") -> None:
    """Prints a formatted status message."""
    print(f" {emoji}  {message}")

# --- Core Logic ---
def load_memories(path: str) -> Optional[List[Dict[str, Any]]]:
    """Loads the pickled memory metadata from disk."""
    print_status(f"Attempting to load memories from '{path}'...", "📂")
    if not os.path.exists(path):
        print_status(f"Error: Memory file not found.", "❌")
        return None
    try:
        with open(path, 'rb') as f:
            memories = pickle.load(f)
        print_status(f"Successfully loaded {len(memories)} memories.", "✅")
        return memories
    except (pickle.UnpicklingError, EOFError) as e:
        print_status(f"Error loading pickle file: {e}", "❌")
        return None

def save_memories_to_csv(memories: List[Dict[str, Any]], path: str) -> bool:
    """Converts the list of memory dictionaries to a CSV file."""
    if not memories:
        print_status("No memories to save.", "⚠️")
        return False
    try:
        df = pd.DataFrame(memories)
        df.to_csv(path, index_label="memory_id")
        print_status(f"Memories successfully saved to '{path}'", "✅")
        return True
    except Exception as e:
        print_status(f"Error saving memories to CSV: {e}", "❌")
        return False

def save_static_graph_image(G: nx.Graph, df: pd.DataFrame, community_colors: Dict[int, str], output_path: str):
    """Generates and saves a static PNG image of the graph using Matplotlib."""
    print_status("Generating static graph image with Matplotlib...", "🎨")
    plt.figure(figsize=(20, 20))

    # Use a spring layout for positioning nodes
    pos = nx.spring_layout(G, seed=Config.RANDOM_SEED, k=0.15, iterations=50)

    # Get node colors and sizes from the DataFrame
    df_sorted = df.sort_index()
    node_colors = [community_colors[community_id] for community_id in df_sorted['community']]
    node_sizes = [float(10 + (row['importance'] * 20)) for _, row in df_sorted.iterrows()]

    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, edge_color="grey")
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)

    plt.title("Sira Memory Graph", fontsize=30, color='white')
    plt.box(False)
    ax = plt.gca()
    ax.set_facecolor('#1a1a1a')
    ax.margins(0.1)
    plt.tight_layout()

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#1a1a1a')
        print_status(f"Static graph image saved to '{output_path}'", "✅")
    except Exception as e:
        print_status(f"Failed to save static image: {e}", "❌")
    plt.close()


def generate_interactive_graph():
    """
    Reads memories, performs analysis, and creates both an interactive HTML
    and a static PNG graph.
    """
    # Steps 1, 2, and 3
    print_header("Step 1: Data Loading & Semantic Analysis")
    try:
        df = pd.read_csv(Config.CSV_OUTPUT_PATH)
        memory_texts = df['text'].astype(str).tolist()
        print_status(f"Loaded {len(df)} memories from CSV.", "📄")
    except FileNotFoundError:
        print_status(f"Error: CSV file not found at '{Config.CSV_OUTPUT_PATH}'.", "❌")
        return None

    print_status(f"Encoding memories with '{Config.MODEL_NAME}' model. This may take a moment...", "⏳")
    embedding_model = SentenceTransformer(Config.MODEL_NAME)
    embeddings = embedding_model.encode(memory_texts, show_progress_bar=True)

    print_header("Step 2: Topic Modeling with BERTopic")
    topic_model = BERTopic(verbose=False, min_topic_size=Config.MIN_TOPIC_SIZE, embedding_model=Config.MODEL_NAME)
    df['topic'] = topic_model.fit_transform(memory_texts, embeddings)[0]
    topic_info = topic_model.get_topic_info()
    print_status(f"Identified {len(topic_info) - 1} distinct topics.", "🏷️")

    print_header("Step 3: Building Graph & Detecting Communities")
    print_status("Calculating similarity matrix...", "🧮")
    similarity_matrix = np.dot(embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
                               (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)).T)

    G = nx.Graph()
    for i in range(len(df)):
        G.add_node(i, title=f"Memory {i}")

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if similarity_matrix[i, j] > Config.SIMILARITY_THRESHOLD:
                G.add_edge(i, j, weight=similarity_matrix[i, j])
    print_status(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.", "📊")

    print_status("Detecting communities using the Louvain algorithm...", "🤝")
    communities = nx.community.louvain_communities(G, weight='weight', seed=Config.RANDOM_SEED)
    df['community'] = -1
    for i, comm in enumerate(communities):
        for node_id in comm:
            df.loc[node_id, 'community'] = i
    print_status(f"Detected {len(communities)} distinct communities (thought clusters).", "✅")


    # Step 4: Create Visualizations
    print_header("Step 4: Generating Visualizations")
    net = Network(height="900px", width="100%", bgcolor="#1a1a1a", font_color="white", notebook=False)

    base_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FED766", "#2D3047", "#F2EFEA", "#FFC15E", "#A26769"]
    community_colors = {i: base_colors[i % len(base_colors)] for i in range(len(communities))}
    community_colors[-1] = "#555555"

    for idx, row in df.iterrows():
        node_id = int(row['memory_id'])
        community_id = int(row['community'])
        topic_id = int(row['topic'])

        topic_words = topic_model.get_topic(topic_id)
        topic_label = f"T{topic_id}: {'_'.join([word for word, _ in topic_words][:3])}" if topic_words else f"T{topic_id}: Misc"

        node_title = (f"<div style='font-family: Arial; padding: 5px;'>"
                      f"<b>Memory #{node_id}</b><br>"
                      f"<hr style='margin: 4px 0;'>"
                      f"<b>Cluster ID:</b> C_{community_id}<br>"
                      f"<b>Topic ID:</b> {topic_label}<br>"
                      f"<b>Importance:</b> {float(row['importance']):.2f}<br>"
                      f"<hr style='margin: 4px 0;'>"
                      f"<em>{row['text']}</em></div>")

        net.add_node(node_id,
                     label=topic_label,
                     title=node_title,
                     size=float(12 + (row['importance'] * 2.5)),
                     color=community_colors[community_id],
                     shape='dot')

    for i, j, data in G.edges(data=True):
        weight = float(data['weight'])
        net.add_edge(i, j, value=weight, title=f"Similarity: {weight:.2f}")

    print_status(f"Added {G.number_of_nodes()} nodes and {G.number_of_edges()} edges to the interactive graph.", "✨")

    # This is the corrected options block.
    # It passes a valid JSON string to the function.
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 2,
        "borderWidthSelected": 4,
        "font": {"size": 14, "face": "arial", "color": "#e0e0e0"},
        "shadow": {"enabled": true, "color": "rgba(0,0,0,0.5)", "size": 10, "x": 5, "y": 5}
      },
      "edges": {
        "color": {"inherit": "from", "opacity": 0.5},
        "smooth": {"type": "continuous", "roundness": 0.2},
        "width": 0.5
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.01,
          "springLength": 230,
          "springConstant": 0.08,
          "avoidOverlap": 0.6
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "tooltipDelay": 200
      }
    }
    """)
    net.write_html(Config.HTML_OUTPUT_PATH)
    print_status(f"Interactive graph saved to '{Config.HTML_OUTPUT_PATH}'", "✅")

    # Call the function to generate the static image
    save_static_graph_image(G, df, community_colors, Config.IMAGE_OUTPUT_PATH)

    return Config.HTML_OUTPUT_PATH

def main():
    """Main function to run the entire memory analysis and visualization pipeline."""
    print_header("Initiating Process")
    memories = load_memories(Config.MEMORY_METADATA_PATH)

    if memories and save_memories_to_csv(memories, Config.CSV_OUTPUT_PATH):
        output_file = generate_interactive_graph()
        if output_file:
            print_header("Process Complete")
            print_status(f"Interactive graph saved as '{output_file}'", "🎉")
            print_status(f"Static image saved as '{Config.IMAGE_OUTPUT_PATH}'", "🖼️")
            try:
                webbrowser.open('file://' + os.path.realpath(output_file))
                print_status(f"Attempting to open interactive graph in your browser...", "🌐")
            except Exception as e:
                print_status(f"Could not automatically open the file. Please open it manually. Error: {e}", "⚠️")

if __name__ == "__main__":
    main()