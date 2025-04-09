# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
from bertopic import BERTopic
import logging
import os
from typing import Optional, Dict

# Set default Plotly renderer (optional, might depend on environment)
# pio.renderers.default = 'browser' # Or 'notebook', 'png', 'svg', 'json', etc.

def save_plot(fig, filename: str, output_dir: str = "plots"):
    """Saves a matplotlib or plotly figure."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filepath = os.path.join(output_dir, filename)
    try:
        # Check if it's a matplotlib figure or axes
        if hasattr(fig, 'savefig'):
            fig.savefig(filepath)
            plt.close(fig) # Close the figure after saving
        # Check if it's a plotly figure
        elif hasattr(fig, 'write_image'):
             # Ensure kaleido is installed: pip install -U kaleido
             try:
                 fig.write_image(filepath)
             except ValueError as ve:
                 logging.warning(f"Plotly image export failed for {filename}. Is 'kaleido' installed? Error: {ve}")
                 fig.write_html(filepath.replace(".png", ".html")) # Fallback to HTML
                 logging.info(f"Saved Plotly figure as HTML: {filepath.replace('.png', '.html')}")
                 return # Exit after saving HTML
        else:
            logging.warning(f"Cannot save figure of type {type(fig)} automatically.")
            return # Cannot save this type

        logging.info(f"Plot saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving plot {filename}: {e}")


def plot_topic_barchart(topic_model: BERTopic, top_n: int = 10, title: str = "Top Topic Word Scores", filename: Optional[str] = "topic_barchart.png", output_dir: str = "plots"):
    """Generates and optionally saves the BERTopic barchart."""
    logging.info(f"Generating topic barchart for top {top_n} topics...")
    try:
        fig = topic_model.visualize_barchart(top_n_topics=top_n, title=title, n_words=5) # Limit words per topic
        fig.show() # Show interactively
        if filename:
            save_plot(fig, filename, output_dir)
    except Exception as e:
        logging.error(f"Error generating/saving topic barchart: {e}")


def plot_topic_hierarchy(topic_model: BERTopic, top_n: int = 10, title: str = "Hierarchical Topic Structure", filename: Optional[str] = "topic_hierarchy.png", output_dir: str = "plots"):
    """Generates and optionally saves the BERTopic hierarchy plot."""
    logging.info(f"Generating topic hierarchy for top {top_n} topics...")
    try:
        # Hierarchical clustering might require recalculation if not done initially
        # Check if hierarchical topics exist, otherwise compute them
        if not hasattr(topic_model, "topic_tree_"):
             logging.info("Hierarchical topics not pre-computed, attempting calculation...")
             # Find associated texts or embeddings if needed by the specific BERTopic version
             # This might require passing texts to hierarchical_topics in some versions
             # For simplicity, assuming it might work without explicit texts here,
             # but adjust if your BERTopic version errors out.
             topic_model.hierarchical_topics()

        fig = topic_model.visualize_hierarchy(top_n_topics=top_n, title=title)
        fig.show()
        if filename:
            save_plot(fig, filename, output_dir)
    except AttributeError:
         logging.warning("`visualize_hierarchy` requires hierarchical topics. Try running `topic_model.hierarchical_topics(docs)` first.")
    except Exception as e:
        logging.error(f"Error generating/saving topic hierarchy: {e}")

def plot_topic_heatmap(topic_model: BERTopic, top_n: int = 10, title: str = "Topic Similarity Heatmap", filename: Optional[str] = "topic_heatmap.png", output_dir: str = "plots"):
    """Generates and optionally saves the BERTopic heatmap."""
    logging.info(f"Generating topic heatmap for top {top_n} topics...")
    try:
        fig = topic_model.visualize_heatmap(top_n_topics=top_n, title=title)
        fig.show()
        if filename:
            save_plot(fig, filename, output_dir)
    except Exception as e:
        logging.error(f"Error generating/saving topic heatmap: {e}")

def plot_topics_over_time(topic_model: BERTopic, topics_over_time_df: pd.DataFrame, top_n: int = 10, title: str = "Topics Over Time", filename: Optional[str] = "topics_over_time.png", output_dir: str = "plots"):
    """Generates and optionally saves the BERTopic topics over time plot."""
    if topics_over_time_df is None or topics_over_time_df.empty:
        logging.warning("No topics over time data to plot.")
        return
    logging.info(f"Generating topics over time plot for top {top_n} topics...")
    try:
        fig = topic_model.visualize_topics_over_time(topics_over_time_df, top_n_topics=top_n, title=title)
        fig.show()
        if filename:
            save_plot(fig, filename, output_dir)
    except Exception as e:
        logging.error(f"Error generating/saving topics over time plot: {e}")

# Removed plot_sentiment_distribution function

def visualize_all_topics(topic_model: BERTopic, topics_over_time_df: Optional[pd.DataFrame] = None, top_n: int = 10, output_dir: str = "plots", prefix: str = "overall"):
     """Runs all standard visualizations for a topic model."""
     plot_topic_barchart(topic_model, top_n=top_n, output_dir=output_dir, filename=f"{prefix}_topic_barchart.png")
     plot_topic_hierarchy(topic_model, top_n=top_n, output_dir=output_dir, filename=f"{prefix}_topic_hierarchy.png")
     plot_topic_heatmap(topic_model, top_n=top_n, output_dir=output_dir, filename=f"{prefix}_topic_heatmap.png")
     if topics_over_time_df is not None:
         plot_topics_over_time(topic_model, topics_over_time_df, top_n=top_n, output_dir=output_dir, filename=f"{prefix}_topics_over_time.png")

def visualize_grouped_topics(grouped_models: Dict[str, BERTopic], top_n: int = 10, output_dir: str = "plots"):
     """Visualizes topic barcharts for each group model."""
     logging.info("Generating topic barcharts for each airline group...")
     for airline, model in grouped_models.items():
         if model and hasattr(model, 'topics_') and model.topics_ is not None: # Added hasattr check
             plot_topic_barchart(
                 model,
                 top_n=top_n,
                 title=f"Top Topics for {airline.capitalize()}",
                 filename=f"{airline}_topic_barchart.png",
                 output_dir=output_dir
             )
         else:
              logging.warning(f"Skipping visualization for {airline} due to invalid or unfitted model.")