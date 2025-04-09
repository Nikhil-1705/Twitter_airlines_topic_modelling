# main.py
import argparse
import logging
import os
import pandas as pd
import plotly.io as pio

# Import project modules
from utils import setup_nltk
from data_loader import load_data
from preprocessing import preprocess_data
from topic_modeling import (
    perform_topic_modeling,
    get_topics_over_time,
    perform_topic_modeling_per_group
)
# Removed: from sentiment_analysis import add_sentiment_analysis
from visualization import (
    visualize_all_topics,
    visualize_grouped_topics
    # Removed: plot_sentiment_distribution
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set default Plotly renderer if needed (can be set per environment)
# pio.renderers.default = 'browser' # Or 'notebook', 'png', 'svg', etc.

def main(args):
    """Main function to orchestrate the Twitter analysis workflow."""

    logging.info("Starting Twitter Analysis Workflow...")

    # --- 1. Setup ---
    logging.info("Setting up NLTK...")
    setup_nltk()

    # --- 2. Load Data ---
    df = load_data(args.input_file)
    if df is None:
        logging.error("Failed to load data. Exiting.")
        return

    # --- 3. Preprocessing ---
    logging.info("Starting data preprocessing...")
    df_processed = preprocess_data(df, min_mention_count=args.min_mentions)
    if df_processed.empty:
        logging.error("Preprocessing resulted in an empty DataFrame. Exiting.")
        return
    logging.info("Preprocessing finished.")

    # Extract cleaned texts and timestamps for modeling
    texts = df_processed['clean_text'].tolist()
    timestamps = df_processed['tweet_created'] if 'tweet_created' in df_processed.columns and pd.api.types.is_datetime64_any_dtype(df_processed['tweet_created']) else None

    # --- 4. Overall Topic Modeling ---
    logging.info("\n--- Performing Overall Topic Modeling ---")
    overall_topic_model, overall_topics, _ = perform_topic_modeling(texts)

    if overall_topic_model and overall_topics is not None and hasattr(overall_topic_model, 'topics_') and overall_topic_model.topics_ is not None:
        logging.info("Overall Topic Modeling Successful.")
        # Analyze topics over time if timestamps are valid
        topics_over_time_df = None
        if timestamps is not None:
             topics_over_time_df = get_topics_over_time(
                 overall_topic_model, texts, timestamps, overall_topics, nr_bins=args.time_bins
             )
        # Visualize overall results
        visualize_all_topics(
            overall_topic_model,
            topics_over_time_df,
            top_n=args.top_n_topics,
            output_dir=args.output_dir,
            prefix="overall"
        )
    else:
        logging.warning("Overall topic modeling failed or produced no topics. Skipping overall visualizations.")

    # --- 5. Topic Modeling Per Airline ---
    logging.info("\n--- Performing Topic Modeling Per Airline ---")
    grouped_topic_models = perform_topic_modeling_per_group(
        df_processed,
        text_column='clean_text', # Use cleaned text
        group_column='airlines_mentioned',
        min_group_size=args.min_mentions # Reuse filter threshold
    )

    if grouped_topic_models:
        # Visualize per-airline results (Barcharts)
        visualize_grouped_topics(
            grouped_topic_models,
            top_n=args.top_n_topics,
            output_dir=args.output_dir
        )
    else:
        logging.warning("No groups met the criteria for per-airline topic modeling.")

    # --- SENTIMENT ANALYSIS SECTION REMOVED ---

    # --- 6. Optional: Save Processed Data ---
    if args.save_processed:
        # Now saves the data after preprocessing and topic modeling ID assignment (if added), but without sentiment
        output_path = os.path.join(args.output_dir, "processed_tweets_no_sentiment.csv")
        try:
            # If you added topic IDs back to the dataframe, df_processed would have them.
            # Otherwise, this saves the dataframe after cleaning and filtering.
            df_processed.to_csv(output_path, index=False)
            logging.info(f"Processed data (without sentiment) saved to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save processed data: {e}")

    logging.info("Twitter Analysis Workflow Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform Topic Modeling on Twitter Data.")
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Path to the input Excel file (.xlsx)."
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="analysis_results",
        help="Directory to save plots and processed data."
    )
    parser.add_argument(
        "--min_mentions",
        type=int,
        default=100,
        help="Minimum number of mentions for an airline to be included in analysis."
    )
    parser.add_argument(
        "--top_n_topics",
        type=int,
        default=10,
        help="Number of top topics to display in visualizations."
    )
    parser.add_argument(
        "--time_bins",
        type=int,
        default=20,
        help="Number of bins for topics over time analysis."
    )
    # Removed sentiment_model argument
    parser.add_argument(
         "--save_processed",
         action='store_true',
         help="Save the final DataFrame after preprocessing to a CSV file."
     )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    main(args)