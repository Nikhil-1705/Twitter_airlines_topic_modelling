# topic_modeling.py
import pandas as pd
from bertopic import BERTopic
import logging
from typing import List, Dict, Tuple, Optional

def perform_topic_modeling(texts: List[str]) -> Tuple[BERTopic, Optional[List[int]], Optional[List[float]]]:
    """
    Performs BERTopic modeling on a list of texts.

    Args:
        texts: A list of documents (strings) to model.

    Returns:
        A tuple containing:
        - The fitted BERTopic model.
        - List of topic assignments for each document (or None if failed).
        - List of probabilities for topic assignments (or None if failed).
    """
    logging.info(f"Starting topic modeling on {len(texts)} documents...")
    if not texts or len(texts) < 5: # BERTopic might need a minimum number of docs
         logging.warning("Not enough documents to perform topic modeling.")
         return BERTopic(), None, None # Return unfitted model and Nones

    try:
        # Consider adding parameters like embedding_model, umap_model, hdbscan_model for customization
        topic_model = BERTopic(verbose=True)
        topics, probs = topic_model.fit_transform(texts)
        logging.info(f"Topic modeling completed. Found {len(topic_model.get_topic_info()) -1} topics.")
        return topic_model, topics, probs
    except Exception as e:
        logging.error(f"Error during topic modeling: {e}")
        return BERTopic(), None, None # Return unfitted model and Nones

def get_topics_over_time(topic_model: BERTopic,
                         texts: List[str],
                         timestamps: pd.Series,
                         topics: List[int],
                         nr_bins: int = 20) -> Optional[pd.DataFrame]:
    """
    Calculates topic frequencies over time.

    Args:
        topic_model: A fitted BERTopic model.
        texts: The original list of documents.
        timestamps: A pandas Series of timestamps corresponding to the texts.
        topics: The list of topic assignments from topic_model.fit_transform.
        nr_bins: The number of time bins to create.

    Returns:
        A DataFrame containing topics over time data, or None if failed.
    """
    if not topic_model or not texts or timestamps is None or topics is None:
        logging.warning("Missing required data for topics over time analysis.")
        return None
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        logging.warning("'timestamps' column is not datetime type. Cannot calculate topics over time.")
        return None
    if len(texts) != len(timestamps) or len(texts) != len(topics):
        logging.error("Texts, timestamps, and topics lists must have the same length.")
        return None

    logging.info(f"Calculating topics over time with {nr_bins} bins...")
    try:
        topics_over_time_df = topic_model.topics_over_time(
            docs=texts,
            timestamps=timestamps,
            topics=topics, # Pass the pre-computed topics
            nr_bins=nr_bins
        )
        logging.info("Topics over time calculation completed.")
        return topics_over_time_df
    except Exception as e:
        logging.error(f"Error calculating topics over time: {e}")
        return None

def perform_topic_modeling_per_group(df: pd.DataFrame,
                                     text_column: str,
                                     group_column: str,
                                     min_group_size: int = 100) -> Dict[str, BERTopic]:
    """
    Performs BERTopic modeling for each group in a DataFrame column.

    Args:
        df: The input DataFrame.
        text_column: The name of the column containing the text documents.
        group_column: The name of the column to group by (e.g., 'airlines_mentioned').
        min_group_size: Minimum number of documents required in a group to perform modeling.

    Returns:
        A dictionary where keys are group names and values are the fitted BERTopic models.
    """
    logging.info(f"Performing topic modeling per group based on '{group_column}'...")
    grouped_models = {}
    unique_groups = df[group_column].unique()

    for group_name in unique_groups:
        if pd.isna(group_name):
            continue # Skip NaN groups if any
        group_texts = df[df[group_column] == group_name][text_column].tolist()

        if len(group_texts) >= min_group_size:
            logging.info(f"\n--- Modeling for group: {group_name} ({len(group_texts)} documents) ---")
            try:
                # Pass only texts for group-specific modeling
                model, _, _ = perform_topic_modeling(group_texts)
                if model and model.topics_ is not None: # Check if model fitting was successful
                     grouped_models[group_name] = model
                else:
                    logging.warning(f"Topic modeling failed for group: {group_name}")
            except Exception as e:
                 logging.error(f"Error during topic modeling for group {group_name}: {e}")
        else:
            logging.info(f"Skipping group '{group_name}': {len(group_texts)} documents is less than minimum {min_group_size}.")

    logging.info("Finished topic modeling per group.")
    return grouped_models