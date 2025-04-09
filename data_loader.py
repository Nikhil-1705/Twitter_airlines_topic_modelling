# data_loader.py
import pandas as pd
import logging
import os

def load_data(file_path: str) -> pd.DataFrame | None:
    """
    Loads data from an Excel file into a pandas DataFrame.

    Args:
        file_path: The path to the Excel file.

    Returns:
        A pandas DataFrame containing the loaded data, or None if loading fails.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    try:
        df = pd.read_excel(file_path)
        logging.info(f"Successfully loaded data from {file_path}. Shape: {df.shape}")
        # Ensure 'text' column exists and handle potential read errors
        if 'text' not in df.columns:
            logging.error("'text' column not found in the Excel file.")
            return None
        # Convert text column to string type to avoid errors later
        df['text'] = df['text'].astype(str)
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return None