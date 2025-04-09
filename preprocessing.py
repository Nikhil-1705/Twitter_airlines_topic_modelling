# preprocessing.py
import pandas as pd
import re
import string
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from utils import AIRLINES # Import AIRLINES constant

# Initialize outside functions for efficiency
stop_words = None
lemmatizer = None

def _initialize_nltk_resources():
    """Initializes NLTK resources if not already done."""
    global stop_words, lemmatizer
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()

def extract_airlines(text: str) -> str | None:
    """
    Extracts airlines mentioned in the tweet text. Returns comma-separated string.
    """
    if not isinstance(text, str):
        return None
    text_lower = text.lower()
    found_airlines = [airline for airline in AIRLINES if airline in text_lower]
    return ", ".join(found_airlines) if found_airlines else None

def merge_usairways_american(value: str | None) -> str | None:
    """
    Merges 'usairways' and 'americanair' mentions into 'usairways'.
    """
    if not isinstance(value, str):
        return value

    airlines_list = [air.strip() for air in value.split(',')]
    # Replace 'americanair' with 'usairways'
    airlines_list = ['usairways' if air == 'americanair' else air for air in airlines_list]
    # Remove duplicates and sort for consistency
    merged_airlines = sorted(list(set(airlines_list)))
    return ', '.join(merged_airlines)

def process_airline_mentions(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Adds and processes the 'airlines_mentioned' column.
    """
    logging.info("Extracting and processing airline mentions...")
    df['airlines_mentioned'] = df[text_column].apply(extract_airlines)
    df['airlines_mentioned'] = df['airlines_mentioned'].apply(merge_usairways_american)
    logging.info("Finished processing airline mentions.")
    # Log value counts for diagnostics
    if 'airlines_mentioned' in df.columns:
        logging.info("Value counts for processed airlines:\n" + str(df['airlines_mentioned'].value_counts().head()))
    return df

def filter_by_airline_mentions(df: pd.DataFrame, min_mention_count: int = 100) -> pd.DataFrame:
    """
    Filters the DataFrame to keep only airlines mentioned at least min_mention_count times.
    """
    logging.info(f"Filtering airlines with less than {min_mention_count} mentions...")
    if 'airlines_mentioned' not in df.columns:
        logging.error("Column 'airlines_mentioned' not found for filtering.")
        return df

    value_counts = df['airlines_mentioned'].value_counts()
    top_airlines = value_counts[value_counts >= min_mention_count].index.tolist()

    original_count = len(df)
    filtered_df = df[df['airlines_mentioned'].isin(top_airlines)].copy() # Use .copy() to avoid SettingWithCopyWarning
    filtered_count = len(filtered_df)
    retained_percentage = (filtered_count / original_count) * 100 if original_count > 0 else 0

    logging.info(f"Original dataset size: {original_count}")
    logging.info(f"Filtered dataset size: {filtered_count}")
    logging.info(f"Percentage of data retained after filtering: {retained_percentage:.2f}%")
    logging.info(f"Airlines kept: {top_airlines}")
    return filtered_df

def clean_text(text: str) -> str:
    """
    Cleans the input text: lowercase, remove mentions/URLs/punctuation/numbers,
    tokenize, remove stopwords, lemmatize.
    """
    _initialize_nltk_resources() # Ensure resources are loaded

    if not isinstance(text, str):
        return ""

    # 1) Normalization
    text = text.lower()
    # 2) Remove mentions (@user) and URLs
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    # 3) Remove punctuation and numeric characters
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\d+', '', text)
    # 4) Tokenization
    tokens = word_tokenize(text)
    # 5) Remove stopwords and short words
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    # 6) Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

def add_cleaned_text(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    Applies text cleaning to the specified column and adds a 'clean_text' column.
    """
    logging.info(f"Applying text cleaning to column '{text_column}'...")
    _initialize_nltk_resources() # Ensure resources are ready
    df['clean_text'] = df[text_column].apply(clean_text)
    logging.info("Finished text cleaning.")
    return df

def preprocess_data(df: pd.DataFrame, text_column: str = 'text', min_mention_count: int = 100) -> pd.DataFrame:
    """
    Runs the full preprocessing pipeline.
    """
    df = process_airline_mentions(df, text_column)
    df = filter_by_airline_mentions(df, min_mention_count)
    df = add_cleaned_text(df, text_column)
    # Convert 'tweet_created' to datetime if it exists and is not already datetime
    if 'tweet_created' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['tweet_created']):
         try:
             df['tweet_created'] = pd.to_datetime(df['tweet_created'])
             logging.info("Converted 'tweet_created' column to datetime.")
         except Exception as e:
             logging.warning(f"Could not convert 'tweet_created' to datetime: {e}. Skipping time-based analysis.")
             # Ensure it exists but maybe set to None or NaT if conversion fails and it's needed
             if 'tweet_created' in df.columns:
                 df['tweet_created'] = pd.NaT
    return df