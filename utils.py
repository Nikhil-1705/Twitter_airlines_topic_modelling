# utils.py
import nltk
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of target airlines (lowercase)
AIRLINES = ['southwestair', 'united', 'jetblue', 'americanair', 'usairways', 'virginamerica']

def setup_nltk():
    """Downloads necessary NLTK data if not already present."""
    packages_to_check = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet'
    }

    all_packages_present = True
    packages_to_download = []

    for pkg_name, pkg_path in packages_to_check.items():
        try:
            nltk.data.find(pkg_path)
            logging.debug(f"NLTK package '{pkg_name}' found locally.")
        except LookupError: # <--- Corrected Exception Type
            logging.info(f"NLTK package '{pkg_name}' not found locally.")
            all_packages_present = False
            packages_to_download.append(pkg_name)

    if not all_packages_present:
        logging.info(f"Attempting to download missing NLTK packages: {packages_to_download}...")
        try:
            for pkg_name in packages_to_download:
                 # Use quiet=True to minimize console output during download
                if nltk.download(pkg_name, quiet=True):
                     logging.info(f"Successfully downloaded NLTK package '{pkg_name}'.")
                else:
                    # nltk.download returns False if download failed or was interrupted
                    logging.warning(f"Download failed or was interrupted for NLTK package '{pkg_name}'. Manual download might be needed.")
                    # Optionally raise an error here if the package is critical
                    # raise RuntimeError(f"Failed to download critical NLTK package: {pkg_name}")
        except Exception as e:
            # Catch any other unexpected errors during the download process
            logging.error(f"An error occurred during NLTK package download: {e}")
            logging.error("Please try downloading manually (e.g., python -m nltk.downloader punkt stopwords wordnet)")

    else:
        logging.info("All required NLTK packages are already present.")

    logging.info("NLTK setup check completed.")