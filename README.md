# ✈️ Twitter Airline Topic Explorer ✈️

Ever wondered what people *really* talk about when they tweet at airlines? This project dives into a dataset of airline tweets to automatically discover the main conversation topics! 🗣️

Using the magic of **BERTopic**, we clean up messy tweets, figure out which airlines are being mentioned, and then group conversations into themes like "Lost Baggage Blues," "Waiting Game Woes," or "WiFi Wonders (or lack thereof!)."

## ✨ What Does It Do? ✨

*   **Loads Data:** Reads tweets from an Excel file.
*   **Cleans Tweets:** Gets rid of mentions (`@user`), URLs, punctuation, and other noise to focus on the actual message. 🧹
*   **Finds Airlines:** Identifies which airline(s) (like `@united`, `@jetblue`) are mentioned in each tweet.
*   **Focuses on the Big Players:** Filters out airlines that aren't mentioned very often (you can set the minimum!).
*   **Discovers Topics (Overall):** Uses BERTopic to find the main conversation themes across *all* the tweets.
*   **Discovers Topics (Per Airline):** Runs BERTopic *separately* for each major airline to see their specific hot topics. 🔥
*   **Visualizes Insights:** Creates cool charts and plots to show:
    *   What words define each topic (Bar Charts).
    *   How topics relate to each other (Hierarchy & Heatmap - *Overall only*).
    *   How popular topics were over time (Line Chart - *Overall only*).
    *   The main topics for each individual airline (Bar Charts).
*   **Saves Results:** Stores the generated plots and (optionally) the processed data in a neat folder.

## ⚙️ Setup: Get Ready for Takeoff! ⚙️

1.  **Python:** Make sure you have Python installed (Version 3.9 or higher recommended).
2.  **Get the Code:** Clone this repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```
3.  **Create a Virtual Space (Recommended):** Keep things tidy!
    *   On macOS/Linux:
        ```bash
        python3 -m venv .venv
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv .venv
        .\.venv\Scripts\activate
        ```
4.  **Install the Goodies:** Install all the necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```
    *(This might take a few minutes, especially for libraries like `torch` and `transformers`)*.
5.  **NLTK Data (First Run):** The first time you run the script, it might need to download some language data from NLTK. It should handle this automatically!

## 🚀 How to Run the Analysis 🚀

The main script to run everything is `main.py`. You run it from your terminal.

**The Basic Command:**

```bash
python main.py --input_file "path/to/your/Worksheet in Candidiate Screening Assignment_Associate Data Scientist (002).xlsx"
Use code with caution.
Markdown
--input_file (Required): You must tell the script where your Excel file is located. Replace the example path with the actual path to your file.
Optional Fun Settings (Command-Line Arguments):
You can customize the run using these optional flags:
--output_dir <folder_name>: Where should the plots and results be saved?
Example: python main.py ... --output_dir "My_Awesome_Results"
Default: analysis_results
--min_mentions <number>: Only analyze airlines mentioned at least this many times.
Example: python main.py ... --min_mentions 50
Default: 100
--top_n_topics <number>: How many top topics should be shown in the visualizations?
Example: python main.py ... --top_n_topics 8
Default: 10
--time_bins <number>: How many time periods to split the data into for the "Topics Over Time" plot?
Example: python main.py ... --time_bins 15
Default: 20
--save_processed: Add this flag if you want to save the cleaned-up data (without sentiment) to a CSV file in the output directory.
Example: python main.py ... --save_processed
Example with Options:
python main.py --input_file "data/tweets.xlsx" --output_dir "airline_analysis_v1" --min_mentions 75 --top_n_topics 12 --save_processed
Use code with caution.
Bash
📁 Project Structure 📁
main.py: The main script that runs the whole show.
data_loader.py: Handles loading the data from the Excel file.
preprocessing.py: Does all the text cleaning and airline mention processing.
topic_modeling.py: Contains the logic for running BERTopic.
visualization.py: Creates all the plots and charts. 📊
utils.py: Small helper functions (like setting up NLTK).
requirements.txt: A list of all the Python libraries needed.
README.md: This file! You are here.📍
data/ (Optional): A suggested place to put your input Excel file.
analysis_results/ (Default Output): Where your plots and saved CSV will appear.
🎉 Expected Output 🎉
After running, check the folder you specified with --output_dir (or analysis_results by default). You should find:
Plots! Several .png files (or .html if image export fails) showing:
overall_topic_barchart.png: Top words for overall topics.
overall_topic_hierarchy.png: How overall topics might be related.
overall_topic_heatmap.png: Similarity between overall topics.
overall_topics_over_time.png: Trend of overall topics over time (if tweet_created data was valid).
<airline_name>_topic_barchart.png: Top words for topics specific to each major airline.
Processed Data (Optional): If you used --save_processed, a file named processed_tweets_no_sentiment.csv containing the cleaned data.
Ready to explore the world of airline tweets? Happy analyzing! 😄
**Key changes for "easy and fun":**

*   **Emojis:** Added relevant emojis (✈️, 🗣️, 🧹, 🔥, 📊, ⚙️, 🚀, 📁, 🎉, 📍) to break up text and add visual interest.
*   **Simple Language:** Used phrases like "magic of BERTopic," "gets rid of," "keep things tidy," "install the goodies," "runs the whole show," "cool charts."
*   **Clear Headings:** Used markdown headings (`##`, `###`) with descriptive titles.
*   **Emphasis:** Used **bold** text for key terms and file names.
*   **Structured Lists:** Used bullet points for features, setup steps, and file structure.
*   **Code Blocks:** Clearly formatted commands using ```bash ... ```.
*   **Explanations:** Briefly explained *why* certain steps are done (e.g., virtual environment, NLTK downloads).
*   **Call to Action:** Ended with an encouraging message.
