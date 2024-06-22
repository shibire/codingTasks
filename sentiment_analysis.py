import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

def sentiment_analysis(file_name):
    """
    Perform sentiment analysis on product reviews from a CSV file.

    This function processes text by:
    1) dropping NaNs
    3) converting text to lowercase
    4) stripping whitespace.
    5) removes stopwords
    
    It then calculates the polarity and subjectivity of the reviews using TextBlob.
    The function samples a subset of the processed reviews and displays the results.

    Parameters:
    file_path (str): The path to the CSV file containing the product reviews. 
    The CSV file must have a column named 'reviews.text' containing the text of the reviews.

    Returns:
    None: The function prints the processed subset of reviews with their polarity and subjectivity scores.

    Example:
    >>> sentiment_analysis('path_to_your_csv_file.csv')

    Notes:
    - Ensure that a 'reviews.text' column exists in the CSV file.
    - The function displays the processed reviews, their polarity, and subjectivity scores.
    - The pandas display options are temporarily modified to ensure the full output is visible and are reset to default values afterward.
    """

    # Load the spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Step 1: Read in a CSV file
    df = pd.read_csv(file_name)

    # Drop rows with NaN values in the 'reviews.text' column
    df.dropna(subset=['reviews.text'], inplace= True)

    # Step 2: Preprocess the text
    def preprocess_text(text):
        text = text.lower().strip()
        doc = nlp(text)
        filtered_tokens = [token.text for token in doc if token.text not in STOP_WORDS]
        return " ".join(filtered_tokens)

    df['processed_review'] = df['reviews.text'].apply(preprocess_text)

    # Step 3: Create a function for sentiment analysis
    def get_sentiment(text):
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return polarity, subjectivity

    df['polarity'], df['subjectivity'] = zip(*df['processed_review'].apply(get_sentiment))

    # Step 4: Test the model on a subset of the dataset
    subset_df = df.sample(n= 50, random_state= 1)

    # Adjust pandas display options
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)

    # Display the subset with sentiment scores
    print(subset_df[['reviews.text', 'processed_review', 'polarity', 'subjectivity']])

    # Reset pandas display options to their default values
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_colwidth')
    pd.reset_option('display.width')

# Example usage
file_name = 'amazon_product_reviews.csv'
sentiment_analysis(file_name)
