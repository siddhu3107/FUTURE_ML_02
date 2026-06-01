import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text, product_name=None):
        if not isinstance(text, str):
            return ""
        
        # Replace placeholders if product_name is provided
        if product_name:
            text = text.replace('{product_purchased}', product_name)
        else:
            text = text.replace('{product_purchased}', 'product')

        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words and token.isalpha()
        ]
        
        return " ".join(cleaned_tokens)

if __name__ == "__main__":
    # Test
    preprocessor = TextPreprocessor()
    sample = "I'm having an issue with the {product_purchased}. Please assist."
    print(f"Original: {sample}")
    print(f"Cleaned: {preprocessor.clean_text(sample, 'GoPro Hero')}")
