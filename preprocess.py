import pandas as pd 
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

# Download nltk resource 
# nltk.download("punkt")
# nltk.download('stopwords')

# Loading the dataset 
df = pd.read_csv('ipc_sections.csv')

def preprocess_text(text):
    if pd.isna(text):  # Check for NaN values
        return []
    # tokenisation 
    tokens = word_tokenize(text.lower())  # converting the text to lowercase 

    # removing stopwords 
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# preprocessing the description of IPC section 
df['preprocessed_description'] = df['Description'].apply(preprocess_text)

# preprocessing offense 
df['preprocessed_offense'] = df['Offense'].apply(preprocess_text)

df.to_csv('preprocessed_data.csv', index= False)