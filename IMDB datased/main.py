import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import requests

# Download necessary NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("NLTK resources could not be downloaded. Using simple preprocessing instead.")

# Function to load JSON data into a pandas DataFrame
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

# Load the train and test datasets
train_df = load_json_data('aclImdb/train.json')
test_df = load_json_data('aclImdb/test.json')

# Enhanced text preprocessing function
def enhanced_preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove special characters and digits but keep important punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    try:
        # Remove stopwords and lemmatize
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
        
        return ' '.join(words)
    except:
        # If NLTK processing fails, use simpler approach
        return text

# Apply enhanced preprocessing
print("Preprocessing training data...")
train_df['text'] = train_df['text'].apply(enhanced_preprocess_text)

print("Preprocessing test data...")
test_df['text'] = test_df['text'].apply(enhanced_preprocess_text)

# TF-IDF feature extraction with optimized parameters
print("Extracting features...")
vectorizer = TfidfVectorizer(
    max_features=15000,  # Increase features
    min_df=2,           # Minimum document frequency
    max_df=0.85,        # Maximum document frequency
    ngram_range=(1, 2), # Use both unigrams and bigrams
    sublinear_tf=True   # Apply sublinear tf scaling
)

X_train = vectorizer.fit_transform(train_df['text'])
X_test = vectorizer.transform(test_df['text'])

# Labels
y_train = train_df['label'].values
y_test = test_df['label'].values

# Optimized Logistic Regression with hyperparameter tuning
print("Training model with hyperparameter tuning...")
param_grid = {
    'C': [0.1, 1.0, 5.0, 10.0],  # Regularization parameter
    'solver': ['liblinear', 'saga'],
    'class_weight': [None, 'balanced']
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000),
    param_grid,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# Get the best model
best_model = grid.best_estimator_
print(f"Best parameters: {grid.best_params_}")

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer together as a pipeline
from sklearn.pipeline import Pipeline
sentiment_pipeline = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', best_model)
])

# Save the pipeline
joblib.dump(sentiment_pipeline, 'skills_assessment.joblib')

# Function to upload the model to the evaluation portal
def upload_model_to_portal(model_file_path):
    url = "http://10.129.205.188:5000/api/upload"
    
    with open(model_file_path, 'rb') as model_file:
        files = {"model": model_file}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print("Model uploaded successfully.")
                print(json.dumps(response.json(), indent=4))
            else:
                print(f"Error: Received status code {response.status_code}")
                print(response.text)
        except requests.exceptions.RequestException as e:
            print(f"Error uploading model: {e}")

# Upload the model
print("Uploading model to evaluation portal...")
upload_model_to_portal('skills_assessment.joblib')