import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import nltk
import re
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv('IMDB Dataset.csv')

# Preprocessing: Clean, tokenize, handle negation, and lemmatize
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Handle negation words: "not happy" -> "not_happy"
    text = re.sub(r"\bnot\s(\w+)", r"not_\1", text)
    
    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    
    # Lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Apply preprocessing to dataset
df['cleaned_review'] = df['review'].apply(clean_text)

# Split dataset
X = df['cleaned_review']
y = df['sentiment'].map({'positive': 1, 'negative': 0})  # Encode labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model pipeline using TF-IDF and Logistic Regression
model = make_pipeline(TfidfVectorizer(), LogisticRegression(class_weight='balanced'))

# Cross-validation for performance evaluation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f'Cross-validation accuracy: {cv_scores.mean():.4f}')

# Train the model
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'sentiment_model.pkl')

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
