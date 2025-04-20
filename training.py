import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def train_model():
    """Train and save the matching model"""
    # Get paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    training_path = os.path.join(current_dir, "training_data.csv")
    model_path = os.path.join(current_dir, "skill_matching_model.pkl")
    vectorizer_path = os.path.join(current_dir, "vectorizer.pkl")
    
    # Load and prepare data
    training_data = pd.read_csv(training_path)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(training_data['skills'])
    y = training_data['match_score']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()