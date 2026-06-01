import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from preprocess import TextPreprocessor

def train_models(file_path):
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    preprocessor = TextPreprocessor()
    
    print("Preprocessing text... (this may take a minute)")
    # We clean text using both description and product purchased for context
    df['cleaned_text'] = df.apply(lambda row: preprocessor.clean_text(row['Ticket Description'], row['Product Purchased']), axis=1)
    
    # --- Model for Ticket Type ---
    print("\n--- Training Ticket Type Classifier ---")
    X = df['cleaned_text']
    y_type = df['Ticket Type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_type, test_size=0.2, random_state=42)
    
    type_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    type_pipeline.fit(X_train, y_train)
    y_pred = type_pipeline.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # --- Model for Ticket Priority ---
    print("\n--- Training Ticket Priority Classifier ---")
    y_priority = df['Ticket Priority']
    
    X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_priority, test_size=0.2, random_state=42)
    
    priority_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    priority_pipeline.fit(X_train_p, y_train_p)
    y_pred_p = priority_pipeline.predict(X_test_p)
    
    print("Accuracy:", accuracy_score(y_test_p, y_pred_p))
    print(classification_report(y_test_p, y_pred_p))
    
    # Save models and preprocessor
    print("\nSaving models...")
    joblib.dump(type_pipeline, 'type_model.joblib')
    joblib.dump(priority_pipeline, 'priority_model.joblib')
    print("Done!")

if __name__ == "__main__":
    train_models('customer_support_tickets.csv')
