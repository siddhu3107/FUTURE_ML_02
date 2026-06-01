import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from preprocess import TextPreprocessor

def evaluate_and_plot():
    df = pd.read_csv('customer_support_tickets.csv')
    preprocessor = TextPreprocessor()
    
    # Load models
    type_model = joblib.load('type_model.joblib')
    priority_model = joblib.load('priority_model.joblib')
    
    # Preprocess a subset for evaluation if needed, or just use the whole set
    # Using a sample of 500 for quick evaluation/viz
    test_df = df.sample(500, random_state=42)
    test_df['cleaned_text'] = test_df.apply(lambda row: preprocessor.clean_text(row['Ticket Description'], row['Product Purchased']), axis=1)
    
    # Evaluate Type
    y_type_true = test_df['Ticket Type']
    y_type_pred = type_model.predict(test_df['cleaned_text'])
    
    # Plot Confusion Matrix for Type
    cm_type = confusion_matrix(y_type_true, y_type_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Blues', 
                xticklabels=type_model.classes_, yticklabels=type_model.classes_)
    plt.title('Confusion Matrix: Ticket Type')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('type_confusion_matrix.png')
    plt.close()
    
    # Evaluate Priority
    y_priority_true = test_df['Ticket Priority']
    y_priority_pred = priority_model.predict(test_df['cleaned_text'])
    
    # Plot Confusion Matrix for Priority
    cm_priority = confusion_matrix(y_priority_true, y_priority_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_priority, annot=True, fmt='d', cmap='Greens', 
                xticklabels=priority_model.classes_, yticklabels=priority_model.classes_)
    plt.title('Confusion Matrix: Ticket Priority')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('priority_confusion_matrix.png')
    plt.close()
    
    print("Evaluation scripts generated plots.")

if __name__ == "__main__":
    evaluate_and_plot()
