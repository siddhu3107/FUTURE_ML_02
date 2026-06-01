import joblib
import sys
from preprocess import TextPreprocessor

class TicketClassifier:
    def __init__(self, type_model_path='type_model.joblib', priority_model_path='priority_model.joblib'):
        try:
            self.type_model = joblib.load(type_model_path)
            self.priority_model = joblib.load(priority_model_path)
            self.preprocessor = TextPreprocessor()
            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)

    def classify(self, text, product_name=None):
        cleaned_text = [self.preprocessor.clean_text(text, product_name)]
        
        ticket_type = self.type_model.predict(cleaned_text)[0]
        priority = self.priority_model.predict(cleaned_text)[0]
        
        return ticket_type, priority

if __name__ == "__main__":
    classifier = TicketClassifier()
    
    print("\n--- Support Ticket Classifier ---")
    while True:
        ticket_text = input("\nEnter ticket description (or 'quit' to exit): ")
        if ticket_text.lower() == 'quit':
            break
        
        product = input("Enter product name (optional, press Enter to skip): ")
        if not product.strip():
            product = None
            
        t_type, priority = classifier.classify(ticket_text, product)
        
        print(f"\nResult:")
        print(f"  Classification: {t_type}")
        print(f"  Priority Level: {priority}")
