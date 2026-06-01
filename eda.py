import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path):
    df = pd.read_csv(file_path)
    
    print("--- Dataset Info ---")
    print(df.info())
    print("\n--- Sample Data ---")
    print(df[['Ticket Type', 'Ticket Priority', 'Ticket Description']].head())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    
    print("\n--- Ticket Type Distribution ---")
    print(df['Ticket Type'].value_counts())
    
    print("\n--- Ticket Priority Distribution ---")
    print(df['Ticket Priority'].value_counts())
    
    # Check for empty descriptions
    empty_desc = df['Ticket Description'].apply(lambda x: len(str(x).strip()) == 0).sum()
    print(f"\nEmpty Descriptions: {empty_desc}")

    # Check for duplicate descriptions
    duplicates = df.duplicated(subset=['Ticket Description']).sum()
    print(f"Duplicate Descriptions: {duplicates}")

if __name__ == "__main__":
    perform_eda('customer_support_tickets.csv')
