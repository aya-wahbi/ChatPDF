import os
from src.ingestion.document_loader import load_document

def main():
    raw_folder = 'data/raw'
    processed_folder = 'data/processed'
    
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    for file in os.listdir(raw_folder):
        file_path = os.path.join(raw_folder, file)
        print(f"Processing {file}...")
        content = load_document(file_path)
        
        if content:
            output_file = os.path.join(processed_folder, f"{os.path.splitext(file)[0]}.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Saved processed text to {output_file}\n")
        else:
            print(f"Failed to process {file}\n")

if __name__ == "__main__":
    main()