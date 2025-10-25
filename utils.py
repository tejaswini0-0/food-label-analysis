import cv2
import pytesseract
import re
import os
import pandas as pd
from fuzzywuzzy import fuzz
from tkinter import Tk, filedialog

# -------- Step 1: Load Excel File --------
def select_excel_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Excel file with additive data", filetypes=[("Excel files", "*.xlsx *.xls")])
    return file_path

# -------- Step 2: Load Image File --------
def select_image_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select image file of ingredients", filetypes=[("Image files", "*.png *.jpg *.jpeg")])
    return file_path

# -------- Step 3: Preprocess Image and Extract Text --------
def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.lower()

# -------- Step 4: Normalize Codes (e.g., "E-102", "ins 102") --------
def extract_codes(text):
    """
    Extracts all additive codes like 330, 150d, 160c from a block of text.
    Returns a set of cleaned codes.
    """
    pattern = r'(?:e|ins)?\s*(\d{3})([a-z]?)'
    matches = re.findall(pattern, text.lower())
    return {num + letter for num, letter in matches}

def normalize_code(text):
    # Extract the first occurrence of a 3-digit number with optional trailing letter
    match = re.search(r'(?:e|ins)?\s*(\d{3})([a-z]?)', text.lower())
    if match:
        return match.group(1) + match.group(2)
    return re.sub(r'[^a-z0-9]', '', text.lower())  # fallback for non-matching cases

# -------- Step 5: Fuzzy Match Additives and Codes --------
def find_matches(text, df, threshold=80):
    matches = []
    normalized_text = extract_codes(text)

    for _, row in df.iterrows():
        name = row['Additive Name'].strip().lower()
        code = normalize_code(str(row['INS Code (Indian Equivalent)']))

        # Check for additive name (direct or fuzzy match)
        if name in text or fuzz.partial_ratio(name, text) >= threshold:
            matches.append({
                "Additive": name,
                "Code": code,
                "Health Risks": row['Linked Health Risks'],
                "ADI": row['ADI']
            })
            continue

        # Check for normalized INS code
        if code in normalized_text or fuzz.partial_ratio(code, normalized_text) >= threshold:
            matches.append({
                "Additive": name,
                "Code": code,
                "Health Risks": row['Linked Health Risks'],
                "ADI": row['ADI']
            })

    return matches

# -------- Main Execution --------
def main():
    print("Select your Excel file with food additive data...")
    excel_path = "C:\\Users\\abhis\\Downloads\\Expanded_Food_Additives_with_Refined_Palmolein.xlsx"
    if not excel_path:
        print("❌ No Excel file selected.")
        return

    print("Select your image file (label, ingredients list)...")
    image_path = "C:\\Users\\abhis\\Downloads\\a.jpeg.jpg"
    if not image_path:
        print("❌ No image file selected.")
        return

    # Load database
    df = pd.read_excel(excel_path)
    df['Additive Name'] = df['Additive Name'].astype(str).str.lower().str.strip()
    df['INS Code (Indian Equivalent)'] = df['INS Code (Indian Equivalent)'].astype(str).str.lower().str.strip()

    # Extract text from image
    print("\n🔍 Extracting text from image...")
    extracted_text = extract_text_from_image(image_path)
    print(extracted_text)

    # Match additives
    print("🔎 Matching additives from image to database...\n")
    results = find_matches(extracted_text, df)

    # Display results
    if results:
        print("✅ Matched Additives and Health Risks:")
        for r in results:
            print(f"\nAdditive: {r['Additive'].title()} ({r['Code'].upper()})")
            print(f"⚠️ Risk: {r['Health Risks']}")
    else:
        print("No additives or food colors from the database found in the image.")

if __name__ == "__main__":
    main()

