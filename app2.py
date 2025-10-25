import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import re
import requests
from bs4 import BeautifulSoup
from googlesearch import search
import pandas as pd
import utils as utils

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


def extract_ingredients(text):
    pattern = r'(?i)ingredients?[:\-]?\s*(.+?)(?=(\n[A-Z][^\n]{3,}|$))'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        ingredients_text = match.group(1)   
        raw_ingredients = re.split(r',|\n|;', ingredients_text)
        ingredients = [re.sub(r'\s+', ' ', i).strip() for i in raw_ingredients if i.strip()]
        return "✅ Ingredients extracted successfully.", ingredients
    else:
        return "⚠️ No ingredients found in the image.", []

def search_health_risk_online(ingredient):
    query = f"{ingredient} food additive health risks"
    try:
        for url in search(query, num_results=5):
            try:
                response = requests.get(url, timeout=5)
                
                soup = BeautifulSoup(response.text, 'html.parser')
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text = p.get_text()
                    if ingredient.lower() in text.lower() or "health" in text.lower():
                        return url, text.strip()
                return url, "No specific health risk text found on page."
            except requests.exceptions.RequestException:
                continue
    except Exception as e:
        return None, f"Search failed: {str(e)}"
    return None, "No results found."

def extract_highlighted_strings(text):
    pattern = r'\bE[1][0-9]{2}\b|\bINS[1][0-9]{2}\b|\b[Cc]olou?rs?[^)]*\)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return matches

def run_ocr(uploaded_file):
    img_array = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8) #converting image to numpy array
    image = cv2.imdecode(img_array,cv2.IMREAD_COLOR) #initializing opencv image
    #Display the uploaded image
    st.image(Image.open(uploaded_file), caption="Uploaded Image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale
    extracted_text = pytesseract.image_to_string(gray)
    return extracted_text.lower()

def main():
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"]{
            background-image: linear-gradient(
                rgba(255, 245, 245, 0.70), 
                rgba(243, 149, 149, 0.85)
            ), url("https://images.unsplash.com/photo-1498837167922-ddd27525d352?q=80&w=2940");
            background-size: cover;
            background-attachment: fixed;
        }
        [data-testid="stHeader"]{
            background: linear-gradient(45deg, rgba(2,0,36,1) 0%, rgba(217,226,179,1) 38%, rgba(0,212,255,1) 100%);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Food Label Analysis using Image Processing")
    st.divider()
    heading_html=f"""<div style='max-width: 800px; margin: 20px auto; padding: 15px;
                    background-color: rgba(255, 255, 255, 0.75);
                    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
                    border-radius: 10px;'>
                    <h4>Upload an image and apply custom processing.<h4>
                    """
    st.markdown(heading_html,unsafe_allow_html=True)
    uploaded_file = st.file_uploader(label="",type=["jpg", "jpeg", "png"])
    st.divider()
    if uploaded_file is not None:
        extracted_text = run_ocr(uploaded_file)
        st.divider()
        st.subheader("🧾 Extracted Ingredients")
        
        status, ingredients = extract_ingredients(extracted_text)
        st.markdown(f"<div style='background-color: rgba(255, 255, 255, 0.75); padding: 15px; border-radius: 12px;'><pre>{ingredients}</pre></div>", unsafe_allow_html=True)
        
        st.success(status)

        # Display Extracted colours :
        highlighted_strings = extract_highlighted_strings(extracted_text)
        if highlighted_strings:
            st.subheader("🎨 Extracted Colour Additives")
            list_items = "".join(f"<li style='font-size: 18px; color: #444;'>Colours | {color}</li>" for color in highlighted_strings)

            # Combine full HTML block
            html_block = f"""
            <div style='max-width: 800px; margin: 40px auto; padding: 25px;
                        background-color: rgba(255, 255, 255, 0.75);
                        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
                        border-radius: 16px;'>
            <p style='font-size: 22px; color: #333; margin-bottom: 16px;'>
                Following are the extracted food colour codes:
            </p>
            <ul style='padding-left: 20px; margin: 0;'>
                {list_items}
            </ul>
            </div>
            """

            # Render it
            st.markdown(html_block, unsafe_allow_html=True)


        if ingredients: 
            if st.button("🔍 Analyze Health Risks Online"):
                with st.spinner("Searching online health risks..."):    
                    for ingredient in ingredients:
                        query_term = re.search(r'\b(E?\d{3,4})\b', ingredient)
                        search_term = query_term.group(1) if query_term else ingredient
                        url, info = search_health_risk_online(search_term)
                        st.markdown(f"""
                            <div style='margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-left: 5px solid #ff4b4b;'>
                            <b>{search_term}</b><br>
                            🌐 <a href="{url}" target="_blank">{url}</a><br>
                            {info}
                            </div>
                        """, unsafe_allow_html=True)
            if st.button("🔄  Analyze Health Risks Using Local database"):
                with st.spinner("Searching database..."):
                    
                    print("Select your Excel file with food additive data...")
                    excel_path = "Expanded_Food_Additives_with_Refined_Palmolein.xlsx"
                    if not excel_path:
                        print("❌ No Excel file selected.")
                        return

                    # Load database
                    df = pd.read_excel(excel_path)
                    df['Additive Name'] = df['Additive Name'].astype(str).str.lower().str.strip()
                    df['INS Code (Indian Equivalent)'] = df['INS Code (Indian Equivalent)'].astype(str).str.lower().str.strip()
                    # Extract text from image
                    print("\n🔍 Extracting text from image...")
                    print(extracted_text)

                    # Match additives
                    print("🔎 Matching additives from image to database...\n")
                    results = utils.find_matches(extracted_text, df)

                    # Display results
                    if results:
                        print("✅ Matched Additives and Health Risks:")
                        for r in results:
                            additive = r['Additive'].title()
                            code = r['Code'].upper()
                            risk = r['Health Risks']
                            Adi_value = r['ADI']
                            
                            
                            st.markdown(f"""
                            <div style="background-color: #fff3cd; border-left: 6px solid #ffecb5; padding: 12px 20px; border-radius: 10px; margin-bottom: 15px;">
                                <h4 style="margin-bottom: 5px;">🔬 Additive: {additive} (<code>{code}</code>)</h4>
                                <p style="margin: 0;"><strong>⚠️ Risk:</strong> {risk}</p>
                                <p style="margin: 0;"><strong>💊 ADI Value:</strong>{Adi_value} gm</p>
                            </div>
                            """, unsafe_allow_html=True)

                    else:
                        st.success("No additives or food colors from the database found in the image.")
                    
       
if __name__ == "__main__":
    main()
