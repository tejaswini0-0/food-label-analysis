# app.py — Nutrition Label Insights (Local Processing)

import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import re
import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd

# Initialize the OCR reader (only needs to be done once)
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

def extract_number(text: str) -> float:
    """Extract numeric value from text, handling various formats."""
    try:
        # Remove everything except digits, decimal points, and negative signs
        clean = re.sub(r'[^0-9.-]', '', text)
        return float(clean)
    except:
        return 0.0

def extract_value_with_unit(text: str, patterns: Dict[str, str]) -> Tuple[float, str]:
    """Extract value and unit from text using patterns."""
    for unit, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = extract_number(match.group(1))
            return value, unit
    return 0.0, ''

def parse_nutrition_text(text_list: List[str]) -> Dict[str, Any]:
    """Parse OCR text into structured nutrition data."""
    result = {
        'product_name': None,
        'serving_size': None,
        'total_weight_g': None,
        'calories': None,
        'protein': None,
        'total_fat': None,
        'saturated_fat': None,
        'carbohydrates': None,
        'sugar': None,
        'fiber': None,
        'sodium': None,
        'unit_type': {},
        'daily_values': {}
    }
    
    # Join all text into one string for easier searching
    full_text = ' '.join(text_list).lower()
    
    # Try to find total package weight
    weight_match = re.search(r'net\s*wt\.?\s*(\d+[\d.]*)\s*g', full_text)
    if weight_match:
        result['total_weight_g'] = extract_number(weight_match.group(1))
    
    # Pattern dictionary with unit patterns
    nutrient_patterns = {
        'serving_size': {
            'g': r'serving size[:\s]+(\d+[\d.]*)\s*g',
            'unit': r'serving size[:\s]+(\d+[\d.]*)\s*([^\d\n]+)'
        },
        'calories': {
            'cal': r'calories[:\s]+(\d+)'
        },
        'protein': {
            'g': r'protein[:\s]+(\d+[\d.]*)\s*g',
            '%': r'protein[:\s]+(\d+[\d.]*)\s*%'
        },
        'total_fat': {
            'g': r'total fat[:\s]+(\d+[\d.]*)\s*g',
            '%': r'total fat[:\s]+(\d+[\d.]*)\s*%'
        },
        'saturated_fat': {
            'g': r'saturated fat[:\s]+(\d+[\d.]*)\s*g',
            '%': r'saturated fat[:\s]+(\d+[\d.]*)\s*%'
        },
        'carbohydrates': {
            'g': r'(total )?carbohydrate[s]?[:\s]+(\d+[\d.]*)\s*g',
            '%': r'(total )?carbohydrate[s]?[:\s]+(\d+[\d.]*)\s*%'
        },
        'sugar': {
            'g': r'sugars?[:\s]+(\d+[\d.]*)\s*g',
            '%': r'sugars?[:\s]+(\d+[\d.]*)\s*%'
        },
        'fiber': {
            'g': r'fiber[:\s]+(\d+[\d.]*)\s*g',
            '%': r'fiber[:\s]+(\d+[\d.]*)\s*%'
        },
        'sodium': {
            'mg': r'sodium[:\s]+(\d+[\d.]*)\s*mg',
            '%': r'sodium[:\s]+(\d+[\d.]*)\s*%'
        }
    }
    
    # Process each nutrient
    for nutrient, unit_patterns in nutrient_patterns.items():
        for unit, pattern in unit_patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                # Extract value (some patterns have multiple groups, take last number group)
                groups = match.groups()
                value = next((g for g in reversed(groups) if g and any(c.isdigit() for c in g)), None)
                if value:
                    result[nutrient] = extract_number(value)
                    result['unit_type'][nutrient] = unit
                break  # Stop after first match for this nutrient
    
    # Try to find product name (usually at the top, capitalized)
    # Take the first few words before "nutrition facts" if present
    name_match = re.search(r'^(.+?)(?:nutrition facts|serving size)', full_text, re.I)
    if name_match:
        result['product_name'] = name_match.group(1).strip().title()
    
    return result

def calculate_health_score(data: Dict[str, Any]) -> Tuple[int, List[str], Dict[str, Any]]:
    """Calculate health score and generate insights."""
    score = 100
    insights = []
    score_breakdown = {
        'base_score': 100,
        'adjustments': []
    }
    
    def add_score_adjustment(points: int, reason: str, insight: str = None):
        score_breakdown['adjustments'].append({
            'points': points,
            'reason': reason
        })
        if insight:
            insights.append(insight)
    
    # Sugar analysis (WHO recommends less than 25g/day)
    sugar_val = data.get('sugar')
    if sugar_val:
        if data['unit_type'].get('sugar') == '%':
            if sugar_val > 25:
                add_score_adjustment(-20, "High sugar content (>25% DV)", f"High sugar content ({sugar_val}% DV)")
            elif sugar_val > 15:
                add_score_adjustment(-10, "Moderate sugar content (>15% DV)", f"Moderate sugar content ({sugar_val}% DV)")
        else:  # Assuming grams
            if sugar_val > 25:
                add_score_adjustment(-20, "High sugar content (>25g)", f"High sugar content ({sugar_val}g)")
            elif sugar_val > 15:
                add_score_adjustment(-10, "Moderate sugar content (>15g)", f"Moderate sugar content ({sugar_val}g)")
    
    # Sodium analysis (WHO recommends less than 2000mg/day)
    sodium_val = data.get('sodium')
    if sodium_val:
        if data['unit_type'].get('sodium') == '%':
            if sodium_val > 50:  # More than 50% DV
                add_score_adjustment(-20, "High sodium content (>50% DV)", f"High sodium content ({sodium_val}% DV)")
            elif sodium_val > 25:
                add_score_adjustment(-10, "Moderate sodium content (>25% DV)", f"Moderate sodium content ({sodium_val}% DV)")
        else:  # Assuming mg
            if sodium_val > 2000:
                add_score_adjustment(-20, "High sodium content (>2000mg)", f"High sodium content ({sodium_val}mg)")
            elif sodium_val > 1000:
                add_score_adjustment(-10, "Moderate sodium content (>1000mg)", f"Moderate sodium content ({sodium_val}mg)")
    
    # Protein analysis
    protein_val = data.get('protein')
    if protein_val:
        if data['unit_type'].get('protein') == '%':
            if protein_val > 20:
                add_score_adjustment(10, "Good protein content (>20% DV)", f"Good source of protein ({protein_val}% DV)")
            elif protein_val < 5:
                add_score_adjustment(-5, "Low protein content (<5% DV)", "Low in protein")
        else:  # Assuming grams
            if protein_val > 20:
                add_score_adjustment(10, "Good protein content (>20g)", f"Good source of protein ({protein_val}g)")
            elif protein_val < 5:
                add_score_adjustment(-5, "Low protein content (<5g)", "Low in protein")
    
    # Fiber analysis
    fiber_val = data.get('fiber')
    if fiber_val:
        if data['unit_type'].get('fiber') == '%':
            if fiber_val > 20:
                add_score_adjustment(10, "Good fiber content (>20% DV)", f"Good source of fiber ({fiber_val}% DV)")
        else:  # Assuming grams
            if fiber_val > 5:
                add_score_adjustment(10, "Good fiber content (>5g)", f"Good source of fiber ({fiber_val}g)")
    
    # Calculate final score
    final_score = score + sum(adj['points'] for adj in score_breakdown['adjustments'])
    final_score = max(0, min(100, final_score))
    
    return final_score, insights, score_breakdown

# --- Helpers ---
def main():
    # Page config
    st.set_page_config(
        page_title="Nutrition Insights",
        page_icon="🥗",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
            .main {
                padding: 2rem;
            }
            .stImage {
                max-width: 400px !important;
                margin: 0 auto;
            }
            .nutrition-stats {
                padding: 1.5rem;
                background-color: #f0f2f6;
                border-radius: 10px;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .stProgress > div > div > div > div {
                background-color: #2ecc71;
            }
            .health-score-container {
                padding: 1.5rem;
                border-radius: 8px;
                background: linear-gradient(135deg, #2ecc71, #27ae60);
                color: white;
                text-align: center;
                margin: 1rem 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .health-score-explanation {
                font-size: 0.9rem;
                color: #666;
                padding: 1.5rem;
                background-color: #fff;
                border-radius: 8px;
                border: 1px solid #eee;
                margin-top: 1rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            .insights-card {
                background-color: #fff;
                padding: 1.5rem;
                border-radius: 8px;
                border: 1px solid #eee;
                margin: 1rem 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }
            h1 {
                color: #2c3e50;
                margin-bottom: 2rem;
                text-align: center;
            }
            h2, h3, h4 {
                color: #34495e;
                margin: 1.5rem 0 1rem 0;
            }
            .stAlert {
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Custom header with better contrast
    st.markdown("""
        <h1 style='color: white; text-align: center; margin-bottom: 1rem;'>
            🥗 Nutrition Label Insights
        </h1>
    """, unsafe_allow_html=True)

    # Add app description with improved styling
    st.markdown("""
    <div class="insights-card">
        <p style='color: #2c3e50; font-size: 1.1rem; line-height: 1.5;'>
            Upload a nutrition label image to get detailed insights about the nutritional content, 
            health score, and macronutrient breakdown. Our advanced analysis will help you make 
            informed decisions about your food choices.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Two-column layout for upload and display
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.subheader("📤 Upload Label")
        uploaded_file = st.file_uploader(
            "Choose a nutrition label image",
            type=["png", "jpg", "jpeg"],
            key="nutrition_label_uploader"
        )

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            
            # Resize image while maintaining aspect ratio
            max_width = 400
            ratio = max_width / float(img.size[0])
            new_size = (max_width, int(float(img.size[1]) * ratio))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            
            st.image(
                img_resized, 
                caption="Uploaded nutrition label", 
                use_container_width=True
            )
            analyze_button = st.button(
                "🔍 Analyze Nutrition",
                key="analyze_button",
                use_container_width=True
            )

            if analyze_button:
                try:
                    # Convert PIL Image to numpy array for OCR
                    img_array = np.array(img)
                    
                    with st.spinner("Analyzing nutrition label..."):
                        # Get OCR reader
                        reader = load_ocr()
                        
                        # Extract text from image
                        results = reader.readtext(img_array, detail=0)
                        
                        # Parse the extracted text
                        nutrition_data = parse_nutrition_text(results)
                        
                        # Calculate health score and insights
                        health_score, insights, score_breakdown = calculate_health_score(nutrition_data)
                        
                        with col2:
                            st.success("✅ Analysis Complete")
                            
                            # Display the results using the existing display code...
                            process_and_display_results(
                                nutrition_data,
                                health_score,
                                insights,
                                score_breakdown
                            )
                            
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")
                    
        else:
            analyze_button = False
            with col2:
                st.info("📸 Upload a nutrition label image to begin analysis")

def process_and_display_results(nutrition_data, health_score, insights, score_breakdown):
    """Process and display the analysis results."""
    # Display health score with explanation
    st.markdown("## 🏆 Health Score Analysis")
    
    # Display score in custom container
    st.markdown(f"""
    <div class="health-score-container">
        <h2 style="color: white; margin: 0;">Health Score: {health_score}/100</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show score explanation
    st.markdown("""
    <div class="health-score-explanation">
        <h4>How is this score calculated?</h4>
        <p>The health score starts at 100 and is adjusted based on:</p>
        <ul>
            <li>Sugar content (WHO recommendation: <25g/day)</li>
            <li>Sodium levels (WHO recommendation: <2000mg/day)</li>
            <li>Protein content (bonus for >20g)</li>
            <li>Fiber content (bonus for >5g)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Show score breakdown
    if score_breakdown['adjustments']:
        st.markdown("### Score Breakdown")
        for adj in score_breakdown['adjustments']:
            if adj['points'] > 0:
                st.success(f"+{adj['points']} points: {adj['reason']}")
            else:
                st.warning(f"{adj['points']} points: {adj['reason']}")
    
    # Calculate and display macronutrient distribution
    total_calories = 0
    if nutrition_data.get('protein'):
        total_calories += nutrition_data['protein'] * 4
    if nutrition_data.get('carbohydrates'):
        total_calories += nutrition_data['carbohydrates'] * 4
    if nutrition_data.get('total_fat'):
        total_calories += nutrition_data['total_fat'] * 9
    
    if total_calories > 0:
        st.markdown("### 📊 Macronutrient Distribution")
        cols = st.columns(3)
        
        with cols[0]:
            protein_pct = int((nutrition_data.get('protein', 0) * 4 / total_calories) * 100)
            st.metric("Protein", f"{protein_pct}%")
        with cols[1]:
            carbs_pct = int((nutrition_data.get('carbohydrates', 0) * 4 / total_calories) * 100)
            st.metric("Carbs", f"{carbs_pct}%")
        with cols[2]:
            fat_pct = int((nutrition_data.get('total_fat', 0) * 9 / total_calories) * 100)
            st.metric("Fats", f"{fat_pct}%")
    
    # Display detailed nutrition data in expander
    with st.expander("📋 Detailed Nutrition Data"):
        # Convert the nutrition data to a prettier format
        display_data = []
        for nutrient, value in nutrition_data.items():
            if nutrient not in ['unit_type', 'daily_values', 'vitamins'] and value is not None:
                unit = nutrition_data['unit_type'].get(nutrient, '')
                if unit:
                    display_data.append({
                        'Nutrient': nutrient.replace('_', ' ').title(),
                        'Value': f"{value}{unit}"
                    })
        
        if display_data:
            df = pd.DataFrame(display_data)
            st.dataframe(df, hide_index=True, use_container_width=True)

if __name__ == "__main__":
    main()

st.caption("Upload nutrition labels to see structured data + health insights. Powered by EasyOCR (offline processing).")
