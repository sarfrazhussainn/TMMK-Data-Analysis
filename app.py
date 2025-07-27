import fitz  # PyMuPDF
import re
import csv
import streamlit as st
import tempfile
import os
import pandas as pd
import time
import google.generativeai as genai
from google.genai import types
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import hashlib
from pathlib import Path

# ---- Configuration ----
st.set_page_config(page_title="PDF Name & Muslim Name Analyzer", layout="wide")

# ---- Helper Functions for File Management ----
def get_file_hash(file_content):
    """Generate MD5 hash of file content to detect duplicates"""
    return hashlib.md5(file_content).hexdigest()

def create_or_select_folder():
    """Create folder selection interface"""
    st.subheader("📁 Select or Create Working Directory")
    
    # Try Downloads directory first, fallback to current working directory
    try:
        downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
        if os.path.exists(downloads_path):
            base_path = os.path.join(downloads_path, "tmmkk")
        else:
            # Fallback to current working directory where Python file exists
            base_path = os.path.join(os.getcwd(), "tmmkk")
    except Exception:
        # Ultimate fallback to current working directory
        base_path = os.path.join(os.getcwd(), "tmmkk")
    
    # Create tmmkk directory if it doesn't exist
    try:
        os.makedirs(base_path, exist_ok=True)
        st.info(f"📂 Working with base directory: {base_path}")
    except Exception as e:
        st.error(f"❌ Error creating base directory: {e}")
        return None
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        folder_option = st.radio(
            "Choose option:",
            ["Select existing folder", "Create new folder"],
            key="folder_option"
        )
    
    if folder_option == "Create new folder":
        with col2:
            new_folder_name = st.text_input(
                "Folder name:", 
                value="new_analysis",
                help="Enter name for new working directory inside tmmkk folder"
            )
        
        if new_folder_name:
            folder_path = os.path.join(base_path, new_folder_name)
            
            try:
                os.makedirs(folder_path, exist_ok=True)
                st.success(f"✅ Created and using folder: {folder_path}")
                return folder_path
            except Exception as e:
                st.error(f"❌ Error creating folder: {e}")
                return None
    
    else:
        # Get list of existing folders in tmmkk directory
        try:
            existing_folders = []
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                if os.path.isdir(item_path):
                    existing_folders.append(item)
            
            if existing_folders:
                with col2:
                    selected_folder = st.selectbox(
                        "Select folder:",
                        options=existing_folders,
                        help="Choose from existing folders in tmmkk directory"
                    )
                
                if selected_folder:
                    folder_path = os.path.join(base_path, selected_folder)
                    st.success(f"✅ Using existing folder: {folder_path}")
                    return folder_path
            else:
                st.warning("⚠️ No existing folders found in tmmkk directory. Please create a new folder.")
                return None
        
        except Exception as e:
            st.error(f"❌ Error reading tmmkk directory: {e}")
            return None
    
    return None

def get_next_file_number(folder_path, file_type):
    """Get the next sequential number for a file type"""
    if not os.path.exists(folder_path):
        return 1
    
    files = os.listdir(folder_path)
    numbers = []
    
    for file in files:
        if file.startswith(f"{file_type}_") and file.endswith(".csv"):
            try:
                number = int(file.split("_")[2].split(".")[0])
                numbers.append(number)
            except (IndexError, ValueError):
                continue
    
    return max(numbers) + 1 if numbers else 1

def save_analysis_files(folder_path, extracted_data, muslim_names, file_hash):
    """Save analysis files with sequential numbering"""
    try:
        # Get next file numbers
        extracted_num = get_next_file_number(folder_path, "extracted_name")
        muslim_num = get_next_file_number(folder_path, "muslim_name")
        
        # Create file paths
        extracted_file = os.path.join(folder_path, f"extracted_name_{extracted_num}.csv")
        muslim_file = os.path.join(folder_path, f"muslim_name_{muslim_num}.csv")
        consolidated_extracted = os.path.join(folder_path, "consolidated_extracted_names.csv")
        consolidated_muslim = os.path.join(folder_path, "consolidated_muslim_names.csv")
        processed_files = os.path.join(folder_path, "processed_files.json")
        
        # Save individual extracted names
        extracted_df = pd.DataFrame(extracted_data)
        extracted_df['file_hash'] = file_hash
        extracted_df['file_number'] = extracted_num
        extracted_df.to_csv(extracted_file, index=False)
        
        # Save individual Muslim names
        muslim_df = pd.DataFrame({'Name': muslim_names})
        muslim_df['file_hash'] = file_hash
        muslim_df['file_number'] = muslim_num
        muslim_df.to_csv(muslim_file, index=False)
        
        # Update consolidated files
        # Consolidated extracted names
        if os.path.exists(consolidated_extracted):
            existing_extracted = pd.read_csv(consolidated_extracted)
            consolidated_extracted_df = pd.concat([existing_extracted, extracted_df], ignore_index=True)
        else:
            consolidated_extracted_df = extracted_df
        
        consolidated_extracted_df.to_csv(consolidated_extracted, index=False)
        
        # Consolidated Muslim names
        if os.path.exists(consolidated_muslim):
            existing_muslim = pd.read_csv(consolidated_muslim)
            consolidated_muslim_df = pd.concat([existing_muslim, muslim_df], ignore_index=True)
        else:
            consolidated_muslim_df = muslim_df
        
        consolidated_muslim_df.to_csv(consolidated_muslim, index=False)
        
        # Update processed files record
        processed_record = {}
        if os.path.exists(processed_files):
            with open(processed_files, 'r') as f:
                processed_record = json.load(f)
        
        processed_record[file_hash] = {
            'extracted_file': f"extracted_name_{extracted_num}.csv",
            'muslim_file': f"muslim_name_{muslim_num}.csv",
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_names': len(extracted_data),
            'muslim_names': len(muslim_names)
        }
        
        with open(processed_files, 'w') as f:
            json.dump(processed_record, f, indent=2)
        
        return {
            'extracted_file': extracted_file,
            'muslim_file': muslim_file,
            'consolidated_extracted': consolidated_extracted,
            'consolidated_muslim': consolidated_muslim
        }
    
    except Exception as e:
        st.error(f"❌ Error saving files: {e}")
        return None

def check_file_processed(folder_path, file_hash):
    """Check if a file has already been processed"""
    processed_files = os.path.join(folder_path, "processed_files.json")
    
    if not os.path.exists(processed_files):
        return False
    
    try:
        with open(processed_files, 'r') as f:
            processed_record = json.load(f)
        
        return file_hash in processed_record
    except Exception:
        return False

# ---- Helper Functions for PDF Processing ----
def draw_grid_on_page(page, box_rects):
    """Draw grid on PDF page"""
    grid_color = (0, 0, 0)  # Black
    grid_width = 1
    for rect in box_rects:
        x0, y0, x1, y1 = rect
        page.draw_line((x0, y0), (x0, y1), color=grid_color, width=grid_width)
        page.draw_line((x1, y0), (x1, y1), color=grid_color, width=grid_width)
        page.draw_line((x0, y0), (x1, y0), color=grid_color, width=grid_width)
        page.draw_line((x0, y1), (x1, y1), color=grid_color, width=grid_width)

def extract_text_from_page(page, box_rects):
    """Extract text from specific rectangles on a page"""
    extracted_texts = []
    for rect in box_rects:
        text = page.get_textbox(rect)
        extracted_texts.append(text.strip())
    return extracted_texts

def extract_data_from_texts(texts):
    """Extract names and guardian names from text using regex"""
    data = []
    name_pattern = re.compile(r'(?<!\S)(?:Nam[e|a|o|¢]|Nmae|Nma|Naem|Name|Nama|Nam)\s*(?:[:|+|¢|*|=]\s*|\s+)([^\n]*?)(?:\n|Father Name|Husband Name|Mother Name)')
    guardian_pattern = re.compile(r'(Father Name|Husband Name|Mother Name|Fahter Name|Husbnad Name|Mother Nam|Fathor Namo|Fathor Name|Husband Nama|Father Names)\s*[:|+|*|=]\s*([^\n]*)')
    
    for text in texts:
        names = name_pattern.findall(text)
        guardians = guardian_pattern.findall(text)
        guardian_names = [g[1] for g in guardians]
        min_length = min(len(names), len(guardian_names))
        for i in range(min_length):
            data.append({
                'Name': names[i].strip(),
                'Guardian Name': guardian_names[i].strip()
            })
    return data

def process_pdf(file_path, top_margin, bottom_margin):
    """Process PDF and extract names"""
    pdf_document = fitz.open(file_path)
    all_texts = []
    
    for page_number in range(len(pdf_document)):
        page = pdf_document.load_page(page_number)
        width, height = page.rect.width, page.rect.height
        usable_height = height - top_margin - bottom_margin
        box_width = width / 3
        box_height = usable_height / 10
        
        box_rects = []
        for row in range(10):
            for col in range(3):
                x0 = col * box_width
                y0 = top_margin + row * box_height
                x1 = x0 + box_width
                y1 = y0 + box_height
                box_rects.append((x0, y0, x1, y1))
        
        draw_grid_on_page(page, box_rects)
        texts = extract_text_from_page(page, box_rects)
        all_texts.extend(texts)
    
    pdf_document.close()
    
    data = extract_data_from_texts(all_texts)
    return data

# ---- Helper Functions for Muslim Name Analysis ----
def extract_muslim_names(names):
    """Extract Muslim names from a list of names using Gemini API"""
    prompt = (
        """Here is a list of names:
""" + "\n".join(names) + """

Analyze each name and identify which ones are Muslim names. Consider variations in spelling and transliterations. Respond with only a JSON array of the names that are Muslim names. Example format: ["Ahmed", "Fatima", "Hassan"]"""
    )
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        
        # Clean the response text
        if text.startswith('```json'):
            text = text.replace('```json', '').replace('```', '').strip()
        elif text.startswith('```'):
            text = text.replace('```', '').strip()
        
        # Try to parse as JSON first
        try:
            parsed_data = json.loads(text)
            if isinstance(parsed_data, list):
                return parsed_data
            else:
                return []
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract names from quotes
            matches = re.findall(r'"([^"]+)"', text)
            return matches if matches else []
    except Exception as e:
        st.error(f"Error processing batch: {e}")
        return []

def analyze_names_statistics(all_names, muslim_names):
    """Generate statistics about the name analysis (simplified version)"""
    total_names = len(all_names)
    total_muslim_names = len(muslim_names)
    non_muslim_names = total_names - total_muslim_names
    muslim_percentage = (total_muslim_names / total_names) * 100 if total_names > 0 else 0
    
    return {
        'total_names': total_names,
        'total_muslim_names': total_muslim_names,
        'non_muslim_names': non_muslim_names,
        'muslim_percentage': muslim_percentage
    }

def process_names_for_analysis(names, batch_size=500, pause=1.0):
    """Process names in batches for Muslim name analysis"""
    total_muslim_names = []
    batch_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        batch_count += 1
        
        status_text.text(f"Processing batch {batch_count}/{(len(names)-1)//batch_size + 1}...")
        
        muslim_names_batch = extract_muslim_names(batch)
        total_muslim_names.extend(muslim_names_batch)
        
        progress = (i + batch_size) / len(names)
        progress_bar.progress(min(progress, 1.0))
        
        if i + batch_size < len(names):  # Don't pause after last batch
            time.sleep(pause)
    
    progress_bar.empty()
    status_text.empty()
    
    return total_muslim_names

# ---- Simplified Visualization Functions ----
def create_comparison_chart(stats):
    """Create simple comparison chart"""
    categories = ['Total Names', 'Muslim Names', 'Non-Muslim Names']
    values = [
        stats['total_names'],
        stats['total_muslim_names'],
        stats['non_muslim_names']
    ]
    
    fig = go.Figure(data=[go.Bar(
        x=categories,
        y=values,
        marker_color=['#1f77b4', '#2E8B57', '#FF6B6B'],
        text=values,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Name Analysis Comparison",
        xaxis_title="Categories",
        yaxis_title="Count",
        title_x=0.5,
        font=dict(size=14),
        height=400
    )
    
    return fig

def create_pie_chart(stats):
    """Create pie chart for Muslim vs Non-Muslim names"""
    fig = go.Figure(data=[go.Pie(
        labels=['Muslim Names', 'Non-Muslim Names'],
        values=[stats['total_muslim_names'], stats['non_muslim_names']],
        hole=0.3,
        marker_colors=['#2E8B57', '#FF6B6B'],
        textinfo='label+percent+value',
        textfont_size=12
    )])
    
    fig.update_layout(
        title="Distribution of Muslim vs Non-Muslim Names",
        title_x=0.5,
        font=dict(size=14),
        height=400
    )
    
    return fig

def create_percentage_gauge(stats):
    """Create a gauge chart showing Muslim percentage"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = stats['muslim_percentage'],
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Muslim Names Percentage"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#2E8B57"},
            'steps': [
                {'range': [0, 25], 'color': "lightgray"},
                {'range': [25, 50], 'color': "gray"},
                {'range': [50, 75], 'color': "lightgreen"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font=dict(size=14)
    )
    
    return fig

# ---- Main Streamlit App ----
def main():
    st.title("🔍 PDF Name & Guardian Extractor with Muslim Name Analysis")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("Google Gemini API Key", type="password", 
                                   help="Enter your Google Gemini API key for Muslim name analysis")
    
    # PDF processing parameters
    st.sidebar.subheader("PDF Processing Parameters")
    top_margin = st.sidebar.number_input("Top Margin", value=29, min_value=0, max_value=100)
    bottom_margin = st.sidebar.number_input("Bottom Margin", value=25, min_value=0, max_value=100)
    
    # Analysis parameters
    st.sidebar.subheader("Analysis Parameters")
    batch_size = st.sidebar.slider("Batch Size", min_value=100, max_value=1000, value=500, step=100)
    pause_time = st.sidebar.slider("Pause Between Batches (seconds)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 Folder & PDF Processing", "📊 Muslim Name Analysis", "📈 Analysis Results"])
    
    with tab1:
        st.header("Folder Selection & PDF Processing")
        
        # Folder selection
        working_folder = create_or_select_folder()
        
        if working_folder:
            st.session_state.working_folder = working_folder
            
            # Show processed files count if any exist
            processed_files_path = os.path.join(working_folder, "processed_files.json")
            if os.path.exists(processed_files_path):
                try:
                    with open(processed_files_path, 'r') as f:
                        processed_record = json.load(f)
                    st.info(f"📂 {len(processed_record)} PDF(s) have been processed in this folder")
                except Exception:
                    pass
            
            st.markdown("---")
            
            # PDF upload
            st.subheader("📄 PDF Upload & Processing")
            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
            
            if uploaded_file is not None:
                # Check if file was already processed
                file_content = uploaded_file.read()
                file_hash = get_file_hash(file_content)
                
                if check_file_processed(working_folder, file_hash):
                    st.warning("⚠️ This PDF has already been processed! Skipping to avoid duplicate analysis.")
                    
                    # Load existing data for display
                    try:
                        consolidated_extracted = os.path.join(working_folder, "consolidated_extracted_names.csv")
                        if os.path.exists(consolidated_extracted):
                            df = pd.read_csv(consolidated_extracted)
                            current_file_data = df[df['file_hash'] == file_hash]
                            st.subheader("Previously Extracted Data from This File")
                            st.dataframe(current_file_data[['Name', 'Guardian Name']].head(20))
                    except Exception as e:
                        st.error(f"Error loading existing data: {e}")
                else:
                    # Process new file
                    with tempfile.TemporaryDirectory() as tmpdir:
                        input_path = os.path.join(tmpdir, "input.pdf")
                        
                        # Save uploaded file
                        with open(input_path, 'wb') as f:
                            f.write(file_content)
                        
                        with st.spinner("Processing PDF..."):
                            extracted_data = process_pdf(input_path, top_margin, bottom_margin)
                        
                        if extracted_data:
                            st.success(f"✅ PDF processed successfully! Found {len(extracted_data)} name records.")
                            
                            # Display extracted data
                            st.subheader("Extracted Data Preview")
                            df = pd.DataFrame(extracted_data)
                            st.dataframe(df.head(20))
                            
                            # Store in session state for analysis
                            st.session_state.extracted_data = extracted_data
                            st.session_state.current_file_hash = file_hash
                            st.session_state.df = df
                            
                            # Save to files
                            st.info("💾 Data will be saved to files after Muslim name analysis in Tab 2")
                        else:
                            st.error("❌ No data could be extracted from the PDF. Please check the file and processing parameters.")
        else:
            st.warning("⚠️ Please select or create a working folder to continue.")
    
    with tab2:
        st.header("Muslim Name Analysis")
        
        if 'working_folder' not in st.session_state:
            st.warning("⚠️ Please select a working folder first in Tab 1.")
            return
        
        if 'extracted_data' not in st.session_state:
            st.warning("⚠️ Please process a PDF first in Tab 1.")
            return
        
        if not api_key:
            st.warning("⚠️ Please enter your Google Gemini API key in the sidebar.")
            return
        
        # Initialize Gemini client
        try:
            genai.configure(api_key=api_key)
            # Test the API with a simple request
            model = genai.GenerativeModel('gemini-1.5-flash')
            test_response = model.generate_content("Test message")
            st.success("✅ Gemini API connection successful!")
        except Exception as e:
            st.error(f"❌ Error initializing Gemini API: {e}")
            return
        
        df = st.session_state.df
        extracted_data = st.session_state.extracted_data
        file_hash = st.session_state.current_file_hash
        working_folder = st.session_state.working_folder
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Names", len(df))
        with col2:
            st.metric("Total Records", len(df))
        
        if st.button("🔍 Analyze Muslim Names & Save Files", type="primary"):
            names = df['Name'].dropna().astype(str).tolist()
            
            if not names:
                st.error("❌ No valid names found for analysis.")
                return
            
            with st.spinner("Analyzing names with Gemini AI..."):
                try:
                    muslim_names = process_names_for_analysis(names, batch_size, pause_time)
                    
                    # Save all files
                    saved_files = save_analysis_files(working_folder, extracted_data, muslim_names, file_hash)
                    
                    if saved_files:
                        st.success("✅ Analysis completed and files saved successfully!")
                        
                        # Show saved files info
                        st.subheader("📁 Saved Files")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info(f"📄 Individual extracted names: {os.path.basename(saved_files['extracted_file'])}")
                            st.info(f"📄 Individual Muslim names: {os.path.basename(saved_files['muslim_file'])}")
                        
                        with col2:
                            st.info(f"📄 Consolidated extracted names: {os.path.basename(saved_files['consolidated_extracted'])}")
                            st.info(f"📄 Consolidated Muslim names: {os.path.basename(saved_files['consolidated_muslim'])}")
                        
                        # Store results for visualization
                        stats = analyze_names_statistics(names, muslim_names)
                        st.session_state.muslim_names = muslim_names
                        st.session_state.stats = stats
                        
                        # Display basic results
                        st.subheader("📊 Analysis Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Muslim Names", stats['total_muslim_names'])
                        with col2:
                            st.metric("Muslim Percentage", f"{stats['muslim_percentage']:.1f}%")
                        with col3:
                            st.metric("Non-Muslim Names", stats['non_muslim_names'])
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
    
    with tab3:
        st.header("Analysis Results & Visualizations")
        
        if 'stats' not in st.session_state:
            st.warning("⚠️ Please complete the Muslim name analysis first in Tab 2.")
            return
        
        stats = st.session_state.stats
        working_folder = st.session_state.working_folder
        
        # Load consolidated data for complete analysis
        try:
            consolidated_extracted_path = os.path.join(working_folder, "consolidated_extracted_names.csv")
            consolidated_muslim_path = os.path.join(working_folder, "consolidated_muslim_names.csv")
            
            if os.path.exists(consolidated_extracted_path) and os.path.exists(consolidated_muslim_path):
                consolidated_extracted_df = pd.read_csv(consolidated_extracted_path)
                consolidated_muslim_df = pd.read_csv(consolidated_muslim_path)
                
                # Calculate consolidated stats
                total_consolidated_names = len(consolidated_extracted_df)
                total_consolidated_muslim = len(consolidated_muslim_df)
                total_consolidated_non_muslim = total_consolidated_names - total_consolidated_muslim
                consolidated_muslim_percentage = (total_consolidated_muslim / total_consolidated_names) * 100 if total_consolidated_names > 0 else 0
                
                consolidated_stats = {
                    'total_names': total_consolidated_names,
                    'total_muslim_names': total_consolidated_muslim,
                    'non_muslim_names': total_consolidated_non_muslim,
                    'muslim_percentage': consolidated_muslim_percentage
                }
                
                # Display consolidated results
                st.subheader("📊 Consolidated Results (All Processed Files)")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files Processed", len(consolidated_extracted_df['file_number'].unique()))
                with col2:
                    st.metric("Total Names", consolidated_stats['total_names'])
                with col3:
                    st.metric("Total Muslim Names", consolidated_stats['total_muslim_names'])
                with col4:
                    st.metric("Muslim Percentage", f"{consolidated_stats['muslim_percentage']:.1f}%")
                
                # Visualizations
                st.markdown("---")
                
                # Row 1: Current file vs Consolidated comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Current File Analysis")
                    current_comparison_fig = create_comparison_chart(stats)
                    st.plotly_chart(current_comparison_fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Consolidated Analysis")
                    consolidated_comparison_fig = create_comparison_chart(consolidated_stats)
                    st.plotly_chart(consolidated_comparison_fig, use_container_width=True)
                
                # Row 2: Pie charts
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🥧 Current File Distribution")
                    current_pie_fig = create_pie_chart(stats)
                    st.plotly_chart(current_pie_fig, use_container_width=True)
                
                with col2:
                    st.subheader("🥧 Consolidated Distribution")
                    consolidated_pie_fig = create_pie_chart(consolidated_stats)
                    st.plotly_chart(consolidated_pie_fig, use_container_width=True)
                
                # Row 3: Percentage gauges
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Current File Muslim %")
                    current_gauge_fig = create_percentage_gauge(stats)
                    st.plotly_chart(current_gauge_fig, use_container_width=True)
                
                with col2:
                    st.subheader("📊 Consolidated Muslim %")
                    consolidated_gauge_fig = create_percentage_gauge(consolidated_stats)
                    st.plotly_chart(consolidated_gauge_fig, use_container_width=True)
                
                # Summary table
                st.subheader("📋 Summary Report")
                summary_data = {
                    'Metric': [
                        'Total Files Processed',
                        'Current File - Total Names',
                        'Current File - Muslim Names',
                        'Current File - Muslim %',
                        'Consolidated - Total Names',
                        'Consolidated - Muslim Names', 
                        'Consolidated - Muslim %'
                    ],
                    'Value': [
                        len(consolidated_extracted_df['file_number'].unique()),
                        stats['total_names'],
                        stats['total_muslim_names'],
                        f"{stats['muslim_percentage']:.2f}%",
                        consolidated_stats['total_names'],
                        consolidated_stats['total_muslim_names'],
                        f"{consolidated_stats['muslim_percentage']:.2f}%"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Download consolidated data
                st.subheader("📥 Download Options")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    extracted_csv = consolidated_extracted_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download All Extracted Names",
                        extracted_csv,
                        file_name="all_extracted_names.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    muslim_csv = consolidated_muslim_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download All Muslim Names",
                        muslim_csv,
                        file_name="all_muslim_names.csv",
                        mime="text/csv"
                    )
                
                with col3:
                    summary_csv = summary_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Summary Report",
                        summary_csv,
                        file_name="analysis_summary.csv",
                        mime="text/csv"
                    )
            
            else:
                st.warning("⚠️ Consolidated files not found. Please complete the analysis process first.")
        
        except Exception as e:
            st.error(f"❌ Error loading consolidated data: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🔧 **Made with Streamlit** | "
        "💡 **Powered by Google Gemini AI** | "
        "📄 **PDF Processing with PyMuPDF**"
    )

if __name__ == "__main__":
    main()
