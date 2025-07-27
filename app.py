import fitz  # PyMuPDF
import re
import csv
import streamlit as st
import tempfile
import os
import pandas as pd
import time
import google.generativeai as genai
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
import zipfile
from io import BytesIO
import base64

# ---- Configuration ----
st.set_page_config(page_title="PDF Name & Muslim Name Analyzer", layout="wide")

# ---- Helper Functions for File Management ----
def get_file_hash(file_content):
    """Generate MD5 hash of file content to detect duplicates"""
    return hashlib.md5(file_content).hexdigest()

def create_session_folder():
    """Create a session-based folder structure in memory/temp"""
    # Use session state to maintain folder structure
    if 'session_folder' not in st.session_state:
        # Create a unique session identifier
        import uuid
        session_id = str(uuid.uuid4())[:8]
        st.session_state.session_id = session_id
        st.session_state.session_folder = f"tmmkk_session_{session_id}"
        st.session_state.processed_files = {}
        st.session_state.all_extracted_data = []
        st.session_state.all_muslim_names = []
    
    return st.session_state.session_folder

def save_to_session_storage(extracted_data, muslim_names, file_hash, filename):
    """Save analysis data to session state instead of files"""
    try:
        # Add to session storage
        file_record = {
            'file_hash': file_hash,
            'filename': filename,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'extracted_data': extracted_data,
            'muslim_names': muslim_names,
            'total_names': len(extracted_data),
            'muslim_count': len(muslim_names)
        }
        
        # Store in session state
        st.session_state.processed_files[file_hash] = file_record
        
        # Update consolidated data
        st.session_state.all_extracted_data.extend(extracted_data)
        st.session_state.all_muslim_names.extend(muslim_names)
        
        return True
    
    except Exception as e:
        st.error(f"Error saving to session: {e}")
        return False

def create_downloadable_files():
    """Create downloadable files from session data"""
    try:
        # Create DataFrames
        all_extracted_df = pd.DataFrame(st.session_state.all_extracted_data)
        all_muslim_df = pd.DataFrame({'Name': st.session_state.all_muslim_names})
        
        # Add file information
        file_info = []
        for file_hash, record in st.session_state.processed_files.items():
            for i in range(record['total_names']):
                file_info.append({
                    'filename': record['filename'],
                    'file_hash': file_hash,
                    'timestamp': record['timestamp']
                })
        
        file_info_df = pd.DataFrame(file_info)
        if len(file_info_df) > 0 and len(all_extracted_df) > 0:
            all_extracted_df = pd.concat([all_extracted_df, file_info_df], axis=1)
        
        # Add file information to Muslim names
        muslim_file_info = []
        for file_hash, record in st.session_state.processed_files.items():
            for name in record['muslim_names']:
                muslim_file_info.append({
                    'filename': record['filename'],
                    'file_hash': file_hash,
                    'timestamp': record['timestamp']
                })
        
        muslim_file_info_df = pd.DataFrame(muslim_file_info)
        if len(muslim_file_info_df) > 0 and len(all_muslim_df) > 0:
            all_muslim_df = pd.concat([all_muslim_df, muslim_file_info_df], axis=1)
        
        return all_extracted_df, all_muslim_df
    
    except Exception as e:
        st.error(f"Error creating downloadable files: {e}")
        return pd.DataFrame(), pd.DataFrame()

def create_zip_download():
    """Create a ZIP file containing all analysis results"""
    try:
        # Create a BytesIO buffer for the ZIP file
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add individual file results
            for file_hash, record in st.session_state.processed_files.items():
                # Individual extracted names file
                extracted_df = pd.DataFrame(record['extracted_data'])
                extracted_csv = extracted_df.to_csv(index=False)
                zip_file.writestr(f"extracted_names_{record['filename']}.csv", extracted_csv)
                
                # Individual Muslim names file
                muslim_df = pd.DataFrame({'Name': record['muslim_names']})
                muslim_csv = muslim_df.to_csv(index=False)
                zip_file.writestr(f"muslim_names_{record['filename']}.csv", muslim_csv)
            
            # Add consolidated files
            all_extracted_df, all_muslim_df = create_downloadable_files()
            
            if not all_extracted_df.empty:
                consolidated_extracted_csv = all_extracted_df.to_csv(index=False)
                zip_file.writestr("consolidated_extracted_names.csv", consolidated_extracted_csv)
            
            if not all_muslim_df.empty:
                consolidated_muslim_csv = all_muslim_df.to_csv(index=False)
                zip_file.writestr("consolidated_muslim_names.csv", consolidated_muslim_csv)
            
            # Add summary report
            summary_data = create_summary_report()
            summary_csv = summary_data.to_csv(index=False)
            zip_file.writestr("analysis_summary.csv", summary_csv)
            
            # Add processing log
            processing_log = create_processing_log()
            zip_file.writestr("processing_log.json", json.dumps(processing_log, indent=2))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    except Exception as e:
        st.error(f"Error creating ZIP file: {e}")
        return None

def create_summary_report():
    """Create a summary report of all processed files"""
    try:
        summary_data = []
        total_names = len(st.session_state.all_extracted_data)
        total_muslim = len(st.session_state.all_muslim_names)
        
        # Overall summary
        summary_data.append({
            'File': 'OVERALL SUMMARY',
            'Total Names': total_names,
            'Muslim Names': total_muslim,
            'Non-Muslim Names': total_names - total_muslim,
            'Muslim Percentage': f"{(total_muslim / total_names * 100):.2f}%" if total_names > 0 else "0%",
            'Processing Date': time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # Individual file summaries
        for file_hash, record in st.session_state.processed_files.items():
            summary_data.append({
                'File': record['filename'],
                'Total Names': record['total_names'],
                'Muslim Names': record['muslim_count'],
                'Non-Muslim Names': record['total_names'] - record['muslim_count'],
                'Muslim Percentage': f"{(record['muslim_count'] / record['total_names'] * 100):.2f}%" if record['total_names'] > 0 else "0%",
                'Processing Date': record['timestamp']
            })
        
        return pd.DataFrame(summary_data)
    
    except Exception as e:
        st.error(f"Error creating summary report: {e}")
        return pd.DataFrame()

def create_processing_log():
    """Create a processing log with detailed information"""
    try:
        log_data = {
            'session_id': st.session_state.session_id,
            'total_files_processed': len(st.session_state.processed_files),
            'total_names_extracted': len(st.session_state.all_extracted_data),
            'total_muslim_names_found': len(st.session_state.all_muslim_names),
            'processing_details': []
        }
        
        for file_hash, record in st.session_state.processed_files.items():
            log_data['processing_details'].append({
                'filename': record['filename'],
                'file_hash': file_hash,
                'timestamp': record['timestamp'],
                'names_extracted': record['total_names'],
                'muslim_names_found': record['muslim_count'],
                'muslim_percentage': f"{(record['muslim_count'] / record['total_names'] * 100):.2f}%" if record['total_names'] > 0 else "0%"
            })
        
        return log_data
    
    except Exception as e:
        st.error(f"Error creating processing log: {e}")
        return {}

def check_file_processed(file_hash):
    """Check if a file has already been processed in the current session"""
    return file_hash in st.session_state.get('processed_files', {})

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
    """Generate statistics about the name analysis"""
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

# ---- Visualization Functions ----
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
    st.title("TMMK Data Analysis")
    st.markdown("---")
    
    # Initialize session storage
    session_folder = create_session_folder()
    
    # Display session information
    st.sidebar.header("📊 Session Information")
    st.sidebar.info(f"Session ID: {st.session_state.session_id}")
    if 'processed_files' in st.session_state:
        st.sidebar.metric("Files Processed", len(st.session_state.processed_files))
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input("Enter API Key", type="password", 
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
    tab1, tab2, tab3, tab4 = st.tabs(["📄 PDF Processing", "🔍 Muslim Name Analysis", "📊 Analysis Results", "📥 Download Center"])
    
    with tab1:
        st.header("PDF Processing")
        
        # Show session status
        if 'processed_files' in st.session_state and len(st.session_state.processed_files) > 0:
            st.success(f"✅ Session active with {len(st.session_state.processed_files)} processed file(s)")
            
            # Show processed files
            st.subheader("📋 Processed Files in Current Session")
            for file_hash, record in st.session_state.processed_files.items():
                with st.expander(f"📄 {record['filename']} - {record['timestamp']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Names", record['total_names'])
                    with col2:
                        st.metric("Muslim Names", record['muslim_count'])
                    with col3:
                        percentage = (record['muslim_count'] / record['total_names'] * 100) if record['total_names'] > 0 else 0
                        st.metric("Muslim %", f"{percentage:.1f}%")
        else:
            st.info("🚀 Ready to process your first PDF file!")
        
        st.markdown("---")
        
        # PDF upload
        st.subheader("📄 Upload PDF")
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        
        if uploaded_file is not None:
            # Check if file was already processed
            file_content = uploaded_file.read()
            file_hash = get_file_hash(file_content)
            filename = uploaded_file.name
            
            if check_file_processed(file_hash):
                st.warning("⚠️ This PDF has already been processed in this session!")
                
                # Show existing data
                record = st.session_state.processed_files[file_hash]
                st.subheader("Previously Extracted Data")
                df = pd.DataFrame(record['extracted_data'])
                st.dataframe(df.head(20))
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
                        st.session_state.current_extracted_data = extracted_data
                        st.session_state.current_file_hash = file_hash
                        st.session_state.current_filename = filename
                        st.session_state.current_df = df
                        
                        st.info("✨ Ready for Muslim name analysis! Go to the next tab.")
                    else:
                        st.error("❌ No data could be extracted from the PDF. Please check the file and processing parameters.")
    
    with tab2:
        st.header("Muslim Name Analysis")
        
        if 'current_extracted_data' not in st.session_state:
            st.warning("⚠️ Please process a PDF first in the PDF Processing tab.")
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
        
        df = st.session_state.current_df
        extracted_data = st.session_state.current_extracted_data
        file_hash = st.session_state.current_file_hash
        filename = st.session_state.current_filename
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Names", len(df))
        with col2:
            st.metric("Current File", filename)
        
        if st.button("🔍 Analyze Muslim Names", type="primary"):
            names = df['Name'].dropna().astype(str).tolist()
            
            if not names:
                st.error("❌ No valid names found for analysis.")
                return
            
            with st.spinner("Analyzing names with Gemini AI..."):
                try:
                    muslim_names = process_names_for_analysis(names, batch_size, pause_time)
                    
                    # Save to session storage
                    success = save_to_session_storage(extracted_data, muslim_names, file_hash, filename)
                    
                    if success:
                        st.success("✅ Analysis completed and saved to session!")
                        
                        # Display basic results
                        stats = analyze_names_statistics(names, muslim_names)
                        
                        st.subheader("📊 Analysis Results")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Muslim Names", stats['total_muslim_names'])
                        with col2:
                            st.metric("Muslim Percentage", f"{stats['muslim_percentage']:.1f}%")
                        with col3:
                            st.metric("Non-Muslim Names", stats['non_muslim_names'])
                        
                        # Clear current file data
                        for key in ['current_extracted_data', 'current_file_hash', 'current_filename', 'current_df']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        st.info("🎉 Analysis complete! Check the Analysis Results tab for visualizations and the Download Center for files.")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
    
    with tab3:
        st.header("Analysis Results & Visualizations")
        
        if 'processed_files' not in st.session_state or len(st.session_state.processed_files) == 0:
            st.warning("⚠️ No analysis results available. Please process and analyze a PDF first.")
            return
        
        # Calculate consolidated statistics
        total_names = len(st.session_state.all_extracted_data)
        total_muslim = len(st.session_state.all_muslim_names)
        total_non_muslim = total_names - total_muslim
        muslim_percentage = (total_muslim / total_names * 100) if total_names > 0 else 0
        
        consolidated_stats = {
            'total_names': total_names,
            'total_muslim_names': total_muslim,
            'non_muslim_names': total_non_muslim,
            'muslim_percentage': muslim_percentage
        }
        
        # Display consolidated results
        st.subheader("📊 Consolidated Results (All Processed Files)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Files Processed", len(st.session_state.processed_files))
        with col2:
            st.metric("Total Names", consolidated_stats['total_names'])
        with col3:
            st.metric("Total Muslim Names", consolidated_stats['total_muslim_names'])
        with col4:
            st.metric("Muslim Percentage", f"{consolidated_stats['muslim_percentage']:.1f}%")
        
        # Visualizations
        st.markdown("---")
        
        # Row 1: Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Overall Analysis")
            comparison_fig = create_comparison_chart(consolidated_stats)
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        with col2:
            st.subheader("🥧 Distribution")
            pie_fig = create_pie_chart(consolidated_stats)
            st.plotly_chart(pie_fig, use_container_width=True)
        
        # Row 2: Gauge and file breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Muslim Percentage")
            gauge_fig = create_percentage_gauge(consolidated_stats)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            st.subheader("📋 File Breakdown")
            breakdown_data = []
            for file_hash, record in st.session_state.processed_files.items():
                breakdown_data.append({
                    'File': record['filename'][:30] + '...' if len(record['filename']) > 30 else record['filename'],
                    'Names': record['total_names'],
                    'Muslim': record['muslim_count'],
                    'Muslim %': f"{(record['muslim_count'] / record['total_names'] * 100):.1f}%" if record['total_names'] > 0 else "0%"
                })
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True)
    
    with tab4:
            st.header("📥 Download Center")
            
            if 'processed_files' not in st.session_state or len(st.session_state.processed_files) == 0:
                st.warning("⚠️ No data available for download. Please process and analyze PDF files first.")
                return
            
            st.subheader("📊 Available Downloads")
            
            # Create download columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📁 Individual File Downloads")
                
                # Individual file downloads
                for file_hash, record in st.session_state.processed_files.items():
                    with st.expander(f"📄 {record['filename']}"):
                        st.write(f"**Processed:** {record['timestamp']}")
                        st.write(f"**Total Names:** {record['total_names']}")
                        st.write(f"**Muslim Names:** {record['muslim_count']}")
                        
                        # Download buttons for individual files
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            # All extracted names CSV
                            extracted_df = pd.DataFrame(record['extracted_data'])
                            extracted_csv = extracted_df.to_csv(index=False)
                            
                            st.download_button(
                                label="📄 All Names CSV",
                                data=extracted_csv,
                                file_name=f"extracted_names_{record['filename'].replace('.pdf', '')}.csv",
                                mime="text/csv",
                                help="Download all extracted names from this file"
                            )
                        
                        with col_b:
                            # Muslim names only CSV
                            muslim_df = pd.DataFrame({'Name': record['muslim_names']})
                            muslim_csv = muslim_df.to_csv(index=False)
                            
                            st.download_button(
                                label="🕌 Muslim Names CSV",
                                data=muslim_csv,
                                file_name=f"muslim_names_{record['filename'].replace('.pdf', '')}.csv",
                                mime="text/csv",
                                help="Download only Muslim names from this file"
                            )
            
            with col2:
                st.markdown("### 📊 Consolidated Downloads")
                
                # Create consolidated data
                all_extracted_df, all_muslim_df = create_downloadable_files()
                
                # Summary report
                summary_data = create_summary_report()
                
                # Download buttons for consolidated data
                st.markdown("#### 📋 All Files Combined")
                
                # Consolidated all names
                if not all_extracted_df.empty:
                    consolidated_csv = all_extracted_df.to_csv(index=False)
                    st.download_button(
                        label="📊 All Names (Consolidated)",
                        data=consolidated_csv,
                        file_name=f"TMMK_consolidated_all_names_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download all names from all processed files",
                        type="primary"
                    )
                
                # Consolidated Muslim names
                if not all_muslim_df.empty:
                    consolidated_muslim_csv = all_muslim_df.to_csv(index=False)
                    st.download_button(
                        label="🕌 Muslim Names (Consolidated)",
                        data=consolidated_muslim_csv,
                        file_name=f"TMMK_consolidated_muslim_names_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download all Muslim names from all processed files",
                        type="primary"
                    )
                
                # Summary report
                if not summary_data.empty:
                    summary_csv = summary_data.to_csv(index=False)
                    st.download_button(
                        label="📈 Analysis Summary",
                        data=summary_csv,
                        file_name=f"TMMK_analysis_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download summary report of all processed files"
                    )
            
            st.markdown("---")
            
            # ZIP download section
            st.subheader("📦 Complete Package Download")
            st.info("💡 **Recommended:** Download everything as a ZIP file for easy organization")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("📦 Create & Download Complete Package", type="primary", use_container_width=True):
                    with st.spinner("📦 Creating ZIP package..."):
                        zip_data = create_zip_download()
                        
                        if zip_data:
                            st.success("✅ ZIP package created successfully!")
                            
                            # Create download button for ZIP
                            st.download_button(
                                label="⬇️ Download ZIP Package",
                                data=zip_data,
                                file_name=f"TMMK_Analysis_Package_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                help="Download all files and reports in a single ZIP package",
                                type="primary",
                                use_container_width=True
                            )
                            
                            # Show ZIP contents
                            st.markdown("### 📋 ZIP Package Contents:")
                            zip_contents = [
                                "📄 Individual extracted names (CSV for each file)",
                                "🕌 Individual Muslim names (CSV for each file)", 
                                "📊 Consolidated all names (CSV)",
                                "📊 Consolidated Muslim names (CSV)",
                                "📈 Analysis summary report (CSV)",
                                "📝 Processing log (JSON)"
                            ]
                            
                            for content in zip_contents:
                                st.write(f"• {content}")
                        else:
                            st.error("❌ Error creating ZIP package. Please try again.")
            
            st.markdown("---")
            
            # Session management
            st.subheader("🔄 Session Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Session", st.session_state.session_id)
            
            with col2:
                st.metric("Files in Session", len(st.session_state.processed_files))
            
            with col3:
                st.metric("Total Names", len(st.session_state.all_extracted_data))
            
            # Clear session button
            st.markdown("#### ⚠️ Session Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Reset Session", help="Clear all processed data and start fresh"):
                    # Clear all session data
                    keys_to_clear = [
                        'processed_files', 'all_extracted_data', 'all_muslim_names',
                        'session_folder', 'session_id', 'current_extracted_data',
                        'current_file_hash', 'current_filename', 'current_df'
                    ]
                    
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.success("✅ Session reset successfully!")
                    st.experimental_rerun()
            
            with col2:
                # Show session info
                if st.button("ℹ️ Session Info", help="Show detailed session information"):
                    st.json({
                        'session_id': st.session_state.session_id,
                        'files_processed': len(st.session_state.processed_files),
                        'total_names_extracted': len(st.session_state.all_extracted_data),
                        'total_muslim_names': len(st.session_state.all_muslim_names),
                        'files_list': [record['filename'] for record in st.session_state.processed_files.values()]
                    })
            
            # Instructions for users
            st.markdown("---")
            st.subheader("📋 Download Instructions")
            
            with st.expander("🔍 How to organize downloaded files"):
                st.markdown("""
                ### 📁 Recommended File Organization
                
                **After downloading, organize your files like this:**
                
                ```
                📁 TMMK_Analysis_[Date]/
                ├── 📁 Individual_Files/
                │   ├── extracted_names_file1.csv
                │   ├── muslim_names_file1.csv
                │   ├── extracted_names_file2.csv
                │   └── muslim_names_file2.csv
                ├── 📁 Consolidated_Results/
                │   ├── TMMK_consolidated_all_names.csv
                │   ├── TMMK_consolidated_muslim_names.csv
                │   └── TMMK_analysis_summary.csv
                └── 📄 processing_log.json
                ```
                
                ### 💡 Tips:
                - **Use the ZIP download** for automatic organization
                - **Individual CSV files** are good for specific file analysis
                - **Consolidated files** are perfect for overall statistics
                - **Summary report** gives you quick insights
                - **Processing log** contains technical details
                
                ### 🔄 Browser Downloads:
                - Files will appear in your browser's default download folder
                - You can change download location in browser settings
                - ZIP files will automatically extract when opened
                """)
            
            # Additional features
            st.markdown("---")
            st.subheader("🛠️ Additional Features")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Data Export Options")
                st.write("• CSV format for Excel compatibility")
                st.write("• JSON format for technical analysis") 
                st.write("• ZIP packaging for easy sharing")
                st.write("• Individual and consolidated reports")
            
            with col2:
                st.markdown("#### 🔍 Analysis Features")
                st.write("• Muslim name identification using AI")
                st.write("• Statistical analysis and percentages")
                st.write("• Visual charts and graphs")
                st.write("• Batch processing support")
    
    # Add this helper function if not already present
    def get_download_filename(base_name, extension="csv"):
        """Generate a standardized download filename with timestamp"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        return f"TMMK_{base_name}_{timestamp}.{extension}"
    
if __name__ == "__main__":
    main()
