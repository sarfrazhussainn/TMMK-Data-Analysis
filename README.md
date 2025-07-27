# PDF Name & Muslim Name Analyzer

A Streamlit web application that extracts names from PDF documents and uses Google Gemini AI to identify Muslim names with statistical analysis and visualizations.

## 🚀 Live Demo

Access the deployed application: [Your Streamlit Cloud URL will be here]

## ✨ Features

- 📄 **PDF Processing**: Extract names and guardian names from structured PDF documents
- 🤖 **AI-Powered Analysis**: Use Google Gemini AI to identify Muslim names
- 📊 **Statistical Analysis**: Comprehensive statistics and visualizations
- 💾 **Data Management**: Save and organize analysis results in CSV format
- 📁 **File Organization**: Automatic folder management and file versioning
- 📈 **Interactive Visualizations**: Charts, graphs, and analytics dashboard

## 🛠️ How to Use

### Prerequisites
- Get a **Google Gemini API key** from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Prepare PDF files with structured name data

### Steps
1. **Select Working Directory**: Choose or create a folder for your analysis
2. **Upload PDF**: Upload your PDF file containing names
3. **Enter API Key**: Input your Google Gemini API key in the sidebar
4. **Process PDF**: Extract names and guardian information
5. **Analyze Names**: Run AI analysis to identify Muslim names  
6. **View Results**: Explore statistics, charts, and download data

## 📋 Requirements

- Python 3.8+
- Google Gemini API key (users provide their own)
- PDF files with structured name/guardian data
- Internet connection for API calls

## 🔧 Local Development

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/pdf-muslim-name-analyzer.git
cd pdf-muslim-name-analyzer
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

## 📊 Analysis Features

- **Name Extraction**: Regex-based extraction from PDF text
- **Muslim Name Identification**: AI-powered classification
- **Statistical Reports**: Percentages, counts, and distributions
- **Data Visualization**: Bar charts, pie charts, and gauge charts
- **Batch Processing**: Handle multiple files with consolidated results
- **Export Options**: Download results as CSV files

## 🔒 Privacy & Security

- **No API Key Storage**: Users enter their own Google Gemini API keys
- **Local Processing**: PDF processing happens locally
- **Temporary Storage**: Files are managed in user-selected directories
- **No Data Retention**: Application doesn't store user data permanently

## 📝 File Structure

```
pdf-muslim-name-analyzer/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── .gitignore           # Git ignore rules
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## ⚠️ Important Notes

- **API Costs**: Users are responsible for their Google Gemini API usage costs
- **PDF Format**: Works best with structured PDF documents containing name tables
- **Processing Time**: Large PDFs may take longer to process
- **Internet Required**: Requires internet connection for AI analysis

## 🆘 Support

If you encounter issues:
1. Check that your PDF has the expected structure
2. Verify your Google Gemini API key is valid
3. Ensure stable internet connection
4. Try smaller batch sizes for large datasets

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

**Made with ❤️ using Streamlit and Google Gemini AI**