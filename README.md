# 📄 Multi-Format Document Parser

**Team T34** - Project 2  
**Team Members**: Aryan & Bhavan

> A cost-effective hybrid document processing system that learns from each document to reduce AI costs while maintaining consistent JSON output across all file formats.

## 🎥 Demo Video

[🎬 **Watch Demo Video - Multi-Format Document Parser**](https://drive.google.com/file/d/1oxNRAwG5tCVIl5vQzUY_Wj23qi0Xkjcz/view?usp=sharing)

*Watch our 6-minute demo showcasing the hybrid pipeline, cost optimization, and signature learning in action.*

## 🚀 Problem Statement

Organizations receive documents in various formats (PDFs, scans, emails, HTML) but need consistent JSON output for downstream systems. Traditional AI-only approaches become prohibitively expensive at scale.

**Our Solution**: A hybrid pipeline that learns document patterns, creates reusable signatures, and uses AI sparingly - getting smarter and cheaper with every processed document.

## ✨ Key Features

### 🧠 **Smart Processing Pipeline**
- **Signature Matching**: Reuse learned patterns for cost-free processing
- **AI Extraction**: Dynamic schema generation with automatic signature creation  
- **Intelligent Fallbacks**: Always produces results, never fails silently

### 💰 **Cost Optimization**
- **Predictable Costs**: Each document type gets cheaper to process over time
- **Real-time Tracking**: Monitor AI usage and cost savings
- **Pattern Reuse**: 0-cost processing for recognized document layouts

### 📊 **Consistent Output** 
- **Normalized JSON**: Same structure regardless of input format
- **Smart Field Mapping**: Automatic categorization of extracted data
- **Table Extraction**: Preserves structured data from documents

### 🔍 **Full Interpretability**
- **Processing Logs**: Detailed explanation of every decision
- **Confidence Scores**: Know how reliable each extraction is
- **Strategy Tracking**: See which approach was used for each document


### Core Components

1. **HybridDocumentParser**: Main orchestrator managing the pipeline
2. **SignatureManager**: Learns and stores document patterns with versioning  
3. **AIExtractor**: Gemini-powered extraction with cost tracking
4. **Docling Integration**: Multi-format text extraction engine

## 📋 Prerequisites

- Python 3.11+
- UV package manager (recommended) or pip
- Google Gemini API Key
- **Tesseract OCR Engine** (for image/scanned document processing)
- 4GB+ RAM (for processing large documents)

## ⚡ Quick Start

### 1. Install Tesseract OCR

**Tesseract is required for processing images and scanned PDFs. Install it first before setting up the Python environment.**

#### Windows Installation

**Option 1: Download Installer (Recommended)**
1. Go to [Tesseract GitHub Releases](https://github.com/UB-Mannheim/tesseract/wiki)
2. Download the latest Windows installer (e.g., `tesseract-ocr-w64-setup-5.3.3.20231005.exe`)
3. Run the installer with administrator privileges
4. **Important**: During installation, note the installation path (usually `C:\Program Files\Tesseract-OCR`)

**Option 2: Using Chocolatey**
```powershell
# Install Chocolatey first if not installed
# Then run:
choco install tesseract
```

**Option 3: Using Winget**
```powershell
winget install --id UB-Mannheim.TesseractOCR
```

#### macOS Installation

**Using Homebrew (Recommended):**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Tesseract
brew install tesseract
```

**Using MacPorts:**
```bash
sudo port install tesseract
```

#### Linux Installation

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng
```

**CentOS/RHEL/Fedora:**
```bash
# CentOS/RHEL
sudo yum install tesseract tesseract-langpack-eng

# Fedora
sudo dnf install tesseract tesseract-langpack-eng
```

**Arch Linux:**
```bash
sudo pacman -S tesseract tesseract-data-eng
```

### 2. Set Up Tesseract Environment Variable

#### Windows
1. **Find Tesseract Installation Path**:
   - Default: `C:\Program Files\Tesseract-OCR`
   - If different, check your installation directory

2. **Add to System PATH**:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click "Environment Variables"
   - Under "System Variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\Program Files\Tesseract-OCR`
   - Click "OK" to save

3. **Alternative: Add to .env file**:
   ```bash
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   ```

#### macOS/Linux
Tesseract should be automatically available in PATH after installation. Verify with:
```bash
tesseract --version
```

If not found, add to your shell profile (`~/.bashrc`, `~/.zshrc`):
```bash
export PATH="/usr/local/bin:$PATH"  # macOS Homebrew
# or
export PATH="/opt/homebrew/bin:$PATH"  # macOS Apple Silicon
```

### 3. Clone the Repository
```bash
git clone https://github.com/your-username/multi-format-document-parser.git
cd multi-format-document-parser
```

### 4. Set Up Python Environment

**Using UV (Recommended):**
```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

**Using pip (Alternative):**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5. Configure API Key

**Step 1: Get Gemini API Key**
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Copy the generated key

**Step 2: Set Up Environment File**
```bash
# Copy the sample environment file
cp .env.sample .env
```

**Step 3: Add Your API Key**
Edit the `.env` file and add your Gemini API key:
```bash
# Required - Replace with your actual API key
GEMINI_API_KEY=your_actual_gemini_api_key_here

# Optional - Tesseract path (Windows only, if not in PATH)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Optional - Storage paths  
PARSER_DATA_PATH=parser_data
TEMP_FILES_PATH=temp

# Optional - Processing settings
DEFAULT_CONFIDENCE_THRESHOLD=0.8
MAX_AI_TOKENS=800
LOG_LEVEL=INFO
```

### 6. Verify Installation
```bash
# Test Tesseract installation
tesseract --version

# Test Python environment
python -c "import docling; print('Docling installed successfully')"
```

### 7. Run the Application
```bash
streamlit run app.py
```

The application will start and automatically open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
multi-format-document-parser/
├── README.md                 # This comprehensive guide
├── requirements.txt          # Python dependencies  
├── .env.sample              # Environment template
├── .env                     # Your actual environment (create this)
├── .gitignore               # Git ignore rules
├── app.py                   # Streamlit web interface
├── doc_parser.py            # Core parsing engine
├── demo/                    # Demo materials (optional)
│   └── sample_documents/    # Test files
├── parser_data/             # Auto-created signature storage
│   └── signatures.pkl       # Learned patterns (auto-generated)
└── temp/                    # Temporary processing files (auto-created)
```

## 🚀 Usage Guide

### First Time Setup
1. **Launch App**: Run `streamlit run app.py`
2. **Upload Document**: Drag and drop any supported file format
3. **Process**: Click "🚀 Process Documents"
4. **View Results**: Check the normalized JSON output and processing logs

### Document Processing Flow
1. **Upload**: Supports PDF, DOCX, images, HTML, CSV, and more
2. **Text Extraction**: Docling processes the document (with Tesseract OCR for images)
3. **Smart Processing**: 
   - First document: Uses AI extraction + creates signature
   - Similar documents: Uses signature (free!)
4. **JSON Output**: Consistent structured data every time

### Batch Processing
1. Enable "Batch Processing Mode" in sidebar
2. Upload multiple files
3. Monitor progress and costs in real-time
4. Download results individually or in bulk

## 📊 Understanding the Output

### Normalized JSON Schema

Every processed document returns this consistent structure:

```json
{
  "document_id": "abc123...",
  "document_type": "invoice",
  "processing_date": "2024-01-15T10:30:00Z",
  "sender": {
    "name": "ABC Company Ltd",
    "contact": "+1-555-0123",
    "address": "123 Business St"
  },
  "recipient": {
    "name": "Customer Name",
    "contact": "customer@email.com"
  },
  "metadata": {
    "date": "2024-01-10",
    "reference_number": "INV-2024-001",
    "subject": "Monthly Services"
  },
  "financial": {
    "currency": "USD",
    "subtotal": 1000.00,
    "tax": 100.00,
    "total": 1100.00
  },
  "line_items": [...],
  "tables": [...],
  "custom_fields": {...},
  "processing_info": {
    "strategy": "signature_match",
    "confidence": 0.95,
    "signature_id": "sig_abc123",
    "ai_calls": 0,
    "processing_time": 0.45
  }
}
```

### Processing Strategies

- **signature_match**: Used existing pattern (0 cost) ✅
- **ai_extraction**: Used AI to extract and learn (small cost, creates signature) 🤖

## 💰 Cost Optimization in Action

### How Costs Decrease Over Time

1. **First Document from Vendor A**: 
   - Uses AI extraction (~$0.0001-$0.001)
   - Creates signature automatically
   
2. **Second Document from Vendor A**: 
   - Uses signature matching (FREE!)
   - Processing time: ~0.2 seconds
   
3. **Result**: 90-95% cost reduction after learning

### Real-Time Monitoring

The app provides detailed cost tracking:
- Current session costs
- Per-document breakdown  
- Strategy distribution charts
- Optimization recommendations

## 🔍 Advanced Features

### Signature Management
- **Automatic Learning**: Every successful AI extraction creates a reusable signature
- **Export/Import**: Backup signatures or share across deployments
- **Version Control**: Signatures are versioned to prevent breaking changes

### Custom Processing Options
- **Sender ID**: Specify for better signature matching
- **Force AI Extraction**: Override signatures for testing
- **Batch Processing**: Process multiple files simultaneously

### Interpretation Logs
Every document processing includes detailed logs:
- Text extraction results
- Signature matching attempts  
- AI processing decisions
- Field mapping explanations
- Cost calculations

## 🐛 Troubleshooting

### Common Issues & Solutions

**❌ API Key Error**
```
Error: No Gemini API key found
Solution: Check your .env file - ensure GEMINI_API_KEY is set correctly
```

**❌ Tesseract Not Found**
```
Error: TesseractNotFoundError
Solutions:
1. Verify installation: tesseract --version
2. Add to PATH (Windows) or check installation path
3. Set TESSERACT_CMD in .env file
4. Restart terminal/command prompt after PATH changes
```

**❌ Import Error: docling**
```bash
Solution: pip install docling[pdf] or uv pip install docling[pdf]
```

**❌ Streamlit Port Already in Use**
```bash
Solution: streamlit run app.py --server.port 8502
```

**❌ Memory Issues with Large Files**
```
Solution: Process smaller batches or increase system memory
```

**❌ Signature Not Matching**
```
Solution: Check interpretation logs for pattern matching scores
Enable debug logging to see detailed matching process
```

**❌ OCR Not Working on Images**
```
Solutions:
1. Ensure Tesseract is installed and in PATH
2. Check image quality (should be clear, high contrast)
3. Try different image formats (PNG, JPEG, TIFF)
4. Verify TESSERACT_CMD path in .env file (Windows)
```

### Debug Mode

Enable detailed logging for troubleshooting:
```bash
export LOG_LEVEL=DEBUG
streamlit run app.py
```

### Testing Tesseract Installation

Create a test image with text and run:
```bash
# Test basic OCR functionality
tesseract test_image.png output_text.txt

# Check supported languages
tesseract --list-langs
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run app.py
```

## 🆘 Support & Contributing

**For Issues**:
1. Check existing GitHub Issues
2. Create detailed bug reports
3. Include error logs and system information

**For Contributions**:
1. Fork the repository
2. Create feature branch
3. Submit pull request with clear description

## 🎖️ Acknowledgments

- **Docling Team**: Excellent multi-format document processing
- **Google AI**: Gemini API for intelligent extraction
- **Tesseract OCR**: Robust optical character recognition
- **Streamlit**: Intuitive web framework
- **Competition Organizers**: Challenging real-world problem

---

**🏆 Built by Team T34 - Aryan & Bhavan**

*Transforming document processing with intelligence, efficiency, and cost-effectiveness.*