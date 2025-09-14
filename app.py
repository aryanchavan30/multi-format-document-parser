import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the core parser module
from doc_parser import (
    HybridDocumentParser,
    ProcessingStrategy,
    DocumentType,
    ProcessingResult
)

# Page configuration
st.set_page_config(
    page_title="Multi-Format Document Parser",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 2px solid #e0e0e0;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .success-msg {
        color: #28a745;
        font-weight: bold;
    }
    .error-msg {
        color: #dc3545;
        font-weight: bold;
    }
    .info-msg {
        color: #17a2b8;
        font-style: italic;
    }
    .log-container {
        background-color: #2b2b2b;
        color: #f0f0f0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        height: 400px;
        overflow-y: auto;
    }
    .json-viewer {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        max-height: 500px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


class DocumentParserApp:
    """Streamlit application for document parsing"""
    
    def __init__(self):
        self.init_session_state()
        self.parser = self.init_parser()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        if 'total_cost' not in st.session_state:
            st.session_state.total_cost = 0.0
        if 'signatures_count' not in st.session_state:
            st.session_state.signatures_count = 0
    
    def init_parser(self) -> HybridDocumentParser:
        """Initialize the document parser"""
        # Try to get API key from multiple sources with proper fallback
        api_key = None
        
        # Priority order: .env file > Streamlit secrets > environment variable
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            try:
                api_key = st.secrets.get("GEMINI_API_KEY")
            except:
                pass
        
        if not api_key:
            st.sidebar.error("🚨 No Gemini API key found!")
            st.sidebar.info("Please set GEMINI_API_KEY in your .env file or environment variables")
            return None
        
        enable_ai = api_key is not None
        storage_path = os.getenv("PARSER_DATA_PATH", "parser_data")
        
        if not enable_ai:
            st.sidebar.warning("⚠️ No Gemini API key found. AI extraction disabled.")
        
        return HybridDocumentParser(
            enable_ai=enable_ai,
            api_key=api_key,
            storage_path=storage_path
        )
    
    def run(self):
        """Run the Streamlit app"""
        if not self.parser:
            st.error("⚠️ Parser initialization failed. Please check your API key configuration.")
            return
            
        # Header
        st.markdown('<h1 class="main-header">📄 Multi-Format Document Parser</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        self.render_sidebar()
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📤 Upload & Process", 
            "📊 Processing Status", 
            "💾 JSON Output",
            "📈 Analytics",
            "🔧 Settings"
        ])
        
        with tab1:
            self.render_upload_tab()
        
        with tab2:
            self.render_status_tab()
        
        with tab3:
            self.render_json_tab()
        
        with tab4:
            self.render_analytics_tab()
        
        with tab5:
            self.render_settings_tab()
    
    def render_sidebar(self):
        """Render sidebar with statistics and controls"""
        st.sidebar.header("📊 Processing Statistics")
        
        # Calculate stats directly from session state instead of parser stats
        if st.session_state.processing_history:
            total_processed = len(st.session_state.processing_history)
            sig_matched = sum(1 for item in st.session_state.processing_history 
                            if item['strategy'] == 'signature_match')
            rule_based = sum(1 for item in st.session_state.processing_history 
                            if item['strategy'] == 'rule_based')
            ai_processed = sum(1 for item in st.session_state.processing_history 
                            if item['strategy'] == 'ai_extraction')
            total_ai_calls = sum(item.get('ai_calls', 0) for item in st.session_state.processing_history)
            total_cost = sum(item.get('estimated_cost', 0) for item in st.session_state.processing_history)
        else:
            total_processed = sig_matched = rule_based = ai_processed = total_ai_calls = total_cost = 0
        
        # Display stats
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.metric("Total Processed", total_processed)
            st.metric("AI Processed", ai_processed)
        
        with col2:
            st.metric("Signatures", len(self.parser.signature_manager.signatures) if self.parser else 0)
            st.metric("AI Calls", total_ai_calls)
            st.metric("Est. Cost", f"${total_cost:.4f}")
        
        st.sidebar.divider()
        
        # Processing options
        st.sidebar.header("⚙️ Processing Options")
        
        st.session_state.sender_id = st.sidebar.text_input(
            "Sender ID (optional)",
            help="Specify sender ID for better signature matching"
        )
        
        st.session_state.force_ai = st.sidebar.checkbox(
            "Force AI Extraction",
            help="Always use AI instead of signatures"
        )
        
        st.session_state.batch_mode = st.sidebar.checkbox(
            "Batch Processing Mode",
            help="Process multiple files at once"
        )
        
        # Export/Import buttons
        st.sidebar.divider()
        st.sidebar.header("📁 Data Management")
        
        if st.sidebar.button("📥 Export Signatures"):
            self.export_signatures()
        
        uploaded_sigs = st.sidebar.file_uploader(
            "📤 Import Signatures",
            type=['json'],
            key="sig_import"
        )
        if uploaded_sigs:
            self.import_signatures(uploaded_sigs)
    
    def render_upload_tab(self):
        """Render file upload and processing tab"""
        st.header("Upload Documents")
        
        # File uploader
        if st.session_state.batch_mode:
            uploaded_files = st.file_uploader(
                "Choose files to process",
                type=['pdf', 'docx', 'xlsx', 'pptx', 'html', 'md', 'txt', 
                      'png', 'jpg', 'jpeg', 'csv'],
                accept_multiple_files=True,
                key="file_uploader"
            )
        else:
            uploaded_file = st.file_uploader(
                "Choose a file to process",
                type=['pdf', 'docx', 'xlsx', 'pptx', 'html', 'md', 'txt',
                      'png', 'jpg', 'jpeg', 'csv'],
                accept_multiple_files=False,
                key="single_file_uploader"
            )
            uploaded_files = [uploaded_file] if uploaded_file else []
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} file(s) selected")
            
            # Process button
            if st.button("🚀 Process Documents", type="primary"):
                self.process_documents(uploaded_files)
        
        # Display recent results
        if st.session_state.current_result:
            st.divider()
            self.display_current_result()
    
    def render_status_tab(self):
        """Render processing status tab"""
        st.header("Processing Status")
        
        if not st.session_state.processing_history:
            st.info("No documents processed yet. Upload files in the first tab to begin.")
            return
        
        # Create status table
        df_data = []
        for item in st.session_state.processing_history:
            df_data.append({
                'Filename': item['filename'],
                'Status': '✅' if item['success'] else '❌',
                'Strategy': item['strategy'],
                'Confidence': f"{item['confidence']:.2%}",
                'AI Calls': item['ai_calls'],
                'Time (s)': f"{item['processing_time']:.2f}",
                'Timestamp': item['timestamp']
            })
        
        df = pd.DataFrame(df_data)
        
        # Display with custom styling
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Status': st.column_config.TextColumn('Status', width='small'),
                'Confidence': st.column_config.ProgressColumn(
                    'Confidence',
                    min_value=0,
                    max_value=1,
                    format='%.0%'
                ),
            }
        )
        
        # Show interpretation log for selected document
        st.subheader("📋 Interpretation Logs")
        
        selected_doc = st.selectbox(
            "Select document to view log",
            [item['filename'] for item in st.session_state.processing_history]
        )
        
        if selected_doc:
            doc_data = next(
                item for item in st.session_state.processing_history 
                if item['filename'] == selected_doc
            )
            
            if 'interpretation_log' in doc_data:
                log_text = '\n'.join(doc_data['interpretation_log'])
                st.text_area(
                    "Processing Log", 
                    log_text, 
                    height=400,
                    disabled=True
                )
    
    def render_json_tab(self):
        """Render JSON output tab"""
        st.header("JSON Output")
        
        if not st.session_state.processing_history:
            st.info("No documents processed yet.")
            return
        
        # Select document
        doc_options = [item['filename'] for item in st.session_state.processing_history]
        selected_doc = st.selectbox(
            "Select document",
            doc_options,
            key="json_doc_select"
        )
        
        if selected_doc:
            doc_data = next(
                item for item in st.session_state.processing_history 
                if item['filename'] == selected_doc
            )
            
            # Display JSON
            st.subheader("Normalized JSON")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # JSON viewer
                json_str = json.dumps(doc_data['normalized_json'], indent=2)
                st.code(json_str, language='json')
            
            with col2:
                # Download button
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"{selected_doc}_normalized.json",
                    mime="application/json"
                )
                
                # Validation status
                st.metric("Fields Extracted", 
                         len([v for v in doc_data['normalized_json'].get('custom_fields', {}).values() if v]))
                
                if doc_data['normalized_json'].get('tables'):
                    st.metric("Tables Found", len(doc_data['normalized_json']['tables']))
    
    def render_analytics_tab(self):
        """Render analytics and insights tab"""
        st.header("Analytics & Insights")
        
        if not st.session_state.processing_history:
            st.info("No data available yet. Process some documents first.")
            return
        
        # Calculate stats from session state
        total_processed = len(st.session_state.processing_history)
        sig_matched = sum(1 for item in st.session_state.processing_history 
                        if item['strategy'] == 'signature_match')
        ai_processed = sum(1 for item in st.session_state.processing_history 
                        if item['strategy'] == 'ai_extraction')
        total_ai_calls = sum(item.get('ai_calls', 0) for item in st.session_state.processing_history)
        total_cost = sum(item.get('estimated_cost', 0) for item in st.session_state.processing_history)
        
        # Create local stats dict
        stats = {
            'total_processed': total_processed,
            'signature_matched': sig_matched,
            'ai_processed': ai_processed,
            'total_ai_calls': total_ai_calls,
            'total_cost': total_cost
        }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Processing strategy distribution
            strategy_labels = []
            strategy_values = []
            
            if stats['signature_matched'] > 0:
                strategy_labels.append('Signature Match')
                strategy_values.append(stats['signature_matched'])
            
            if stats['ai_processed'] > 0:
                strategy_labels.append('AI Extraction')
                strategy_values.append(stats['ai_processed'])
            
            if strategy_labels:
                fig_strategy = go.Figure(data=[
                    go.Pie(
                        labels=strategy_labels,
                        values=strategy_values,
                        hole=0.3
                    )
                ])
                fig_strategy.update_layout(
                    title="Processing Strategy Distribution",
                    height=300,
                    margin=dict(t=50, b=20, l=20, r=20)
                )
                st.plotly_chart(fig_strategy, use_container_width=True)
            else:
                st.info("No strategy data available")
        
        with col2:
            # Cost breakdown
            if stats['total_ai_calls'] > 0:
                avg_cost = stats['total_cost'] / stats['total_ai_calls']
                st.metric("Average Cost per AI Call", f"${avg_cost:.6f}")
            else:
                st.metric("Average Cost per AI Call", "$0.000000")
            
            st.metric("Total Cost", f"${stats['total_cost']:.4f}")
            
            # Cost projection
            if stats['total_processed'] > 0:
                cost_per_doc = stats['total_cost'] / stats['total_processed']
                st.info(f"Average cost per document: ${cost_per_doc:.6f}")
        
        with col3:
            # Efficiency metrics
            if stats['total_processed'] > 0:
                sig_percentage = (stats['signature_matched'] / stats['total_processed']) * 100
                ai_avoidance = 100 - ((stats['ai_processed'] / stats['total_processed']) * 100)
            else:
                sig_percentage = ai_avoidance = 0
            
            st.metric("Signature Match %", f"{sig_percentage:.1f}%")
            st.metric("AI Avoidance Rate", f"{ai_avoidance:.1f}%")
    
    def render_settings_tab(self):
        """Render settings and configuration tab"""
        st.header("Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Parser Configuration")
            
            # Environment variables display
            st.subheader("🔑 Environment Configuration")
            api_key = os.getenv("GEMINI_API_KEY", "Not Set")
            st.text_input("Gemini API Key", value=f"{api_key[:10]}..." if len(api_key) > 10 else api_key, disabled=True)
            st.info("Set your API key in the .env file")
            
            # Storage paths
            storage_path = os.getenv("PARSER_DATA_PATH", "parser_data")
            st.text_input("Storage Path", value=storage_path, disabled=True)
        
        with col2:
            st.subheader("📊 Signature Management")
            
            # Signature statistics
            if self.parser:
                total_sigs = len(self.parser.signature_manager.signatures)
                sender_sigs = len(self.parser.signature_manager.sender_signatures)
                global_sigs = len(self.parser.signature_manager.global_signatures)
                
                st.metric("Total Signatures", total_sigs)
                st.metric("Sender-Specific", sender_sigs)
                st.metric("Global Signatures", global_sigs)
                
                # Signature actions
                if st.button("🗑️ Clear All Signatures"):
                    if st.checkbox("Confirm deletion"):
                        self.parser.signature_manager.signatures.clear()
                        self.parser.signature_manager.sender_signatures.clear()
                        self.parser.signature_manager.global_signatures.clear()
                        self.parser.signature_manager.save_signatures()
                        st.success("All signatures cleared!")
                        st.rerun()
    
    def process_documents(self, files):
        """Process uploaded documents"""
        if not self.parser:
            st.error("Parser not available. Check your API key configuration.")
            return
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        total_files = len(files)
        temp_dir = Path(os.getenv("TEMP_FILES_PATH", "temp"))
        temp_dir.mkdir(exist_ok=True)
        
        for idx, file in enumerate(files):
            # Update progress
            progress = (idx + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Processing {file.name}... ({idx + 1}/{total_files})")
            
            # Save file temporarily
            temp_path = temp_dir / file.name
            
            with open(temp_path, 'wb') as f:
                f.write(file.getbuffer())
            
            try:
                # Process document
                result = self.parser.process_document(
                    str(temp_path),
                    sender_id=st.session_state.get('sender_id'),
                    force_ai=st.session_state.get('force_ai', False)
                )
                
                # Store result
                history_item = {
                    'filename': file.name,
                    'success': result.success,
                    'strategy': result.processing_strategy.value,
                    'confidence': result.confidence_score,
                    'ai_calls': result.ai_calls_used,
                    'processing_time': result.processing_time,
                    'normalized_json': result.normalized_json,
                    'interpretation_log': result.interpretation_log,
                    'estimated_cost': result.estimated_cost,
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.processing_history.append(history_item)
                results.append(result)
                
                # Update cost
                st.session_state.total_cost += result.estimated_cost
                
            except Exception as e:
                st.error(f"Error processing {file.name}: {str(e)}")
            finally:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
        
        # Clear progress
        progress_bar.empty()
        status_text.empty()
        
        # Show summary
        successful = sum(1 for r in results if r.success)
        st.success(f"✅ Processed {successful}/{total_files} documents successfully!")
        
        if successful < total_files:
            failed = total_files - successful
            st.warning(f"⚠️ {failed} document(s) failed to process")
        
        # Update current result
        if results:
            st.session_state.current_result = results[-1]
    
    def display_current_result(self):
        """Display the current processing result"""
        result = st.session_state.current_result
        
        st.subheader("Latest Processing Result")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if result.success:
                st.success("✅ Success")
            else:
                st.error("❌ Failed")
        
        with col2:
            st.metric("Strategy", result.processing_strategy.value)
        
        with col3:
            st.metric("Confidence", f"{result.confidence_score:.2%}")
        
        with col4:
            st.metric("Processing Time", f"{result.processing_time:.2f}s")
        
        # Show sample of extracted data
        if result.normalized_json:
            st.subheader("Extracted Data Sample")
            
            # Show key fields
            metadata = result.normalized_json.get('metadata', {})
            financial = result.normalized_json.get('financial', {})
            
            if metadata:
                st.write("**Metadata:**")
                for key, value in metadata.items():
                    if value:
                        st.write(f"- {key}: {value}")
            
            if financial and any(financial.values()):
                st.write("**Financial Information:**")
                for key, value in financial.items():
                    if value:
                        st.write(f"- {key}: {value}")
    
    def export_signatures(self):
        """Export signatures to file"""
        if not self.parser:
            st.sidebar.error("Parser not available")
            return
            
        try:
            export_path = "exported_signatures.json"
            self.parser.export_signatures(export_path)
            
            with open(export_path, 'r') as f:
                data = f.read()
            
            st.sidebar.download_button(
                label="💾 Download Signatures",
                data=data,
                file_name=f"signatures_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.sidebar.success("Signatures exported successfully!")
        except Exception as e:
            st.sidebar.error(f"Export failed: {str(e)}")
    
    def import_signatures(self, file):
        """Import signatures from file"""
        if not self.parser:
            st.sidebar.error("Parser not available")
            return
            
        try:
            # Save uploaded file temporarily
            temp_path = "temp_import.json"
            with open(temp_path, 'wb') as f:
                f.write(file.getbuffer())
            
            # Import signatures
            self.parser.import_signatures(temp_path)
            
            # Clean up
            Path(temp_path).unlink()
            
            st.sidebar.success("Signatures imported successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Import failed: {str(e)}")


# Main execution
if __name__ == "__main__":
    app = DocumentParserApp()
    app.run()