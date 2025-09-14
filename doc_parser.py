import os
import json
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import pickle
import logging
from enum import Enum

# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, 
    TableFormerMode,
    EasyOcrOptions
)
from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.document import InputDocument

# Gemini imports for AI processing (sparse usage)
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentType(Enum):
    """Document type classification"""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    FORM = "form"
    REPORT = "report"
    EMAIL = "email"
    LETTER = "letter"
    UNKNOWN = "unknown"


class ProcessingStrategy(Enum):
    """Processing strategy for cost optimization"""
    RULE_BASED = "rule_based"          # Use existing rules/patterns
    SIGNATURE_MATCH = "signature_match" # Match against known signatures
    HYBRID = "hybrid"                   # Mix of rules + minimal AI
    AI_EXTRACTION = "ai_extraction"     # Full AI extraction (last resort)


@dataclass
class DocumentSignature:
    signature_id: str
    sender_id: Optional[str]
    document_type: DocumentType
    key_patterns: List[str]
    field_positions: Dict[str, str]
    extraction_rules: Dict[str, str]
    
    # NEW FIELDS:
    dynamic_schema: Dict[str, Any] = None  # Store the extracted schema
    field_regex_patterns: Dict[str, str] = None  # Better regex patterns per field
    sample_data: Dict[str, Any] = None  # Sample extracted data for reference
    
    confidence_threshold: float = 0.8
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    usage_count: int = 0


@dataclass
class ProcessingResult:
    """Result of document processing"""
    success: bool
    normalized_json: Dict[str, Any]
    processing_strategy: ProcessingStrategy
    confidence_score: float
    interpretation_log: List[str]
    processing_time: float
    ai_calls_used: int
    estimated_cost: float
    signature_matched: Optional[str] = None
    errors: List[str] = None


class SignatureManager:
    """Manages document signatures and pattern learning"""
    
    def __init__(self, storage_path: str = "signatures"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.signatures: Dict[str, DocumentSignature] = {}
        self.sender_signatures: Dict[str, List[str]] = {}
        self.global_signatures: List[str] = []
        self.load_signatures()
    
    def load_signatures(self):
        """Load existing signatures from storage"""
        signature_file = self.storage_path / "signatures.pkl"
        if signature_file.exists():
            try:
                with open(signature_file, 'rb') as f:
                    data = pickle.load(f)
                    self.signatures = data.get('signatures', {})
                    self.sender_signatures = data.get('sender_signatures', {})
                    self.global_signatures = data.get('global_signatures', [])
                logger.info(f"Loaded {len(self.signatures)} signatures")
            except Exception as e:
                logger.error(f"Error loading signatures: {e}")
    
    def save_signatures(self):
        """Save signatures to storage"""
        signature_file = self.storage_path / "signatures.pkl"
        try:
            with open(signature_file, 'wb') as f:
                pickle.dump({
                    'signatures': self.signatures,
                    'sender_signatures': self.sender_signatures,
                    'global_signatures': self.global_signatures
                }, f)
            logger.info(f"Saved {len(self.signatures)} signatures")
        except Exception as e:
            logger.error(f"Error saving signatures: {e}")
    
    def find_matching_signature(self, text: str, sender_id: Optional[str] = None) -> Optional[DocumentSignature]:
        """Find best matching signature for document"""
        best_match = None
        best_score = 0.0
        
        # Check sender-specific signatures first
        if sender_id and sender_id in self.sender_signatures:
            for sig_id in self.sender_signatures[sender_id]:
                if sig_id in self.signatures:
                    score = self._calculate_signature_match(text, self.signatures[sig_id])
                    if score > best_score:
                        best_score = score
                        best_match = self.signatures[sig_id]
        
        # Check global signatures
        for sig_id in self.global_signatures:
            if sig_id in self.signatures:
                score = self._calculate_signature_match(text, self.signatures[sig_id])
                if score > best_score:
                    best_score = score
                    best_match = self.signatures[sig_id]
        
        if best_match and best_score >= best_match.confidence_threshold:
            best_match.usage_count += 1
            return best_match
        
        return None
    
    def _calculate_signature_match(self, text: str, signature: DocumentSignature) -> float:
        """Calculate match score between text and signature"""
        score = 0.0
        text_lower = text.lower()
        
        # Check key patterns
        patterns_found = 0
        for pattern in signature.key_patterns:
            if pattern.lower() in text_lower:
                patterns_found += 1
        
        if signature.key_patterns:
            score = patterns_found / len(signature.key_patterns)
        
        return score
    
    def create_signature(self, text: str, extracted_data: Dict[str, Any], dynamic_schema: Dict[str, Any] = None,
                     regex_patterns: Dict[str, str] = None,
                         sender_id: Optional[str] = None, 
                         document_type: DocumentType = DocumentType.UNKNOWN) -> DocumentSignature:
        """Create new signature from document"""
        sig_id = hashlib.md5(f"{sender_id}_{document_type.value}_{datetime.now()}".encode()).hexdigest()[:12]
        
        # Extract key patterns from text
        key_patterns = self._extract_key_patterns(text)
        
        # Create extraction rules based on extracted data
        extraction_rules = self._create_extraction_rules(text, extracted_data)
        
        signature = DocumentSignature(
            signature_id=sig_id,
            sender_id=sender_id,
            dynamic_schema=dynamic_schema,
            field_regex_patterns=regex_patterns,
            sample_data=extracted_data,
            document_type=document_type,
            key_patterns=key_patterns,
            field_positions={},
            extraction_rules=extraction_rules,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        # Store signature
        self.signatures[sig_id] = signature
        
        if sender_id:
            if sender_id not in self.sender_signatures:
                self.sender_signatures[sender_id] = []
            self.sender_signatures[sender_id].append(sig_id)
        else:
            self.global_signatures.append(sig_id)
        
        self.save_signatures()
        return signature
    
    def _extract_key_patterns(self, text: str) -> List[str]:
        """Extract key patterns from document text"""
        patterns = []
        
        # Look for headers and labels
        header_patterns = [
            r'^[A-Z][A-Z\s]+:',  # CAPS headers
            r'^\*\*(.+?)\*\*',    # Bold markdown
            r'^#{1,3}\s+(.+)',    # Markdown headers
        ]
        
        for pattern in header_patterns:
            matches = re.findall(pattern, text, re.MULTILINE)
            patterns.extend([m if isinstance(m, str) else m[0] for m in matches[:5]])
        
        return patterns[:10]  # Limit to top 10 patterns
    
    def _create_extraction_rules(self, text: str, extracted_data: Dict[str, Any]) -> Dict[str, str]:
        """Create extraction rules from successfully extracted data"""
        rules = {}
        
        for field, value in extracted_data.items():
            if isinstance(value, (str, int, float)) and str(value).strip():
                value_str = str(value).strip()
                if value_str in text:
                    # Create more robust patterns
                    if field == 'invoice_number' or field == 'reference_number':
                        rules[field] = r'(?:Number|Invoice|Bill):\s*([A-Z0-9]+)'
                    elif field == 'date':
                        rules[field] = r'(?:Date):\s*(\d{1,2}\s+\w+\s+\d{4})'
                    elif field == 'total_amount' or field == 'total':
                        rules[field] = r'(?:Grand\s+Total|Total).*?(\d+(?:,\d{3})*(?:\.\d{2})?)'
                    elif field == 'vendor_name':
                        rules[field] = r'^([A-Za-z\s&.,Ltd]+(?:Limited|Ltd|Inc|Corp)?)$'
                    elif field == 'customer_name':
                        rules[field] = r'(?:Customer\s+Name|GSTN\s+Customer\s+Name):\s*([A-Za-z\s&.,]+)'
                    else:
                        # Generic pattern - look for the value after a colon or label
                        escaped_value = re.escape(value_str)
                        # Find context around the value
                        value_index = text.find(value_str)
                        if value_index > 50:
                            context_start = max(0, value_index - 50)
                            context = text[context_start:value_index]
                            
                            # Look for label pattern
                            label_match = re.search(r'([A-Za-z\s]+):\s*$', context)
                            if label_match:
                                label = label_match.group(1).strip()
                                rules[field] = f"{re.escape(label)}:\s*([^\n\r]+)"
        
        return rules
    
    def _create_regex_pattern(self, context: str, value: str) -> Optional[str]:
        """Create regex pattern for value extraction"""
        # Escape special regex characters in value
        escaped_value = re.escape(value)
        
        # Look for label pattern before value
        before_text = context[:context.find(value)]
        if ':' in before_text:
            label = before_text.split(':')[-1].strip()
            if label:
                return f"{re.escape(label)}:?\s*(.+)"
        
        return None
    def cleanup_invalid_signatures(self):
        """Remove signatures with no extraction rules or empty patterns"""
        invalid_sigs = []
        
        for sig_id, signature in self.signatures.items():
            if (not signature.extraction_rules or 
                not signature.key_patterns or 
                len(signature.extraction_rules) == 0):
                invalid_sigs.append(sig_id)
        
        for sig_id in invalid_sigs:
            del self.signatures[sig_id]
            # Remove from sender signatures
            for sender_id, sig_list in self.sender_signatures.items():
                if sig_id in sig_list:
                    sig_list.remove(sig_id)
            # Remove from global signatures
            if sig_id in self.global_signatures:
                self.global_signatures.remove(sig_id)
        
        if invalid_sigs:
            logger.info(f"Cleaned up {len(invalid_sigs)} invalid signatures")
            self.save_signatures()
    def get_signature_details(self, signature_id: str) -> Dict[str, Any]:
        """Get detailed information about a signature for inspection"""
        if signature_id not in self.signatures:
            return None
        
        sig = self.signatures[signature_id]
        return {
            'signature_id': sig.signature_id,
            'document_type': sig.document_type.value,
            'usage_count': sig.usage_count,
            'dynamic_schema': sig.dynamic_schema,
            'sample_data': sig.sample_data,
            'regex_patterns': sig.field_regex_patterns,
            'created_at': sig.created_at,
        }

    def list_all_signatures(self) -> List[Dict[str, Any]]:
        """List all signatures with basic info"""
        return [
            {
                'id': sig_id,
                'type': sig.document_type.value,
                'sender': sig.sender_id,
                'usage_count': sig.usage_count,
                'created': sig.created_at
            }
            for sig_id, sig in self.signatures.items()
        ]


class RuleBasedExtractor:
    """Rule-based extraction engine"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize extraction rules"""
        return {
            'invoice': [
                {'field': 'invoice_number', 'patterns': [
                    r'invoice\s*(?:no|number|#)?\s*:?\s*([A-Z0-9-]+)',
                    r'bill\s*(?:no|number|#)?\s*:?\s*([A-Z0-9-]+)',
                ]},
                {'field': 'date', 'patterns': [
                    r'date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                    r'dated?\s*:?\s*(\d{1,2}\s+\w+\s+\d{4})',
                ]},
                {'field': 'total_amount', 'patterns': [
                    r'total\s*(?:amount)?\s*:?\s*[$₹]?\s*([\d,]+\.?\d*)',
                    r'grand\s*total\s*:?\s*[$₹]?\s*([\d,]+\.?\d*)',
                ]},
                {'field': 'vendor_name', 'patterns': [
                    r'from\s*:?\s*([A-Za-z\s&.,]+?)(?:\n|$)',
                    r'vendor\s*:?\s*([A-Za-z\s&.,]+?)(?:\n|$)',
                ]},
            ],
            'receipt': [
                {'field': 'receipt_number', 'patterns': [
                    r'receipt\s*(?:no|number|#)?\s*:?\s*([A-Z0-9-]+)',
                    r'transaction\s*(?:id|no)?\s*:?\s*([A-Z0-9-]+)',
                ]},
                {'field': 'amount', 'patterns': [
                    r'amount\s*:?\s*[$₹]?\s*([\d,]+\.?\d*)',
                    r'paid\s*:?\s*[$₹]?\s*([\d,]+\.?\d*)',
                ]},
            ],
        }
    
    def extract(self, text: str, document_type: str = 'invoice') -> Dict[str, Any]:
        """Extract data using rules"""
        extracted = {}
        rules = self.rules.get(document_type, self.rules['invoice'])
        
        for rule in rules:
            field = rule['field']
            for pattern in rule['patterns']:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    extracted[field] = match.group(1).strip()
                    break
        
        return extracted
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown text"""
        tables = []
        lines = text.split('\n')
        
        current_table = []
        in_table = False
        
        for line in lines:
            if '|' in line and line.count('|') >= 2:
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line)
            else:
                if in_table and current_table:
                    # Process table
                    table_data = self._parse_markdown_table(current_table)
                    if table_data:
                        tables.append(table_data)
                    current_table = []
                    in_table = False
        
        # Handle last table
        if current_table:
            table_data = self._parse_markdown_table(current_table)
            if table_data:
                tables.append(table_data)
        
        return tables
    
    def _parse_markdown_table(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse markdown table lines"""
        if len(lines) < 3:
            return None
        
        # Extract headers
        headers = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
        
        # Extract rows (skip separator line)
        rows = []
        for line in lines[2:]:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                rows.append(cells)
        
        return {
            'headers': headers,
            'rows': rows,
            'row_count': len(rows),
            'column_count': len(headers)
        }


class AIExtractor:
    """AI-based extraction using Gemini (sparse usage)"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.client = None
        self.call_count = 0
        self.total_tokens = 0
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
    
    def extract_with_schema(self, text: str, schema: Dict[str, Any], 
                           max_tokens: int = 500) -> Tuple[Dict[str, Any], int]:
        """Extract data using AI with defined schema"""
        if not self.client:
            raise Exception("AI client not available")
        
        self.call_count += 1
        
        prompt = f"""Extract the following information from the document text.
Return ONLY the requested fields in JSON format.

Schema:
{json.dumps(schema, indent=2)}

Document text:
{text}  

Output JSON:"""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            print("="*80)
            print("This is result JSON :- ")
            print(result)
            print("="*80)
            tokens_used = len(prompt) // 4 + len(response.text) // 4
            self.total_tokens += tokens_used
            
            return result, tokens_used
            
        except Exception as e:
            logger.error(f"AI extraction error: {e}")
            # Don't return empty dict, raise the exception
            raise e
    
    def classify_document(self, text: str) -> DocumentType:
        """Classify document type using AI"""
        if not self.client:
            return DocumentType.UNKNOWN
        
        prompt = f"""Classify this document into one of these categories:
- invoice
- receipt
- contract
- form
- report
- email
- letter

Document (first 2000 chars):
{text[:2000]}

Return only the category name."""
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=20
                )
            )
            
            category = response.text.strip().lower()
            return DocumentType(category) if category in [e.value for e in DocumentType] else DocumentType.UNKNOWN
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return DocumentType.UNKNOWN
    
    def get_estimated_cost(self) -> float:
        """Estimate API cost based on token usage"""
        if self.total_tokens == 0:
            return 0.0
        
        
        cost_per_1m_tokens = 0.1875
        return (self.total_tokens / 1_000_000) * cost_per_1m_tokens
    
    def extract_with_dynamic_schema(self, text: str, max_tokens: int = 800) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, str], int]:
        """Extract data, schema, and regex patterns dynamically"""
        if not self.client:
            raise Exception("AI client not available")
        
        self.call_count += 1
        
        prompt = f"""Analyze this document and extract ALL available information. Then provide:
    1. extracted_data: All the data you found
    2. schema: The structure/schema of the data with field types
    3. regex_patterns: Regex patterns to extract each field from similar documents

    Return JSON with these 3 sections:

    Document text:
    {text}

    Output format:
    {{
    "extracted_data": {{ "field1": "value1", ... }},
    "schema": {{ "field1": "string", "field2": "number", ... }},
    "regex_patterns": {{ "field1": "regex_pattern1", ... }}
    }}"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            print("="*80)
            print("This is result JSON :- ")
            print(result)
            print("="*80)
            
            tokens_used = len(prompt) // 4 + len(response.text) // 4
            self.total_tokens += tokens_used
            logger.info(f"AI Call - Tokens used: {tokens_used}, Total tokens: {self.total_tokens}, Estimated cost: ${self.get_estimated_cost():.6f}")
            # Extract the three components
            extracted_data = result.get('extracted_data', {})
            dynamic_schema = result.get('schema', {})
            regex_patterns = result.get('regex_patterns', {})
            
            return extracted_data, dynamic_schema, regex_patterns, tokens_used
            
        except Exception as e:
            logger.error(f"AI extraction error: {e}")
            raise e
    
    # Rest of the method implementation...


class HybridDocumentParser:
    """Main document parser with hybrid pipeline"""
    
    def __init__(self, 
                 enable_ai: bool = True,
                 api_key: Optional[str] = None,
                 storage_path: str = "parser_data"):
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.signature_manager = SignatureManager(storage_path)
        self.rule_extractor = RuleBasedExtractor()
        self.ai_extractor = AIExtractor(api_key) if enable_ai else None
        
        # Initialize Docling converter
        self._setup_docling()
        
        # Schema definition
        self.output_schema = self._define_output_schema()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'rule_based': 0,
            'signature_matched': 0,
            'ai_processed': 0,
            'total_ai_calls': 0,
            'total_cost': 0.0
        }
    
    def _setup_docling(self):
        """Setup Docling document converter"""
        pipeline_options = PdfPipelineOptions(
            do_table_structure=True,
            do_ocr=True
        )
        
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    
    def _define_output_schema(self) -> Dict[str, Any]:
        """Define normalized output JSON schema"""
        return {
            "document_id": "",
            "document_type": "",
            "processing_date": "",
            "sender": {
                "name": "",
                "id": "",
                "address": "",
                "contact": ""
            },
            "recipient": {
                "name": "",
                "address": "",
                "contact": ""
            },
            "metadata": {
                "date": "",
                "reference_number": "",
                "subject": "",
                "category": ""
            },
            "financial": {
                "currency": "",
                "subtotal": 0.0,
                "tax": 0.0,
                "total": 0.0,
                "payment_method": "",
                "payment_status": ""
            },
            "line_items": [],
            "tables": [],
            "attachments": [],
            "custom_fields": {},
            "processing_info": {
                "strategy": "",
                "confidence": 0.0,
                "signature_id": "",
                "ai_calls": 0,
                "processing_time": 0.0
            }
        }
    
    def process_document(self, 
                 file_path: str, 
                 sender_id: Optional[str] = None,
                 force_ai: bool = False) -> ProcessingResult:
        """Process document with hybrid pipeline"""
        start_time = datetime.now()
        interpretation_log = []
        ai_calls = 0
        signature = None
        
        try:
            # Step 1: Extract text using Docling
            interpretation_log.append(f"[{datetime.now().isoformat()}] Starting document processing")
            text = self._extract_text(file_path)
            interpretation_log.append(f"Extracted {len(text)} characters from document")
            
            # Step 2: Try signature matching first (unless forced to use AI)
            if not force_ai:
                signature = self.signature_manager.find_matching_signature(text, sender_id)
                
            if signature and not force_ai:
                interpretation_log.append(f"Matched signature: {signature.signature_id}")
                result = self._apply_signature(text, signature)
                
                # Check if signature extraction was successful
                extracted_fields = sum(1 for v in result.get('custom_fields', {}).values() if v)
                metadata_fields = sum(1 for v in result.get('metadata', {}).values() if v)
                financial_fields = sum(1 for v in result.get('financial', {}).values() if v)
                
                total_extracted = extracted_fields + metadata_fields + financial_fields
                interpretation_log.append(f"Signature extracted {total_extracted} fields")
                
                if total_extracted >= 3:  # Signature worked well
                    strategy = ProcessingStrategy.SIGNATURE_MATCH
                    confidence = 0.9
                    interpretation_log.append("Signature extraction successful")
                else:
                    # Signature extraction failed, fall back to AI
                    interpretation_log.append("Signature extraction failed, falling back to AI")
                    if self.ai_extractor:
                        try:
                            ai_result, dynamic_schema, regex_patterns, tokens = self.ai_extractor.extract_with_dynamic_schema(text)
                            ai_calls = 1
                            doc_type = self._detect_document_type(text)
                            result = self._normalize_to_schema(ai_result, doc_type)
                            strategy = ProcessingStrategy.AI_EXTRACTION
                            confidence = 0.85
                            interpretation_log.append(f"AI extracted {len(ai_result)} fields")
                            
                            # Create signature for future use if extraction was successful
                            if confidence > 0.8 and ai_result and len(ai_result) >= 3:
                                new_sig = self.signature_manager.create_signature(
                                    text, ai_result, dynamic_schema, regex_patterns, sender_id, doc_type
                                )
                                interpretation_log.append(f"Created new signature: {new_sig.signature_id}")
                                signature = new_sig
                        except Exception as ai_error:
                            interpretation_log.append(f"AI extraction failed: {str(ai_error)}")
                            result = self._define_output_schema()
                            strategy = ProcessingStrategy.AI_EXTRACTION
                            confidence = 0.0
                    else:
                        interpretation_log.append("No AI available, returning empty result")
                        result = self._define_output_schema()
                        strategy = ProcessingStrategy.SIGNATURE_MATCH
                        confidence = 0.0
            else:
                # No signature match or forced AI - use AI directly
                doc_type = self._detect_document_type(text)
                
                if self.ai_extractor:
                    interpretation_log.append("No signature found, using AI extraction")
                    try:
                        ai_result, dynamic_schema, regex_patterns, tokens = self.ai_extractor.extract_with_dynamic_schema(text)
                        ai_calls = 1
                        
                        # Normalize AI result to schema
                        result = self._normalize_to_schema(ai_result, doc_type)
                        strategy = ProcessingStrategy.AI_EXTRACTION
                        confidence = 0.85
                        
                        interpretation_log.append(f"AI extracted {len(ai_result)} fields")
                        
                        # Create signature for future use if extraction was successful
                        if confidence > 0.8 and ai_result and len(ai_result) >= 3:
                            new_sig = self.signature_manager.create_signature(
                                text, ai_result, dynamic_schema, regex_patterns, sender_id, doc_type
                            )
                            interpretation_log.append(f"Created new signature: {new_sig.signature_id}")
                            signature = new_sig
                        else:
                            interpretation_log.append("AI extraction insufficient for signature creation")
                            
                    except Exception as ai_error:
                        # AI extraction failed
                        interpretation_log.append(f"AI extraction failed: {str(ai_error)}")
                        result = self._define_output_schema()
                        strategy = ProcessingStrategy.AI_EXTRACTION
                        confidence = 0.0
                else:
                    # No AI available
                    interpretation_log.append("No signature match and no AI available")
                    result = self._define_output_schema()
                    strategy = ProcessingStrategy.RULE_BASED
                    confidence = 0.0
            
            # Add tables to result
            tables = self.rule_extractor.extract_tables(text)
            result['tables'] = tables
            
            # Add processing info
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_info'] = {
                'strategy': strategy.value,
                'confidence': confidence,
                'signature_id': signature.signature_id if signature else None,
                'ai_calls': ai_calls,
                'processing_time': processing_time
            }
            
            # Update stats
            self._update_stats(strategy, ai_calls)
            
            interpretation_log.append(f"Processing completed in {processing_time:.2f}s with strategy: {strategy.value}")
            final_cost = self.ai_extractor.get_estimated_cost() if self.ai_extractor else 0.0
            
            return ProcessingResult(
                success=True,
                normalized_json=result,
                processing_strategy=strategy,
                confidence_score=confidence,
                interpretation_log=interpretation_log,
                processing_time=processing_time,
                ai_calls_used=ai_calls,
                estimated_cost=final_cost,
                signature_matched=signature.signature_id if signature else None
            )
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            interpretation_log.append(f"Error: {str(e)}")
            
            return ProcessingResult(
                success=False,
                normalized_json=self._define_output_schema(),
                processing_strategy=ProcessingStrategy.AI_EXTRACTION,
                confidence_score=0.0,
                interpretation_log=interpretation_log,
                processing_time=(datetime.now() - start_time).total_seconds(),
                ai_calls_used=ai_calls,
                estimated_cost=0.0,
                errors=[str(e)]
            )
    
    def _extract_text(self, file_path: str) -> str:
        """Extract text from document using Docling"""
        result = self.converter.convert(file_path)
        return result.document.export_to_markdown()
    
    def _detect_document_type(self, text: str) -> DocumentType:
        """Detect document type from text"""
        text_lower = text.lower()
        
        type_keywords = {
            DocumentType.INVOICE: ['invoice', 'bill', 'payment due', 'subtotal'],
            DocumentType.RECEIPT: ['receipt', 'paid', 'transaction', 'payment received'],
            DocumentType.CONTRACT: ['agreement', 'contract', 'terms', 'parties', 'whereas'],
            DocumentType.FORM: ['form', 'application', 'fill', 'submit'],
            DocumentType.REPORT: ['report', 'summary', 'analysis', 'findings'],
            DocumentType.EMAIL: ['from:', 'to:', 'subject:', 'dear', 'regards'],
            DocumentType.LETTER: ['dear', 'sincerely', 'yours', 'regards']
        }
        
        scores = {}
        for doc_type, keywords in type_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[doc_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return DocumentType.UNKNOWN
    
    def _apply_signature(self, text: str, signature: DocumentSignature) -> Dict[str, Any]:
        logger.info(f"SIGNATURE DEBUG: Using signature {signature.signature_id}")
        
        result = self._define_output_schema()
        result['document_type'] = signature.document_type.value
        
        # Store extracted data for normalization
        extracted_data = {}  # Add this line
        extracted_count = 0
        
        patterns_to_use = signature.field_regex_patterns or signature.extraction_rules
        logger.info(f"SIGNATURE DEBUG: Using {len(patterns_to_use)} patterns")
        
        for field, pattern in patterns_to_use.items():
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match and match.group(1).strip():
                    extracted_data[field] = match.group(1).strip()  # Store in extracted_data
                    self._set_nested_field(result, field, match.group(1).strip())
                    extracted_count += 1
                    logger.info(f"SIGNATURE DEBUG: Extracted {field}: {match.group(1).strip()[:50]}")
            except Exception as e:
                logger.warning(f"Failed to apply rule for field {field}: {e}")
        
        logger.info(f"SIGNATURE DEBUG: Extracted {extracted_count} fields using signature")
        
        # Pass extracted data to normalization
        normalized_result = self._normalize_to_schema(extracted_data, signature.document_type)
        
        # Preserve the tables from the original result
        normalized_result['tables'] = result.get('tables', [])
        
        return self._normalize_to_schema(extracted_data, signature.document_type)  # Return normalized result instead of result
    

    def _normalize_signature_data(self, extracted_data: Dict[str, Any], signature: DocumentSignature) -> Dict[str, Any]:
        """Normalize signature-extracted data using the signature's schema"""
        result = self._define_output_schema()
        result['document_type'] = signature.document_type.value
        result['processing_date'] = datetime.now().isoformat()
        
        # Use the signature's sample_data as a guide for mapping
        if signature.sample_data:
            for field, value in extracted_data.items():
                if field in signature.sample_data:
                    # Map based on the original AI extraction structure
                    self._map_field_intelligently(result, field, value)
                else:
                    result['custom_fields'][field] = value
        else:
            # Fallback to regular normalization
            return self._normalize_to_schema(extracted_data, signature.document_type)
        
        return self._normalize_to_schema(extracted_data, signature.document_type)

    def _map_field_intelligently(self, result: Dict[str, Any], field: str, value: str):
        """Intelligently map fields based on field name patterns"""
        field_lower = field.lower()
        
        # Invoice/document numbers
        if any(x in field_lower for x in ['invoice_no', 'ticket_number', 'document_number']):
            result['metadata']['reference_number'] = value
        # Dates
        elif any(x in field_lower for x in ['date', 'issue_date']):
            result['metadata']['date'] = value
        # Company/sender information
        elif any(x in field_lower for x in ['company_name', 'supplier_name', 'vendor_name']):
            result['sender']['name'] = value
        elif any(x in field_lower for x in ['company_phone', 'supplier_phone']):
            result['sender']['contact'] = value
        # Sponsor/recipient information  
        elif any(x in field_lower for x in ['sponsor_name', 'customer_name', 'recipient_name']):
            result['recipient']['name'] = value
        elif any(x in field_lower for x in ['sponsor_phone', 'customer_phone']):
            result['recipient']['contact'] = value
        elif any(x in field_lower for x in ['sponsor_email', 'customer_email']):
            result['recipient']['contact'] = f"{result['recipient'].get('contact', '')} {value}".strip()
        # Currency
        elif 'currency' in field_lower:
            result['financial']['currency'] = value
        # Everything else goes to custom_fields
        else:
            result['custom_fields'][field] = value

    def _normalize_to_schema(self, extracted: Dict[str, Any], doc_type: DocumentType) -> Dict[str, Any]:
        """Normalize extracted data to output schema"""
        result = self._define_output_schema()
        result['document_id'] = hashlib.md5(str(extracted).encode()).hexdigest()[:12]
        result['document_type'] = doc_type.value
        result['processing_date'] = datetime.now().isoformat()
        
        # Map common fields
        field_mappings = {
    'invoice_number': 'metadata.reference_number',
    'invoice_no': 'metadata.reference_number',  # Add this
    'ticket_number': 'metadata.reference_number',  # Add this
    'document_number': 'metadata.reference_number',
    'receipt_number': 'metadata.reference_number',
    'date': 'metadata.date',
    'invoice_date': 'metadata.date',  # Add this
    'issue_date': 'metadata.date',  # Add this
    'document_date': 'metadata.date',
    'total_amount': 'financial.total',
    'total': 'financial.total',
    'vendor_name': 'sender.name',
    'company_name': 'sender.name',  # Add this
    'supplier_name': 'sender.name',
    'customer_name': 'recipient.name',
    'sponsor_name': 'recipient.name',  # Add this
    'currency': 'financial.currency',
    'invoice_currency': 'financial.currency',  # Add this
    'pnr': 'metadata.reference_number',
    'company_phone': 'sender.contact',  # Add this
    'sponsor_phone': 'recipient.contact',  # Add this
    'sponsor_email': 'recipient.contact',  # Add this
}
        
        for source, target in field_mappings.items():
            if source in extracted:
                self._set_nested_field(result, target, extracted[source])
        
        # Add unmapped fields to custom_fields
        for key, value in extracted.items():
            if key not in field_mappings:
                result['custom_fields'][key] = value
        
        return result
    
    def _set_nested_field(self, obj: Dict, path: str, value: Any):
        """Set nested field in dictionary"""
        parts = path.split('.')
        for part in parts[:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
        obj[parts[-1]] = value
    
    def _get_extraction_schema(self, doc_type: DocumentType) -> Dict[str, Any]:
        """Get extraction schema for document type"""
        schemas = {
            DocumentType.INVOICE: {
                "invoice_number": "string",
                "date": "string",
                "vendor_name": "string",
                "customer_name": "string",
                "total_amount": "number",
                "tax_amount": "number",
                "items": "array"
            },
            DocumentType.RECEIPT: {
                "receipt_number": "string",
                "date": "string",
                "amount": "number",
                "payment_method": "string"
            }
        }
        return schemas.get(doc_type, schemas[DocumentType.INVOICE])
    
    def _update_stats(self, strategy: ProcessingStrategy, ai_calls: int):
        """Update processing statistics"""
        self.stats['total_processed'] += 1
        
        if strategy == ProcessingStrategy.RULE_BASED:
            self.stats['rule_based'] += 1
        elif strategy == ProcessingStrategy.SIGNATURE_MATCH:
            self.stats['signature_matched'] += 1
        elif strategy == ProcessingStrategy.AI_EXTRACTION:
            self.stats['ai_processed'] += 1
        elif strategy == ProcessingStrategy.HYBRID:
            self.stats['ai_processed'] += 1  # Count hybrid as AI for cost tracking
        
        self.stats['total_ai_calls'] += ai_calls
        
        if self.ai_extractor:
            self.stats['total_cost'] = self.ai_extractor.get_estimated_cost()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()
    
    def export_signatures(self, output_path: str):
        """Export signatures for backup/sharing"""
        with open(output_path, 'w') as f:
            signatures_data = {
                sig_id: asdict(sig) 
                for sig_id, sig in self.signature_manager.signatures.items()
            }
            json.dump(signatures_data, f, indent=2)
    
    def import_signatures(self, input_path: str):
        """Import signatures from file"""
        with open(input_path, 'r') as f:
            signatures_data = json.load(f)
            
        for sig_id, sig_data in signatures_data.items():
            signature = DocumentSignature(**sig_data)
            self.signature_manager.signatures[sig_id] = signature
            
            if signature.sender_id:
                if signature.sender_id not in self.signature_manager.sender_signatures:
                    self.signature_manager.sender_signatures[signature.sender_id] = []
                self.signature_manager.sender_signatures[signature.sender_id].append(sig_id)
            else:
                self.signature_manager.global_signatures.append(sig_id)
        
        self.signature_manager.save_signatures()
        logger.info(f"Imported {len(signatures_data)} signatures")