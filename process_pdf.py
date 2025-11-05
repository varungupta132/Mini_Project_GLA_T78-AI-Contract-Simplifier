from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np, warnings, logging, contextlib, io
from pdf2image import convert_from_path
import pytesseract
from tqdm import tqdm

# ============================
# 1️⃣ LOAD TRAINED MODELS
# ============================
summarize_model = T5ForConditionalGeneration.from_pretrained("./summarizer_out")
summarize_tokenizer = T5Tokenizer.from_pretrained("./summarizer_out")

cls_model = AutoModelForSequenceClassification.from_pretrained("./classifier_out")
cls_tokenizer = AutoTokenizer.from_pretrained("./classifier_out")


# ============================
# 2️⃣ SUMMARIZATION FUNCTION
# ============================
def summarize_text(text):
    if not text.strip():
        return ""
    inputs = summarize_tokenizer(
        "summarize: " + text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    summary_ids = summarize_model.generate(
        inputs["input_ids"],
        max_length=400,  # ⬆ Increase summary length
        min_length=80,  # Ensure it doesn’t get too short
        length_penalty=1.0,
        num_beams=5,
        early_stopping=True
    )
    return summarize_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# ============================
# 3️⃣ RISK DETECTION FUNCTION
# ============================
def detect_hidden_risks(text):
    if not text.strip():
        return "No Text", 0.0
    inputs = cls_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = cls_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        label_id = np.argmax(probs)
        score = float(probs[label_id])
    label_name = cls_model.config.id2label.get(label_id, str(label_id))
    return label_name, score


# ============================
# 4️⃣ OCR CONVERSION FUNCTION
# ============================
def ocr_pdf_to_page_texts(pdf_path, dpi=300, poppler_path=None):
    try:
        pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        return []
    texts = []
    for i, page in enumerate(tqdm(pages, desc="OCR Processing"), start=1):
        try:
            text = pytesseract.image_to_string(page)
        except Exception as e:
            print(f"⚠️ OCR failed on page {i}: {e}")
            text = ""
        texts.append(text)
    return texts


# ============================
# 5️⃣ MAIN PIPELINE FUNCTION (for Flask)
# ============================
def run_full_pipeline(pdf_path, dpi=300, poppler_path=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger().setLevel(logging.ERROR)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            pages_text = ocr_pdf_to_page_texts(pdf_path, dpi=dpi, poppler_path=poppler_path)

    if not pages_text:
        meta = {
            "doc_summary": "(❌ No text extracted from PDF.)",
            "doc_risks": "No Data",
            "doc_risk_score": 0.0,
            "pages": 0
        }
        return None, meta

    # Combine all text
    full_text = '\n\n'.join(pages_text)

    # Summarize
    doc_summary = summarize_text(full_text)

    # Detect risk
    risk_label, risk_score = detect_hidden_risks(full_text)

    meta = {
        "doc_summary": doc_summary if doc_summary else "(No overall summary generated.)",
        "doc_risks": risk_label,
        "doc_risk_score": risk_score,
        "pages": len(pages_text)
    }

    return None, meta
