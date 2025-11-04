# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils.process_pdf import run_full_pipeline  # ✅ Your AI summarizer

# === Configuration ===
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if the uploaded file is a valid PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the upload form."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Handle the PDF upload and run AI analysis."""
    if 'pdf_file' not in request.files:
        flash("No file uploaded.")
        return redirect(url_for('index'))

    file = request.files['pdf_file']

    if file.filename == '':
        flash("Please select a file to upload.")
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        # Secure filename and save to upload folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # ✅ Run the full AI pipeline
            df, meta = run_full_pipeline(
                file_path,
                dpi=300,
                poppler_path=r"C:\Users\hp\Downloads\Release-25.07.0-0\poppler-25.07.0\Library\bin"
            )

            # Extract key metadata for results page
            summary_text = meta.get('doc_summary', '(No summary generated.)')
            risk_label = meta.get('doc_risks', 'N/A')
            risk_score = meta.get('doc_risk_score', 0.0)
            total_pages = meta.get('pages', 0)

        except Exception as e:
            summary_text = f"⚠️ An error occurred during processing: {e}"
            risk_label = 'Error'
            risk_score = 0.0
            total_pages = 0
        finally:
            # Optional cleanup (uncomment to auto-delete uploaded files)
            # os.remove(file_path)
            pass

        # ✅ Render the results page with summary and analysis
        return render_template(
            'result.html',
            summary=summary_text,
            risk_label=risk_label,
            risk_score=round(risk_score * 100, 2),
            pages=total_pages,
            filename=filename
        )

    flash("Invalid file type. Please upload a valid PDF.")
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
