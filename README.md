# text-summarizer-samsum
An end-to-end abstractive text and dialogue summarizer built using Hugging Face Transformers and the SAMSum dataset
# SAMSum Dialogue Text Summarizer

An end-to-end NLP project that performs abstractive text summarization on multi-turn conversational dialogue using Transformer-based models and the SAMSum dataset.

## Features
- Abstractive summarization tailored for conversational text
- Built with Hugging Face Transformers and PyTorch
- Interactive web UI built using Streamlit
- Evaluated on standard ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)

## Dataset
The project uses the **SAMSum Corpus**, containing ~16k chat dialogues with corresponding human-annotated summaries:
- `samsum-train.csv` (Training split)
- `samsum-validation.csv` (Validation split)
- `samsum-test.csv` (Test split)

## Setup and Installation

1. **Clone the repository:**
   \`\`\`bash
   git clone https://github.com/<YOUR_USERNAME>/text-summarizer-samsum.git
   cd text-summarizer-samsum
   \`\`\`

2. **Create and activate a virtual environment:**
   \`\`\`bash
   python -m venv venv
   # On Windows:
   .\\venv\\Scripts\\activate
   # On Mac/Linux:
   source venv/bin/activate
   \`\`\`

3. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

4. **Run the application:**
   \`\`\`bash
   streamlit run app.py
   \`\`\`
