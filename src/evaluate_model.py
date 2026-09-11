import pandas as pd
import torch
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

MODEL_NAME = "transformersbook/pegasus-samsum"

def run_evaluation(num_samples=15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    
    # Read test split
    test_df = pd.read_csv("data/samsum-test.csv")
    eval_df = test_df.head(num_samples)
    print(f"Running evaluation on {len(eval_df)} test samples...\n")
    
    rouge = evaluate.load("rouge")
    predictions = []
    references = eval_df["summary"].tolist()
    
    for dialogue in tqdm(eval_df["dialogue"], desc="Generating summaries"):
        inputs = tokenizer(dialogue, truncation=True, padding="longest", return_tensors="pt").to(device)
        with torch.no_grad():
            summary_ids = model.generate(
                **inputs,
                min_length=10,
                max_length=60,
                num_beams=2,
                early_stopping=True
            )
        pred_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(pred_text)
        
    results = rouge.compute(predictions=predictions, references=references)
    
    print("\n" + "=" * 32)
    print("      ROUGE BENCHMARK SCORES")
    print("=" * 32)
    for metric, score in results.items():
        print(f"{metric.upper()}: {score * 100:.2f}%")
    print("=" * 32)

if __name__ == "__main__":
    run_evaluation(num_samples=15)