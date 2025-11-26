import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

API_URL = "http://127.0.0.1:8000"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_test_questions(filepath="test_questions.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

def load_transcript(filepath="transcript.json"):
    with open(filepath, 'r') as f:
        data = json.load(f)
        if isinstance(data, dict) and "transcript" in data:
            return data["transcript"]
        return data

def keyword_baseline(question, segments):
    # Simple keyword matching baseline
    question_words = set(question.lower().split())
    
    best_segment = segments[0]
    best_score = 0
    
    for seg in segments:
        seg_words = set(seg['text'].lower().split())
        overlap = len(question_words & seg_words)
        
        if overlap > best_score:
            best_score = overlap
            best_segment = seg
    
    return best_segment['start'], best_score

def query_rag_system(question):
    # Query the RAG system via API
    try:
        response = requests.post(
            f"{API_URL}/query/",
            json={"query": question, "top_k": 3},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        if data['results']:
            return data['results'][0]['start']
        return 0.0
    except Exception as e:
        print(f"  RAG error: {e}")
        return None

def calculate_error(predicted, ground_truth):
    if predicted is None:
        return 999.0
    return abs(predicted - ground_truth)

def run_evaluation(test_questions, segments):
    # Run evaluation on all questions
    results = []
    
    for i, q in enumerate(test_questions):
        print(f"[{i+1}/{len(test_questions)}] {q['question'][:60]}...")
        
        # RAG system
        rag_time = query_rag_system(q['question'])
        rag_error = calculate_error(rag_time, q['ground_truth_time'])
        
        # Baseline
        baseline_time, baseline_score = keyword_baseline(q['question'], segments)
        baseline_error = calculate_error(baseline_time, q['ground_truth_time'])
        
        results.append({
            'question': q['question'],
            'category': q.get('category', 'general'),
            'ground_truth': q['ground_truth_time'],
            'rag_predicted': rag_time if rag_time is not None else 0.0,
            'rag_error': rag_error,
            'baseline_predicted': baseline_time,
            'baseline_error': baseline_error,
            'baseline_score': baseline_score
        })
        
        print(f"  GT: {q['ground_truth_time']:.1f}s | RAG: {rag_time:.1f}s (error: {rag_error:.1f}s) | Baseline: {baseline_time:.1f}s (error: {baseline_error:.1f}s)")
    
    return pd.DataFrame(results)

def print_summary(df):
    # Print summary statistics table
    print("\nSUMMARY STATISTICS\n")
    
    summary = pd.DataFrame({
        'Metric': [
            'Mean Absolute Error (s)',
            'Median Absolute Error (s)',
            'Std Dev Error (s)',
            'Accuracy within 5s (%)',
            'Accuracy within 10s (%)',
            'Accuracy within 30s (%)',
            'Perfect matches (0s error)'
        ],
        'RAG System': [
            df['rag_error'].mean(),
            df['rag_error'].median(),
            df['rag_error'].std(),
            (df['rag_error'] <= 5).mean() * 100,
            (df['rag_error'] <= 10).mean() * 100,
            (df['rag_error'] <= 30).mean() * 100,
            (df['rag_error'] == 0).sum()
        ],
        'Keyword Baseline': [
            df['baseline_error'].mean(),
            df['baseline_error'].median(),
            df['baseline_error'].std(),
            (df['baseline_error'] <= 5).mean() * 100,
            (df['baseline_error'] <= 10).mean() * 100,
            (df['baseline_error'] <= 30).mean() * 100,
            (df['baseline_error'] == 0).sum()
        ]
    })
    
    summary['Improvement'] = summary['Keyword Baseline'] - summary['RAG System']
    
    print(summary.to_string(index=False))
    print()
    return summary

def plot_results(df):
    # Create 2 plots: accuracy vs threshold and category comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy by threshold
    ax = axes[0]
    thresholds = [5, 10, 15, 20, 30, 45, 60]
    rag_acc = [(df['rag_error'] <= t).mean() * 100 for t in thresholds]
    baseline_acc = [(df['baseline_error'] <= t).mean() * 100 for t in thresholds]
    
    ax.plot(thresholds, rag_acc, marker='o', linewidth=2.5, label='RAG System', color='blue')
    ax.plot(thresholds, baseline_acc, marker='s', linewidth=2.5, label='Keyword Baseline', color='orange')
    ax.set_xlabel('Time Threshold (seconds)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Accuracy vs Time Threshold', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Category comparison
    ax = axes[1]
    if df['category'].nunique() > 1:
        category_means = df.groupby('category')[['rag_error', 'baseline_error']].mean()
        x = range(len(category_means))
        width = 0.35
        
        ax.bar([i - width/2 for i in x], category_means['rag_error'], 
               width, label='RAG System', color='blue', alpha=0.7)
        ax.bar([i + width/2 for i in x], category_means['baseline_error'], 
               width, label='Keyword Baseline', color='orange', alpha=0.7)
        
        ax.set_xlabel('Question Category', fontsize=11)
        ax.set_ylabel('Mean Absolute Error (seconds)', fontsize=11)
        ax.set_title('Performance by Category', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(category_means.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/evaluation_plots.png", dpi=300, bbox_inches='tight')
    print(f"Saved plots to {RESULTS_DIR}/evaluation_plots.png")

def show_failure_cases(df, n=3):
    # Show worst predictions for error analysis
    print("\nTOP FAILURE CASES - RAG SYSTEM\n")
    
    worst = df.nlargest(n, 'rag_error')[['question', 'ground_truth', 'rag_predicted', 'rag_error']]
    for idx, row in worst.iterrows():
        print(f"Q: {row['question']}")
        print(f"   Ground Truth: {row['ground_truth']:.1f}s")
        print(f"   Predicted: {row['rag_predicted']:.1f}s")
        print(f"   Error: {row['rag_error']:.1f}s\n")
    
    print("TOP FAILURE CASES - BASELINE\n")
    
    worst = df.nlargest(n, 'baseline_error')[['question', 'ground_truth', 'baseline_predicted', 'baseline_error']]
    for idx, row in worst.iterrows():
        print(f"Q: {row['question']}")
        print(f"   Ground Truth: {row['ground_truth']:.1f}s")
        print(f"   Predicted: {row['baseline_predicted']:.1f}s")
        print(f"   Error: {row['baseline_error']:.1f}s\n")

def main():
    print("VIDEO QA TIMESTAMP EVALUATION\n")
    
    # Check server status
    try:
        response = requests.get(f"{API_URL}/status/")
        status = response.json()
        if not status.get('ready'):
            print("ERROR: System not ready. Upload a video first.")
            return
        print(f"Server ready with {status.get('n_segments', 0)} segments\n")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {API_URL}")
        return
    
    # Load test data
    questions = load_test_questions()
    segments = load_transcript()
    
    # Run evaluation
    results_df = run_evaluation(questions, segments)
    
    # Show results
    summary = print_summary(results_df)
    plot_results(results_df)
    show_failure_cases(results_df, n=3)
    
    print(f"Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()