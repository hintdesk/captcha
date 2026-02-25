import json
from pathlib import Path
import pandas as pd

def compare_trainings():
    """Compare training logs from different training runs."""
    log_dir = Path("./finetuned")
    log_files = sorted(log_dir.glob("training_log_*.json"))
    
    if not log_files:
        print("No training logs found in ./finetuned/")
        return
    
    print(f"\nFound {len(log_files)} training log(s)\n")
    
    results = []
    
    for log_file in log_files:
        with open(log_file, "r") as f:
            data = json.load(f)
        
        # Extract key information
        batch_size = data["batch_size"]
        num_epochs = data["num_train_epochs"]
        learning_rate = data["learning_rate"]
        workers = data["dataloader_num_workers"]
        timestamp = data["timestamp"]
        
      
        # Get final metrics
        final_loss = None
        final_eval_loss = None
        epochs_completed = 0
        
        total_time = data.get("total_time_seconds", 0)
        
        results.append({
            "Timestamp": timestamp,
            "Batch Size": batch_size,
            "Epochs": num_epochs,
            "Learning Rate": learning_rate,
            "Workers": workers,
            "Epochs Completed": epochs_completed,
            "Final Loss": f"{final_loss:.4f}" if final_loss else "N/A",
            "Final Eval Loss": f"{final_eval_loss:.4f}" if final_eval_loss else "N/A",
            "Total Time (s)": f"{total_time:.2f}" if total_time else "N/A",
        })
    
    # Display comparison
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("\n")
    
    # Show detailed logs for each training
    for log_file in log_files:
        with open(log_file, "r") as f:
            data = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"Details for training: {data['timestamp']}")
        print(f"Batch Size: {data['batch_size']}, Workers: {data['dataloader_num_workers']}")
        print(f"{'='*80}")
        
        # Create epoch summary
        epoch_logs = []
        for entry in data["log_history"]:
            if "epoch" in entry:
                epoch_logs.append({
                    "Epoch": entry["epoch"],
                    "Step": int(entry.get("step", 0)),
                    "Loss": f"{entry.get('loss', 'N/A'):.4f}" if "loss" in entry else "N/A",
                    "Eval Loss": f"{entry.get('eval_loss', 'N/A'):.4f}" if "eval_loss" in entry else "N/A",
                    "Learning Rate": f"{entry.get('learning_rate', 0):.2e}",
                })
        
        if epoch_logs:
            epoch_df = pd.DataFrame(epoch_logs)
            print(epoch_df.to_string(index=False))

if __name__ == "__main__":
    compare_trainings()
