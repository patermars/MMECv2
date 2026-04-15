"""
Quick test script for K-fold temporal cross-validation.

Usage:
  # Quick test (3 folds, 5 epochs)
  python -m src.test_kfold

  # Full run (5 folds, 50 epochs)
  python -m src.train_kfold --k 5 --epochs 50
"""

import subprocess
import sys

if __name__ == "__main__":
    print("Running K-fold temporal cross-validation test...")
    print("Configuration: K=3 folds, 5 epochs, 1 seed per fold\n")
    
    cmd = [
        sys.executable, "-m", "src.train_kfold",
        "--k", "3",
        "--epochs", "5",
        "--seeds", "1",
        "--model_type", "cross_modal"
    ]
    
    subprocess.run(cmd)
