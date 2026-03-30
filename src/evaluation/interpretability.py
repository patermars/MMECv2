import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def explain_audio_features(model, audio_features: np.ndarray,
                           feature_names: list[str] = None,
                           output_path: str = "shap_summary.png"):
    try:
        import shap
        explainer = shap.DeepExplainer(model, audio_features[:100])
        shap_vals = explainer.shap_values(audio_features)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals, audio_features,
                         feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"SHAP summary saved to {output_path}")
    except Exception as e:
        print(f"SHAP analysis failed: {e}")


def plot_prediction_scatter(preds: np.ndarray, targets: np.ndarray,
                            output_path: str = "prediction_scatter.png"):
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5, s=20)
    plt.xlabel("Actual Abnormal Volatility")
    plt.ylabel("Predicted Abnormal Volatility")
    plt.title("Prediction vs Actual")

    min_val = min(targets.min(), preds.min())
    max_val = max(targets.max(), preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_section_importance(section_attention_weights: dict,
                            output_path: str = "section_importance.png"):
    sections = list(section_attention_weights.keys())
    weights = list(section_attention_weights.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sections, y=weights)
    plt.xlabel("Section")
    plt.ylabel("Average Attention Weight")
    plt.title("Section Importance in Volatility Prediction")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ablation_comparison(results_df, output_path: str = "ablation_results.png"):
    plt.figure(figsize=(14, 8))
    results_sorted = results_df.sort_values("spearman_rho", ascending=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(results_sorted)))
    plt.barh(range(len(results_sorted)), results_sorted["spearman_rho"],
             color=colors)
    plt.yticks(range(len(results_sorted)),
               [f"{idx}: {row.get('notes', '')}"
                for idx, row in results_sorted.iterrows()])
    plt.xlabel("Spearman Correlation (rho)")
    plt.title("Ablation Study Results")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
