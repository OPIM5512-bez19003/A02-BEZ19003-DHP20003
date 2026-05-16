import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
 
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
 
def metrics(y_true, y_pred):
    r2   = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return r2, rmse
 
def make_plot(y_true, y_pred, split_label, color, save_path):
    r2, rmse = metrics(y_true, y_pred)
    residuals = y_true - y_pred
 
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
 
    # Actual vs Predicted
    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(y_true, y_pred, alpha=0.35, s=14, color=color, label="Samples")
    lo = min(y_true.min(), y_pred.min()) - 0.1
    hi = max(y_true.max(), y_pred.max()) + 0.1
    ax1.plot([lo, hi], [lo, hi], "r--", linewidth=1.8, label="Perfect fit")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Actual Price (100k USD)")
    ax1.set_ylabel("Predicted Price (100k USD)")
    ax1.set_title(f"Actual vs. Predicted — {split_label}", fontweight="bold")
    ax1.text(0.05, 0.93, f"R² = {r2:.4f}\nRMSE = {rmse:.4f}",
             transform=ax1.transAxes, fontsize=10, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
    ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.5)
 
    # Residual distribution
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(residuals, bins=60, color=color, alpha=0.75, edgecolor="white")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.6, label="Zero error")
    ax2.set_xlabel("Residual (Actual − Predicted)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Residual Distribution — {split_label}", fontweight="bold")
    ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.5)
 
    plt.suptitle("California Housing — MLPRegressor(128-64-32, early_stopping=True)",
                 fontsize=11, color="#475569")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}  |  R²={r2:.4f}  RMSE={rmse:.4f}")
 
def run():
    X_train, X_test, y_train, y_test, _ = load_and_split()
 
    scaler      = StandardScaler()
    X_train_s   = scaler.fit_transform(X_train)
    X_test_s    = scaler.transform(X_test)
 
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        alpha=0.001,
        learning_rate_init=0.005,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train_s, y_train)
    print(f"Converged in {model.n_iter_} iterations")
 
    make_plot(y_train, model.predict(X_train_s), "Train Set", "#2563EB", "plot_train.png")
    make_plot(y_test,  model.predict(X_test_s),  "Test Set",  "#16A34A", "plot_test.png")
 
if __name__ == "__main__":
    run()

plt.savefig('figs/boxplot.png') 
print("Graph saved successfully to figs/boxplot.png")