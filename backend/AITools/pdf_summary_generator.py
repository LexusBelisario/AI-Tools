import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List, Dict
import numpy as np

def generate_model_summary_page(
    pp,
    model_type: str,
    metrics: Dict[str, float],
    features: List[str],
    importance_values: np.ndarray,
    target_variable: str,
    n_samples: int,
    accent_color: str = "#1e88e5"
):
    """
    Generate a comprehensive summary page for the PDF report.
    
    Args:
        pp: PdfPages object
        model_type: "Linear Regression", "Random Forest", or "XGBoost"
        metrics: Dictionary containing R², RMSE, MAE, MSE
        features: List of feature names
        importance_values: Array of feature importance/coefficient values
        target_variable: Name of the dependent variable
        n_samples: Number of samples used in training
        accent_color: Color for styling
    """
    fig = plt.figure(figsize=(8.5, 11))  # Letter size
    fig.suptitle(f"{model_type} Model Summary", fontsize=16, fontweight='bold', color=accent_color)
    
    # Create text content
    y_position = 0.92
    line_height = 0.04
    
    # Helper function to add text
    def add_text(text, bold=False, indent=0, color='black'):
        nonlocal y_position
        weight = 'bold' if bold else 'normal'
        fig.text(0.1 + indent, y_position, text, 
                fontsize=10, weight=weight, color=color,
                ha='left', va='top', wrap=True)
        y_position -= line_height
    
    # ========== SECTION 1: MODEL PERFORMANCE ==========
    add_text("MODEL PERFORMANCE INTERPRETATION", bold=True, color=accent_color)
    y_position -= 0.01
    
    r2 = metrics.get('R²', 0)
    rmse = metrics.get('RMSE', 0)
    mae = metrics.get('MAE', 0)
    
    # R² interpretation
    if r2 >= 0.9:
        r2_quality = "excellent"
        r2_desc = "The model explains over 90% of the variance in the data, indicating very strong predictive power."
    elif r2 >= 0.7:
        r2_quality = "good"
        r2_desc = "The model explains a substantial portion of the variance, showing solid predictive capability."
    elif r2 >= 0.5:
        r2_quality = "moderate"
        r2_desc = "The model shows moderate predictive power. Consider adding more relevant features or exploring non-linear relationships."
    else:
        r2_quality = "weak"
        r2_desc = "The model has limited predictive power. The selected features may not adequately explain the target variable."
    
    add_text(f"• R² Score ({r2:.3f}): {r2_quality.upper()}", indent=0.02)
    add_text(f"  {r2_desc}", indent=0.04)
    y_position -= 0.01
    
    # Error metrics interpretation
    add_text(f"• Prediction Errors:", indent=0.02)
    add_text(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f}", indent=0.04)
    add_text(f"  On average, predictions deviate by approximately {mae:.2f} units from actual values.", indent=0.04)
    add_text(f"  RMSE is {'significantly higher' if rmse > mae * 1.5 else 'similar'} than MAE, indicating {'the presence of large outliers' if rmse > mae * 1.5 else 'relatively consistent error distribution'}.", indent=0.04)
    
    y_position -= 0.02
    
    # ========== SECTION 2: FEATURE IMPORTANCE ==========
    add_text("FEATURE IMPORTANCE ANALYSIS", bold=True, color=accent_color)
    y_position -= 0.01
    
    # Get top 3 features
    sorted_indices = np.argsort(np.abs(importance_values))[::-1]
    top_features = [features[i] for i in sorted_indices[:3]]
    top_values = [importance_values[i] for i in sorted_indices[:3]]
    
    add_text(f"The model identified {len(features)} features as predictors for '{target_variable}'.", indent=0.02)
    add_text(f"Training was performed on {n_samples:,} valid samples.", indent=0.02)
    y_position -= 0.01
    
    add_text("Top 3 Most Influential Features:", indent=0.02, bold=True)
    for i, (feat, val) in enumerate(zip(top_features, top_values), 1):
        add_text(f"{i}. {feat} (importance: {abs(val):.4f})", indent=0.04)
    
    y_position -= 0.01
    
    # Model-specific insights
    if model_type == "Linear Regression":
        add_text("Linear Regression Insights:", indent=0.02, bold=True)
        add_text("• This model assumes a linear relationship between features and the target.", indent=0.04)
        add_text("• Coefficient signs indicate direction: positive coefficients increase the target,", indent=0.04)
        add_text("  while negative coefficients decrease it.", indent=0.04)
        add_text("• The model is interpretable but may underperform with complex non-linear patterns.", indent=0.04)
    
    elif model_type == "Random Forest":
        add_text("Random Forest Insights:", indent=0.02, bold=True)
        add_text("• This ensemble model can capture complex non-linear relationships and interactions.", indent=0.04)
        add_text("• Feature importance reflects how much each feature reduces prediction error.", indent=0.04)
        add_text("• The model is robust to outliers and missing values, but less interpretable than", indent=0.04)
        add_text("  linear models.", indent=0.04)
    
    elif model_type == "XGBoost":
        add_text("XGBoost Insights:", indent=0.02, bold=True)
        add_text("• This gradient boosting model excels at capturing complex patterns through", indent=0.04)
        add_text("  iterative refinement.", indent=0.04)
        add_text("• Feature importance shows contribution to splitting decisions across all trees.", indent=0.04)
        add_text("• The model offers strong predictive power but requires careful tuning to avoid", indent=0.04)
        add_text("  overfitting.", indent=0.04)
    
    y_position -= 0.02
    
    # ========== SECTION 3: RESIDUAL ANALYSIS ==========
    add_text("RESIDUAL ANALYSIS", bold=True, color=accent_color)
    y_position -= 0.01
    
    add_text("Residuals represent the difference between actual and predicted values.", indent=0.02)
    y_position -= 0.01
    
    add_text("What to look for:", indent=0.02, bold=True)
    add_text("• Residual Distribution: Should be approximately normal (bell-curved) and centered at 0.", indent=0.04)
    add_text("  Skewed distributions suggest the model may be biased.", indent=0.04)
    y_position -= 0.005
    add_text("• Residuals vs Predicted Plot: Should show random scatter around the zero line.", indent=0.04)
    add_text("  Patterns indicate the model is missing important information.", indent=0.04)
    
    y_position -= 0.02
    
    # ========== SECTION 4: RECOMMENDATIONS ==========
    add_text("RECOMMENDATIONS FOR IMPROVEMENT", bold=True, color=accent_color)
    y_position -= 0.01
    
    # Dynamic recommendations based on performance
    if r2 < 0.7:
        add_text("Based on the moderate performance metrics:", indent=0.02, bold=True)
        add_text("• Consider adding additional relevant features that may explain more variance.", indent=0.04)
        add_text("• Examine feature distributions - transformations (log, sqrt) may help with skewed data.", indent=0.04)
        add_text("• Check for outliers in the data that may be affecting model performance.", indent=0.04)
        if model_type == "Linear Regression":
            add_text("• Try Random Forest or XGBoost to capture potential non-linear relationships.", indent=0.04)
    else:
        add_text("The model shows strong performance. To maintain quality:", indent=0.02, bold=True)
        add_text("• Validate predictions on new, unseen data to ensure generalization.", indent=0.04)
        add_text("• Monitor for data drift - model may need retraining if input patterns change.", indent=0.04)
        add_text("• Document the model version and training data for reproducibility.", indent=0.04)
    
    y_position -= 0.01
    add_text("General best practices:", indent=0.02, bold=True)
    add_text("• Always validate on held-out test data before deployment.", indent=0.04)
    add_text("• Review the 'Actual vs Predicted' plot - points should cluster near the diagonal line.", indent=0.04)
    add_text("• Consider domain expertise - statistically significant features should make practical sense.", indent=0.04)
    
    y_position -= 0.02
    
    # ========== FOOTER ==========
    fig.text(0.5, 0.05, 
            f"Model Type: {model_type} | Training Samples: {n_samples:,} | R²: {r2:.3f} | RMSE: {rmse:.2f}",
            ha='center', fontsize=8, style='italic', color='gray')
    
    # Add box around the entire page
    rect = mpatches.Rectangle((0.08, 0.03), 0.84, 0.94, 
                              linewidth=2, edgecolor=accent_color, 
                              facecolor='none', transform=fig.transFigure)
    fig.patches.append(rect)
    
    pp.savefig(fig, facecolor='white')
    plt.close(fig)
    print("   ✅ Added comprehensive summary page to PDF report")