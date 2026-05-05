import React, { useState, useEffect, useMemo } from "react";
import "./TrainingLoader.css";

const TRAINING_MESSAGES = {
  lr: [
    "Fitting linear regression model...",
    "Calculating regression coefficients...",
    "Computing ordinary least squares...",
    "Analyzing residual patterns...",
    "Performing coefficient t-tests...",
    "Optimizing linear parameters...",
  ],
  rf: [
    "Building random forest ensemble...",
    "Growing decision trees...",
    "Calculating feature splits...",
    "Aggregating tree predictions...",
    "Computing out-of-bag scores...",
    "Optimizing forest parameters...",
  ],
  xgb: [
    "Training XGBoost model...",
    "Optimizing gradient boosting...",
    "Computing boosting iterations...",
    "Fine-tuning hyperparameters...",
    "Calculating feature gains...",
    "Minimizing loss function...",
  ],
  slm: [
    "Building spatial weights matrix...",
    "Detecting Queen contiguity neighbors...",
    "Row-standardizing weights...",
    "Fitting Spatial Lag Model (GM_Lag)...",
    "Estimating spatial lag coefficient ρ...",
    "Computing Moran's I on residuals...",
    "Generating spatial diagnostics...",
  ],
  hybrid: [
    "Stage 1: Building spatial weights matrix...",
    "Stage 1: Fitting Spatial Lag Model...",
    "Stage 1: Extracting SLM residuals...",
    "Stage 1: Computing Moran's I on residuals...",
    "Stage 2: Training Random Forest on residuals...",
    "Stage 2: Growing decision trees...",
    "Stage 2: Computing RF corrections...",
    "Combining SLM predictions + RF corrections...",
    "Computing hybrid diagnostics...",
  ],
  general: [
    "Loading training dataset...",
    "Preprocessing input features...",
    "Splitting data into train/test sets...",
    "Normalizing feature values...",
    "Evaluating model performance...",
    "Generating performance metrics...",
  ],
};

export default function TrainingLoader({ isTraining, selectedModels = [] }) {
  const [messageIndex, setMessageIndex] = useState(0);
  const [dots, setDots] = useState("");

  // 🆕 Use useMemo to prevent infinite loop
  const messages = useMemo(() => {
    const allMessages = [...TRAINING_MESSAGES.general];
    
    if (selectedModels.includes("lr")) {
      allMessages.push(...TRAINING_MESSAGES.lr);
    }
    if (selectedModels.includes("rf")) {
      allMessages.push(...TRAINING_MESSAGES.rf);
    }
    if (selectedModels.includes("xgb")) {
      allMessages.push(...TRAINING_MESSAGES.xgb);
    }
    if (selectedModels.includes("slm")) {
      allMessages.push(...TRAINING_MESSAGES.slm);
    }
    if (selectedModels.includes("hybrid")) {
      allMessages.push(...TRAINING_MESSAGES.hybrid);
    }

    return allMessages;
  }, [selectedModels.join(",")]); // 🆕 Convert array to string for stable dependency

  useEffect(() => {
    if (!isTraining || messages.length === 0) return;

    // Reset message index when training starts
    setMessageIndex(0);

    // Change message every 3 seconds
    const messageInterval = setInterval(() => {
      setMessageIndex((prev) => (prev + 1) % messages.length);
    }, 3000);

    // Animate dots every 500ms
    const dotsInterval = setInterval(() => {
      setDots((prev) => (prev.length >= 3 ? "" : prev + "."));
    }, 500);

    return () => {
      clearInterval(messageInterval);
      clearInterval(dotsInterval);
    };
  }, [isTraining, messages.length]); // 🆕 Only depend on length, not entire array

  if (!isTraining) return null;

  return (
    <div className="training-loader-overlay">
      <div className="training-loader-container">
        <div className="training-spinner"></div>
        <div className="training-message">
          {messages[messageIndex] || "Processing..."}
          <span className="training-dots">{dots}</span>
        </div>
        <div className="training-subtext">
          Model training in progress. Please wait while we process your data.
        </div>
      </div>
    </div>
  );
}