import React, { memo } from "react";
import "./TrainingLoader.css";

const MODEL_LABEL = { lr: "Linear Regression", rf: "Random Forest", xgb: "XGBoost" };

const ModelRow = memo(function ModelRow({ model, status }) {
  const isDone = status === "done";
  const isFailed = status === "failed";
  const isRunning = !isDone && !isFailed;

  // status is either "running", "done", "failed", or a real progress string from the backend
  const message = isDone
    ? "Training complete"
    : isFailed
    ? "Training failed"
    : status === "running"
    ? `Starting ${MODEL_LABEL[model]}...`
    : status; // real string emitted by backend

  return (
    <div className={`tl-model-row tl-model-row--${isDone ? "done" : isFailed ? "failed" : "running"}`}>
      <div className="tl-model-header">
        <span className="tl-model-name">{MODEL_LABEL[model]}</span>
        <span className={`tl-model-badge tl-model-badge--${isDone ? "done" : isFailed ? "failed" : "running"}`}>
          {isDone ? "✓ Done" : isFailed ? "✗ Failed" : "Running"}
        </span>
      </div>
      <div className="tl-model-step">{message}</div>
    </div>
  );
});

export default function TrainingLoader({ isTraining, selectedModels = [], modelStatuses = {} }) {
  if (!isTraining) return null;

  return (
    <div className="training-loader-overlay">
      <div className="training-loader-container tl-multi">
        <div className="tl-top">
          <div className="training-spinner" />
          <div className="tl-title">Training in progress</div>
        </div>
        <div className="tl-rows">
          {selectedModels.map((m) => (
            <ModelRow
              key={m}
              model={m}
              status={modelStatuses[m] || "running"}
            />
          ))}
        </div>
      </div>
    </div>
  );
}