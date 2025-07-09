from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np
import pandas as pd

# Add this function before the train_survival endpoint
def calculate_business_metrics(cph, df, time_col, event_col):
    """Calculate business-friendly metrics for survival analysis"""
    try:
        # Get predictions
        partial_hazard = cph.predict_partial_hazard(df)
        survival_func = cph.predict_survival_function(df)
        
        # Calculate concordance index (C-index)
        c_index = concordance_index(df[time_col], -partial_hazard, df[event_col])
        
        # Calculate median survival times for each individual
        median_survival_times = []
        for idx in survival_func.columns:
            surv_func = survival_func[idx]
            # Find median survival time (where survival probability = 0.5)
            median_idx = np.where(surv_func <= 0.5)[0]
            if len(median_idx) > 0:
                median_survival_times.append(surv_func.index[median_idx[0]])
            else:
                # If survival never drops to 0.5, use the last time point
                median_survival_times.append(surv_func.index[-1])
        
        median_survival_times = np.array(median_survival_times)
        
        # Risk stratification: classify patients into risk groups based on partial hazard
        risk_scores = partial_hazard.values
        high_risk_threshold = np.percentile(risk_scores, 75)
        low_risk_threshold = np.percentile(risk_scores, 25)
        
        high_risk_mask = risk_scores >= high_risk_threshold
        low_risk_mask = risk_scores <= low_risk_threshold
        
        # Calculate actual event rates for each risk group
        high_risk_event_rate = df[event_col][high_risk_mask].mean() if high_risk_mask.sum() > 0 else 0
        low_risk_event_rate = df[event_col][low_risk_mask].mean() if low_risk_mask.sum() > 0 else 0
        
        # Risk stratification performance
        risk_separation = high_risk_event_rate - low_risk_event_rate
        
        # Time prediction accuracy (for censored data, this is approximate)
        observed_times = df[time_col].values
        time_prediction_mae = np.mean(np.abs(median_survival_times - observed_times))
        time_prediction_accuracy = 1 - (time_prediction_mae / np.mean(observed_times))
        time_prediction_accuracy = max(0, min(1, time_prediction_accuracy))  # Clamp between 0 and 1
        
        # High-risk identification accuracy
        actual_events = df[event_col].values
        predicted_high_risk = (risk_scores >= high_risk_threshold).astype(int)
        
        # Calculate precision and recall for high-risk identification
        if predicted_high_risk.sum() > 0:
            high_risk_precision = (actual_events * predicted_high_risk).sum() / predicted_high_risk.sum()
        else:
            high_risk_precision = 0
            
        if actual_events.sum() > 0:
            high_risk_recall = (actual_events * predicted_high_risk).sum() / actual_events.sum()
        else:
            high_risk_recall = 0
        
        # Overall model reliability (combination of C-index and other factors)
        overall_reliability = (c_index + time_prediction_accuracy + (risk_separation + 1) / 2) / 3
        
        return {
            "concordance_index": float(c_index),
            "survival_accuracy": float(c_index),  # C-index is the standard survival accuracy measure
            "risk_stratification": float(risk_separation),
            "time_prediction_accuracy": float(time_prediction_accuracy),
            "high_risk_identification": float((high_risk_precision + high_risk_recall) / 2) if (high_risk_precision + high_risk_recall) > 0 else 0.0,
            "overall_reliability": float(overall_reliability),
            "evaluation_threshold": float(high_risk_threshold),
            "median_predicted_survival": float(np.median(median_survival_times)),
            "high_risk_event_rate": float(high_risk_event_rate),
            "low_risk_event_rate": float(low_risk_event_rate)
        }
        
    except Exception as e:
        print(f"Error calculating business metrics: {str(e)}")
        return {
            "concordance_index": 0.5,
            "survival_accuracy": 0.5,
            "risk_stratification": 0.0,
            "time_prediction_accuracy": 0.0,
            "high_risk_identification": 0.0,
            "overall_reliability": 0.0,
            "evaluation_threshold": 0.0,
            "median_predicted_survival": 0.0,
            "high_risk_event_rate": 0.0,
            "low_risk_event_rate": 0.0
        }
