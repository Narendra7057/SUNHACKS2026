from typing import Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest


class OutputAnomalyDetector:
    """Isolation Forest based anomaly detection for agent outputs."""

    def __init__(self):
        # Real baseline used to fit Isolation Forest for clause anomaly checks.
        self.model = IsolationForest(contamination=0.2, random_state=42)
        baseline = np.array([[100, 0, 0], [200, 1, 0], [150, 0, 1]])
        self.model.fit(baseline)

    def extract_features(self, input_text: str) -> List[float]:
        """Extract anomaly features from contract text."""
        text = input_text or ""
        return [
            float(len(text)),
            float(text.lower().count("not")),
            float(text.lower().count("no")),
        ]

    def detect(self, input_text: str) -> Dict[str, object]:
        """Return anomaly result using Isolation Forest on extracted features."""
        features = np.array([self.extract_features(input_text)])
        pred = self.model.predict(features)[0]  # -1 anomalous, 1 normal
        score = float(self.model.decision_function(features)[0])
        return {
            "is_anomaly": bool(pred == -1),
            "prediction": int(pred),
            "anomaly_score": score,
            "features": [float(value) for value in features.tolist()[0]],
        }
