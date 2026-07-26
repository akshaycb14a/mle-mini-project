"""
edge_inference.py
Example of integrating the predictor.
"""
from src.inference.predictor import EdgePredictor

predictor = EdgePredictor()

def run_inference(features: dict):
    result = predictor.predict(features)

    print("="*40)
    print("EDGE AI PREDICTION")
    print("="*40)
    print(f"Prediction : {result['prediction']}")
    print(f"Confidence : {result['confidence']*100:.2f}%")
    print("Probabilities:")
    for cls,p in zip(["critical","normal","warning"], result["probabilities"]):
        print(f"  {cls:10s}: {p:.4f}")
    print("="*40)
    return result

if __name__ == "__main__":
    dummy = {
        "temp_mean":5.3,
        "temp_std":0.4,
        "temp_rate":0.1,
        "vibration_rms":0.2,
        "vibration_peak":0.5,
        "vibration_kurtosis":3.1,
    }
    run_inference(dummy)
