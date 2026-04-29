from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os
import gdown

app = Flask(__name__)
CORS(app)

# Google Drive FILE ID
MODEL_ID = "16dfzlI_4Jq5SprUkwBBXTjyrQOPyZsOi"

MODEL_PATH = "smartlogix_maintenance_model.pkl"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Download complete.")


# Load model + encoders
try:
    download_model()

    model = joblib.load(MODEL_PATH)

    le_v = joblib.load('le_vehicle.pkl')
    le_r = joblib.load('le_route.pkl')
    le_target = joblib.load('le_target.pkl')

    print("✅ Model + encoders loaded successfully")

except Exception as e:
    print(f"❌ Error loading artifacts: {e}")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = pd.DataFrame([{
            'Vehicle_Type_Enc': le_v.transform([data['vehicle_type']])[0],
            'Usage_Hours': float(data['usage_hours']),
            'Route_Info_Enc': le_r.transform([data['route_info']])[0],
            'Load_Intensity': float(data['actual_load']) / float(data['load_capacity']),
            'Days_Since_Last_Service': int(data['days_since_service'])
        }])

        prediction_idx = model.predict(input_df)[0]
        result = le_target.inverse_transform([prediction_idx])[0]

        return jsonify({
            'maintenance_prediction': result,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004)
