from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app) # Allows your MERN stack to connect to this local API

# 1. Load the artifacts you just moved
try:
    model = joblib.load('smartlogix_maintenance_model.pkl')
    le_v = joblib.load('le_vehicle.pkl')
    le_r = joblib.load('le_route.pkl')
    le_target = joblib.load('le_target.pkl')
    print("AI Bridge: All models and encoders loaded successfully.")
except Exception as e:
    print(f"Error loading artifacts: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # 2. Convert incoming JSON into a format the model understands
        input_df = pd.DataFrame([{
            'Vehicle_Type_Enc': le_v.transform([data['vehicle_type']])[0],
            'Usage_Hours': float(data['usage_hours']),
            'Route_Info_Enc': le_r.transform([data['route_info']])[0],
            'Load_Intensity': float(data['actual_load']) / float(data['load_capacity']),
            'Days_Since_Last_Service': int(data['days_since_service'])
        }])
        
        # 3. Generate the Prediction
        prediction_idx = model.predict(input_df)[0]
        result = le_target.inverse_transform([prediction_idx])[0]
        
        return jsonify({'maintenance_prediction': result, 'status': 'success'})
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'failed'}), 400

if __name__ == '__main__':
    # Running locally on port 5004
    app.run(host='0.0.0.0', port=5004, debug=True)