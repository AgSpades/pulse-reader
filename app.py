"""
PulseReader: Predictive Analysis Engine
FastAPI application for AI-powered Remaining Useful Life (RUL) prediction
Multi-sensor health monitoring and predictive maintenance system
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field
import joblib
import numpy as np
from typing import List
import uvicorn
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

# Initialize FastAPI app
app = FastAPI(
    title="PulseReader API",
    description="AI-powered predictive analysis engine for remaining useful life prediction and health monitoring",
    version="1.0.0"
)

# Load the trained model and preprocessing components
try:
    model = joblib.load('model/mlp_model.pkl')
    print("✓ Neural Network model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Try to load preprocessing components
try:
    scaler = joblib.load('model/scaler.pkl')
    poly = joblib.load('model/poly.pkl')
    print("✓ Preprocessing components loaded successfully")
    try:
        feature_selector = joblib.load('model/feature_selector.pkl')
        print("✓ Feature selector loaded")
    except:
        feature_selector = None
        print("⚠ Feature selector not found, will use all polynomial features")
    preprocessors_loaded = True
except Exception as e:
    print(f"⚠ Preprocessing components not found: {e}")
    print("  Run 'python save_preprocessors.py' to generate them")
    # Fallback to creating new ones (will be fitted on the fly)
    scaler = MinMaxScaler()
    poly = PolynomialFeatures(2)
    feature_selector = None
    preprocessors_loaded = False

# Sensor names used in the model (after dropping unused sensors)
SENSOR_NAMES = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 
                's_12', 's_13', 's_14', 's_15', 's_17', 's_20', 's_21']


class SensorData(BaseModel):
    """Input data model for sensor readings"""
    s_2: float = Field(..., description="Sensor 2 reading (Total Temperature)")
    s_3: float = Field(..., description="Sensor 3 reading (Total Pressure)")
    s_4: float = Field(..., description="Sensor 4 reading (Temperature)")
    s_7: float = Field(..., description="Sensor 7 reading (Pressure)")
    s_8: float = Field(..., description="Sensor 8 reading (Physical fan speed)")
    s_9: float = Field(..., description="Sensor 9 reading (Physical core speed)")
    s_11: float = Field(..., description="Sensor 11 reading (Static pressure)")
    s_12: float = Field(..., description="Sensor 12 reading (Fuel flow ratio)")
    s_13: float = Field(..., description="Sensor 13 reading (Corrected fan speed)")
    s_14: float = Field(..., description="Sensor 14 reading (Corrected core speed)")
    s_15: float = Field(..., description="Sensor 15 reading (Bypass Ratio)")
    s_17: float = Field(..., description="Sensor 17 reading (Bleed Enthalpy)")
    s_20: float = Field(..., description="Sensor 20 reading (HPT coolant bleed)")
    s_21: float = Field(..., description="Sensor 21 reading (LPT coolant bleed)")

    class Config:
        json_schema_extra = {
            "example": {
                "s_2": 641.82,
                "s_3": 1589.7,
                "s_4": 1400.6,
                "s_7": 554.36,
                "s_8": 2388.06,
                "s_9": 9046.19,
                "s_11": 47.47,
                "s_12": 521.66,
                "s_13": 2388.02,
                "s_14": 8138.62,
                "s_15": 8.4195,
                "s_17": 392,
                "s_20": 39.06,
                "s_21": 23.419
            }
        }


class PredictionResponse(BaseModel):
    """Response model for RUL prediction"""
    rul_prediction: float = Field(..., description="Predicted Remaining Useful Life (cycles)")
    health_status: str = Field(..., description="Engine health status")
    confidence_level: str = Field(..., description="Prediction confidence level")
    maintenance_recommendation: str = Field(..., description="Maintenance recommendation")
    risk_level: str = Field(..., description="Risk level based on RUL")


def preprocess_input(sensor_data: SensorData) -> np.ndarray:
    """
    Preprocess sensor data to match model's expected input format
    """
    # Convert to numpy array in correct order
    features = np.array([[
        sensor_data.s_2, sensor_data.s_3, sensor_data.s_4, sensor_data.s_7,
        sensor_data.s_8, sensor_data.s_9, sensor_data.s_11, sensor_data.s_12,
        sensor_data.s_13, sensor_data.s_14, sensor_data.s_15, sensor_data.s_17,
        sensor_data.s_20, sensor_data.s_21
    ]])
    
    if preprocessors_loaded:
        # Use saved preprocessors
        scaled_features = scaler.transform(features)
        poly_features = poly.transform(scaled_features)
        
        if feature_selector is not None:
            # Convert to dense array if sparse
            if hasattr(poly_features, 'toarray'):
                poly_features = poly_features.toarray()
            selected_features = poly_features[:, feature_selector.get_support()]
            return selected_features
        else:
            # Use all polynomial features
            if hasattr(poly_features, 'toarray'):
                poly_features = poly_features.toarray()
            return poly_features
    else:
        # Fallback: use approximate normalization
        # These ranges are estimates from the training data
        min_vals = np.array([640, 1580, 1390, 550, 2385, 9040, 46, 520, 2385, 8130, 8.3, 388, 38, 23])
        max_vals = np.array([645, 1605, 1425, 560, 2395, 9065, 49, 530, 2400, 8165, 8.7, 402, 41, 25])
        
        # Avoid division by zero
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        scaled_features = (features - min_vals) / range_vals
        poly_features = poly.fit_transform(scaled_features)
        
        if hasattr(poly_features, 'toarray'):
            poly_features = poly_features.toarray()
        
        return poly_features
    scaled_features = scaler.fit_transform(features)
    
    # Apply polynomial features
    poly_features = poly.fit_transform(scaled_features)
    
    # Apply feature selection (simplified version - in production use the actual selector)
    # For now, we'll use all polynomial features
    
    return poly_features


def get_health_status(rul: float) -> tuple:
    """
    Determine health status, risk level, and maintenance recommendation based on RUL
    """
    if rul > 80:
        health = "Excellent"
        risk = "Low"
        recommendation = "Continue normal operations. Schedule routine maintenance as planned."
        confidence = "High"
    elif rul > 50:
        health = "Good"
        risk = "Low-Medium"
        recommendation = "Monitor closely. Plan for maintenance within next 30-50 cycles."
        confidence = "High"
    elif rul > 30:
        health = "Fair"
        risk = "Medium"
        recommendation = "Schedule maintenance soon. Increase monitoring frequency."
        confidence = "Medium"
    elif rul > 15:
        health = "Poor"
        risk = "High"
        recommendation = "⚠️ Schedule immediate maintenance. Reduce operational load if possible."
        confidence = "Medium"
    else:
        health = "Critical"
        risk = "Very High"
        recommendation = "🚨 URGENT: Immediate maintenance required. Consider grounding until serviced."
        confidence = "High"
    
    return health, risk, recommendation, confidence


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "Neural Network (MLP)",
        "api_version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(sensor_data: SensorData):
    """
    Predict Remaining Useful Life based on sensor readings
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess input data
        processed_data = preprocess_input(sensor_data)
        
        # Make prediction
        rul_prediction = model.predict(processed_data)[0]
        
        # Ensure RUL is not negative
        rul_prediction = max(0, rul_prediction)
        
        # Get health status and recommendations
        health, risk, recommendation, confidence = get_health_status(rul_prediction)
        
        return PredictionResponse(
            rul_prediction=round(rul_prediction, 2),
            health_status=health,
            confidence_level=confidence,
            maintenance_recommendation=recommendation,
            risk_level=risk
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get information about the trained model"""
    return {
        "model_type": "Multi-Layer Perceptron (Neural Network)",
        "architecture": "100-50-25 neurons (3 hidden layers)",
        "training_dataset": "NASA C-MAPSS Turbofan Engine Degradation",
        "performance_metrics": {
            "test_rmse": "~20-25 cycles",
            "r2_score": "~0.75-0.85"
        },
        "sensors_used": SENSOR_NAMES,
        "output": "Remaining Useful Life (in cycles)",
        "max_rul": 125
    }


@app.get("/example-data")
async def get_example_data():
    """Get example sensor data for testing"""
    examples = [
        {
            "name": "Healthy Engine",
            "data": {
                "s_2": 641.82, "s_3": 1589.7, "s_4": 1400.6, "s_7": 554.36,
                "s_8": 2388.06, "s_9": 9046.19, "s_11": 47.47, "s_12": 521.66,
                "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_17": 392,
                "s_20": 39.06, "s_21": 23.419
            },
            "expected_rul": "~100+ cycles"
        },
        {
            "name": "Moderate Degradation",
            "data": {
                "s_2": 642.5, "s_3": 1592.0, "s_4": 1407.0, "s_7": 555.0,
                "s_8": 2388.5, "s_9": 9050.0, "s_11": 47.5, "s_12": 523.0,
                "s_13": 2390.0, "s_14": 8140.0, "s_15": 8.5, "s_17": 395,
                "s_20": 39.5, "s_21": 23.5
            },
            "expected_rul": "~50-80 cycles"
        },
        {
            "name": "High Degradation",
            "data": {
                "s_2": 643.54, "s_3": 1601.41, "s_4": 1427.20, "s_7": 551.25,
                "s_8": 2388.32, "s_9": 9033.22, "s_11": 48.25, "s_12": 520.08,
                "s_13": 2388.32, "s_14": 8110.93, "s_15": 8.5113, "s_17": 396,
                "s_20": 38.48, "s_21": 22.9649
            },
            "expected_rul": "~10-30 cycles"
        }
    ]
    return {"examples": examples}


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Starting Turbofan RUL Prediction API Server")
    print("=" * 70)
    print("📊 Model: Neural Network (Multi-Layer Perceptron)")
    print("🌐 API Docs: http://localhost:8000/docs")
    print("🖥️  Web Interface: http://localhost:8000")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000)
