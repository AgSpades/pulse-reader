# Turbofan RUL Prediction API

A FastAPI application with an intuitive web interface for predicting the Remaining Useful Life (RUL) of aircraft turbofan engines using Machine Learning.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```

Or use uvicorn directly:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 3. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## 📊 Features

### Web Interface

- **Intuitive UI**: Modern, responsive design with gradient themes
- **Real-time Prediction**: Instant RUL prediction based on sensor inputs
- **Example Data**: Pre-loaded examples for healthy and degraded engines
- **Visual Feedback**:
  - Health status indicators (Excellent, Good, Fair, Poor, Critical)
  - Risk assessment with color-coded progress bars
  - Maintenance recommendations based on RUL
  - Confidence levels for each prediction

### API Endpoints

#### `POST /predict`

Predict RUL based on sensor data

**Request Body:**

```json
{
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
```

**Response:**

```json
{
  "rul_prediction": 98.5,
  "health_status": "Excellent",
  "confidence_level": "High",
  "maintenance_recommendation": "Continue normal operations. Schedule routine maintenance as planned.",
  "risk_level": "Low"
}
```

#### `GET /health`

Check API health status

#### `GET /model-info`

Get information about the trained model

#### `GET /example-data`

Get example sensor data for testing

## 🧠 About the Model

- **Type**: Multi-Layer Perceptron (Neural Network)
- **Architecture**: 3 hidden layers (100-50-25 neurons)
- **Training Dataset**: NASA C-MAPSS Turbofan Engine Degradation Dataset
- **Performance**:
  - Test RMSE: ~20-25 cycles
  - R² Score: ~0.75-0.85
- **Input**: 14 sensor readings (temperature, pressure, speed, ratios)
- **Output**: Remaining Useful Life in operating cycles

## 📡 Sensor Inputs

The model uses 14 critical sensors:

- **s_2**: Total Temperature at fan inlet
- **s_3**: Total Pressure at fan inlet
- **s_4**: Total Temperature at LPC outlet
- **s_7**: Total Pressure at HPC outlet
- **s_8**: Physical fan speed
- **s_9**: Physical core speed
- **s_11**: Static pressure at HPC outlet
- **s_12**: Ratio of fuel flow to Ps30
- **s_13**: Corrected fan speed
- **s_14**: Corrected core speed
- **s_15**: Bypass Ratio
- **s_17**: Bleed Enthalpy
- **s_20**: HPT coolant bleed
- **s_21**: LPT coolant bleed

## 🎯 Health Status Categories

| RUL Range    | Health Status | Risk Level | Recommendation                         |
| ------------ | ------------- | ---------- | -------------------------------------- |
| > 80 cycles  | Excellent     | Low        | Continue normal operations             |
| 50-80 cycles | Good          | Low-Medium | Plan maintenance within 30-50 cycles   |
| 30-50 cycles | Fair          | Medium     | Schedule maintenance soon              |
| 15-30 cycles | Poor          | High       | Schedule immediate maintenance         |
| < 15 cycles  | Critical      | Very High  | URGENT: Immediate maintenance required |

## 🔧 Project Structure

```
pulse-reader/
├── app.py                 # FastAPI application
├── static/
│   └── index.html        # Web interface
├── model/
│   └── mlp_model.pkl     # Trained Neural Network model
├── requirements.txt       # Python dependencies
└── README_APP.md         # This file
```

## 🛠️ Development

### Running in Development Mode

```bash
uvicorn app:app --reload
```

### Testing the API

Use the interactive API documentation at `http://localhost:8000/docs` or use curl:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "s_2": 641.82, "s_3": 1589.7, "s_4": 1400.6, "s_7": 554.36,
       "s_8": 2388.06, "s_9": 9046.19, "s_11": 47.47, "s_12": 521.66,
       "s_13": 2388.02, "s_14": 8138.62, "s_15": 8.4195, "s_17": 392,
       "s_20": 39.06, "s_21": 23.419
     }'
```

## 📚 Use Cases

- **Predictive Maintenance**: Schedule maintenance before failures occur
- **Fleet Management**: Monitor multiple engines across aircraft fleet
- **Cost Optimization**: Reduce unplanned downtime and maintenance costs
- **Safety Enhancement**: Prevent catastrophic failures through early detection
- **Operations Planning**: Optimize flight schedules based on engine health

## 🔬 Technical Details

The application uses:

- **FastAPI**: Modern, fast web framework for building APIs
- **Pydantic**: Data validation using Python type annotations
- **Scikit-learn**: Machine learning preprocessing (scaling, polynomial features)
- **Joblib**: Model serialization and loading
- **Vanilla JS**: Frontend interactivity without heavy frameworks

## 📝 License

This project is built for educational and research purposes using the NASA C-MAPSS dataset.

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

---

**Built with ❤️ for predictive maintenance and aviation safety**
