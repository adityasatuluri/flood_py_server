from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn

# Load model and scaler once
model = joblib.load('best_flood_model.pkl')
scaler = joblib.load('scaler.pkl')

# Expected columns
expected_columns = [
    'Rainfall (mm)', 'Temperature (°C)', 'Humidity (%)', 
    'River Discharge (m³/s)', 'Water Level (m)', 'Elevation (m)', 
    'Population Density', 'Infrastructure', 'Historical Floods',
    'Land Cover_Agricultural', 'Land Cover_Desert', 'Land Cover_Forest', 
    'Land Cover_Urban', 'Land Cover_Water Body',
    'Soil Type_Clay', 'Soil Type_Loam', 'Soil Type_Peat', 
    'Soil Type_Sandy', 'Soil Type_Silt'
]

# Create FastAPI app
app = FastAPI()

# Request model
class InputData(BaseModel):
    Rainfall_mm: float = 0
    Temperature_C: float = 0
    Humidity_percent: float = 0
    River_Discharge_m3s: float = 0
    Water_Level_m: float = 0
    Elevation_m: float = 0
    Population_Density: float = 0
    Infrastructure: float = 0
    Historical_Floods: float = 0
    Land_Cover_Agricultural: int = 0
    Land_Cover_Desert: int = 0
    Land_Cover_Forest: int = 0
    Land_Cover_Urban: int = 0
    Land_Cover_Water_Body: int = 0
    Soil_Type_Clay: int = 0
    Soil_Type_Loam: int = 0
    Soil_Type_Peat: int = 0
    Soil_Type_Sandy: int = 0
    Soil_Type_Silt: int = 0

@app.post("/predict")
def predict(data: InputData):
    # Map input to dataframe
    input_dict = {
        'Rainfall (mm)': data.Rainfall_mm,
        'Temperature (°C)': data.Temperature_C,
        'Humidity (%)': data.Humidity_percent,
        'River Discharge (m³/s)': data.River_Discharge_m3s,
        'Water Level (m)': data.Water_Level_m,
        'Elevation (m)': data.Elevation_m,
        'Population Density': data.Population_Density,
        'Infrastructure': data.Infrastructure,
        'Historical Floods': data.Historical_Floods,
        'Land Cover_Agricultural': data.Land_Cover_Agricultural,
        'Land Cover_Desert': data.Land_Cover_Desert,
        'Land Cover_Forest': data.Land_Cover_Forest,
        'Land Cover_Urban': data.Land_Cover_Urban,
        'Land Cover_Water Body': data.Land_Cover_Water_Body,
        'Soil Type_Clay': data.Soil_Type_Clay,
        'Soil Type_Loam': data.Soil_Type_Loam,
        'Soil Type_Peat': data.Soil_Type_Peat,
        'Soil Type_Sandy': data.Soil_Type_Sandy,
        'Soil Type_Silt': data.Soil_Type_Silt
    }
    
    new_data = pd.DataFrame([input_dict])
    new_data = new_data[expected_columns]
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)

    return {"flood_occurred": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
