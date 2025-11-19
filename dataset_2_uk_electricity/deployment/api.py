from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import time

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_PATH = 'xgb_tuned_model_20251107_134336.pkl'
MODEL_VERSION = 'xgb_tuned_v1.0'
EXPECTED_RMSE = 2238.88  # From training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess_input_data(df):
    """
    Preprocess input data for model inference.
    CRITICAL: Must match training preprocessing exactly.
    """
    df = df.copy()
    
    # Handle datetime if present
    if 'settlement_date' in df.columns:
        df['settlement_date'] = pd.to_datetime(df['settlement_date'])
        
        if 'year' not in df.columns:
            df['year'] = df['settlement_date'].dt.year
        if 'month' not in df.columns:
            df['month'] = df['settlement_date'].dt.month
        if 'day' not in df.columns:
            df['day'] = df['settlement_date'].dt.day
        if 'dayofweek' not in df.columns:
            df['dayofweek'] = df['settlement_date'].dt.dayofweek
        if 'week' not in df.columns:
            df['week'] = df['settlement_date'].dt.isocalendar().week
        if 'is_weekend' not in df.columns:
            df['is_weekend'] = (df['settlement_date'].dt.dayofweek >= 5).astype(int)
    
    # Remove excluded features (DATA LEAKAGE + unused features)
    excluded = ['settlement_date', 'england_wales_demand', 'nd', 'tsd', 'hour', 'quarter']
    for col in excluded:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Handle missing values
    flow_features = ['scottish_transfer', 'ifa_flow', 'ifa2_flow', 'britned_flow', 
                     'moyle_flow', 'east_west_flow', 'nemo_flow', 
                     'nsl_flow', 'eleclink_flow', 'viking_flow', 'greenlink_flow']
    
    for col in flow_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    renewable_features = ['embedded_wind_generation', 'embedded_solar_generation',
                         'embedded_wind_capacity', 'embedded_solar_capacity']
    
    for col in renewable_features:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Ensure correct data types
    df['settlement_period'] = df['settlement_period'].astype(int)
    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['day'] = df['day'].astype(int)
    df['dayofweek'] = df['dayofweek'].astype(int)
    df['week'] = df['week'].astype(int)
    df['is_weekend'] = df['is_weekend'].astype(int)
    
    # Ensure correct feature order (matches training)
    expected_features = [
        'settlement_period', 'embedded_wind_generation', 'embedded_wind_capacity',
        'embedded_solar_generation', 'embedded_solar_capacity', 'non_bm_stor',
        'pump_storage_pumping', 'scottish_transfer', 'ifa_flow', 'ifa2_flow',
        'britned_flow', 'moyle_flow', 'east_west_flow', 'nemo_flow', 'nsl_flow',
        'eleclink_flow', 'viking_flow', 'greenlink_flow', 'year', 'month', 'day',
        'dayofweek', 'week', 'is_weekend'
    ]
    
    # Select only available features in correct order
    available_features = [col for col in expected_features if col in df.columns]
    df = df[available_features]
    
    return df

# ============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# ============================================================================

class PredictionRequest(BaseModel):
    """Request schema for single prediction."""
    
    # Time features
    settlement_period: int = Field(..., ge=1, le=48, description="Half-hourly period (1-48)")
    year: int = Field(..., ge=2001, le=2030, description="Year")
    month: int = Field(..., ge=1, le=12, description="Month")
    day: int = Field(..., ge=1, le=31, description="Day of month")
    dayofweek: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    week: int = Field(..., ge=1, le=53, description="Week of year")
    is_weekend: int = Field(..., ge=0, le=1, description="Is weekend (0=no, 1=yes)")
    
    # Energy features
    embedded_wind_generation: Optional[float] = Field(0.0, description="Embedded wind generation (MW)")
    embedded_wind_capacity: Optional[float] = Field(0.0, description="Embedded wind capacity (MW)")
    embedded_solar_generation: Optional[float] = Field(0.0, description="Embedded solar generation (MW)")
    embedded_solar_capacity: Optional[float] = Field(0.0, description="Embedded solar capacity (MW)")
    non_bm_stor: Optional[int] = Field(0, description="Non-BM STOR (MW)")
    pump_storage_pumping: Optional[int] = Field(0, description="Pump storage pumping (MW)")
    
    # Interconnector flows
    scottish_transfer: Optional[float] = Field(0.0, description="Scottish transfer (MW)")
    ifa_flow: Optional[int] = Field(0, description="IFA interconnector flow (MW)")
    ifa2_flow: Optional[float] = Field(0.0, description="IFA2 interconnector flow (MW)")
    britned_flow: Optional[float] = Field(0.0, description="BritNed interconnector flow (MW)")
    moyle_flow: Optional[float] = Field(0.0, description="Moyle interconnector flow (MW)")
    east_west_flow: Optional[float] = Field(0.0, description="East-West interconnector flow (MW)")
    nemo_flow: Optional[float] = Field(0.0, description="Nemo interconnector flow (MW)")
    nsl_flow: Optional[float] = Field(0.0, description="NSL interconnector flow (MW)")
    eleclink_flow: Optional[float] = Field(0.0, description="ElecLink interconnector flow (MW)")
    viking_flow: Optional[float] = Field(0.0, description="Viking interconnector flow (MW)")
    greenlink_flow: Optional[float] = Field(0.0, description="Greenlink interconnector flow (MW)")
    
    class Config:
        schema_extra = {
            "example": {
                "settlement_period": 24,
                "year": 2025,
                "month": 11,
                "day": 13,
                "dayofweek": 2,
                "week": 46,
                "is_weekend": 0,
                "embedded_wind_generation": 5000.0,
                "embedded_wind_capacity": 15000.0,
                "embedded_solar_generation": 3000.0,
                "embedded_solar_capacity": 14000.0,
                "non_bm_stor": 500,
                "pump_storage_pumping": 200,
                "scottish_transfer": 2000.0,
                "ifa_flow": 1000,
                "ifa2_flow": 0.0,
                "britned_flow": 500.0,
                "moyle_flow": 300.0,
                "east_west_flow": 0.0,
                "nemo_flow": 0.0,
                "nsl_flow": 0.0,
                "eleclink_flow": 0.0,
                "viking_flow": 0.0,
                "greenlink_flow": 0.0
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    
    predicted_demand_mw: float = Field(..., description="Predicted national demand (MW)")
    confidence_interval_lower: float = Field(..., description="Lower bound of confidence interval (MW)")
    confidence_interval_upper: float = Field(..., description="Upper bound of confidence interval (MW)")
    model_version: str = Field(..., description="Model version used")
    prediction_time_ms: Optional[float] = Field(None, description="Prediction time in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "predicted_demand_mw": 36542.15,
                "confidence_interval_lower": 34303.27,
                "confidence_interval_upper": 38781.03,
                "model_version": "xgb_tuned_v1.0",
                "prediction_time_ms": 12.5
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    
    predictions: List[float] = Field(..., description="List of predicted demands (MW)")
    count: int = Field(..., description="Number of predictions")
    mean_demand: float = Field(..., description="Mean of predictions")
    min_demand: float = Field(..., description="Minimum prediction")
    max_demand: float = Field(..., description="Maximum prediction")
    model_version: str = Field(..., description="Model version used")
    prediction_time_ms: float = Field(..., description="Total prediction time in milliseconds")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Electricity Demand Forecasting API",
    description="Predict UK national electricity demand using XGBoost model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global model variable
model = None

@app.on_event("startup")
async def startup_event():
    """Load model when API starts."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        logger.info(f"   Model type: {type(model).__name__}")
    except FileNotFoundError:
        logger.error(f"❌ Model file not found: {MODEL_PATH}")
        logger.error("   API will not be able to make predictions!")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")


@app.get("/")
def root():
    """API information."""
    return {
        "name": "Electricity Demand Forecasting API",
        "version": "1.0.0",
        "model": MODEL_VERSION,
        "model_loaded": model is not None,
        "endpoints": {
            "info": "GET /",
            "health": "GET /health",
            "docs": "GET /docs",
            "single_prediction": "POST /predict",
            "batch_prediction": "POST /batch_predict"
        },
        "model_performance": {
            "test_r2": 0.8697,
            "test_rmse_mw": EXPECTED_RMSE,
            "test_mae_mw": 1722.70
        }
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "error": "Model not loaded"
            }
        )
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make a single prediction for electricity demand.
    
    Returns predicted national demand with confidence intervals.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )
    
    start_time = time.time()
    
    try:
        # Convert request to DataFrame
        input_df = pd.DataFrame([request.dict()])
        
        logger.info(f"Prediction request: Period {request.settlement_period}, "
                   f"Date {request.year}-{request.month:02d}-{request.day:02d}")
        
        # Preprocess
        input_df = preprocess_input_data(input_df)
        
        # Predict
        prediction = model.predict(input_df)[0]
        
        # Calculate confidence interval (±1 RMSE)
        lower_bound = prediction - EXPECTED_RMSE
        upper_bound = prediction + EXPECTED_RMSE
        
        # Sanity check
        if not 15000 <= prediction <= 60000:
            logger.warning(f"⚠️  Prediction {prediction:.2f} MW is outside typical UK demand range!")
        
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"✅ Prediction: {prediction:.2f} MW (took {prediction_time:.2f}ms)")
        
        return PredictionResponse(
            predicted_demand_mw=float(prediction),
            confidence_interval_lower=float(lower_bound),
            confidence_interval_upper=float(upper_bound),
            model_version=MODEL_VERSION,
            prediction_time_ms=prediction_time
        )
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(requests: List[PredictionRequest]):
    """
    Make batch predictions for multiple time periods.
    
    Accepts a list of prediction requests and returns predictions for all.
    Maximum 1000 predictions per request.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check server logs."
        )
    
    if len(requests) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 predictions per batch request"
        )
    
    start_time = time.time()
    
    try:
        # Convert requests to DataFrame
        input_df = pd.DataFrame([req.dict() for req in requests])
        
        logger.info(f"Batch prediction request: {len(requests)} predictions")
        
        # Preprocess
        input_df = preprocess_input_data(input_df)
        
        # Predict
        predictions = model.predict(input_df)
        
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        
        logger.info(f"✅ Batch prediction complete: {len(predictions)} predictions "
                   f"(took {prediction_time:.2f}ms, {prediction_time/len(predictions):.2f}ms per prediction)")
        
        return BatchPredictionResponse(
            predictions=[float(pred) for pred in predictions],
            count=len(predictions),
            mean_demand=float(predictions.mean()),
            min_demand=float(predictions.min()),
            max_demand=float(predictions.max()),
            model_version=MODEL_VERSION,
            prediction_time_ms=prediction_time
        )
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ============================================================================
# MAIN (for direct execution)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║   Electricity Demand Forecasting API                             ║
    ║   Starting server...                                              ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )