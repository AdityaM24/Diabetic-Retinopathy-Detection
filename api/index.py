from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image
import io
import os

# Try to import tflite_runtime, fallback to tensorflow if not available (for local dev)
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Neither tflite_runtime nor tensorflow is installed.")

app = FastAPI()

# Load TFLite model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model.tflite')
interpreter = None

def get_interpreter():
    global interpreter
    if interpreter is None:
        if not os.path.exists(MODEL_PATH):
             raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        
        # Load the TFLite model and allocate tensors.
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
    return interpreter

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Serve the index.html from templates
    template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'index.html')
    with open(template_path, 'r') as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Preprocess image
        image = image.resize((224, 224))
        input_data = np.array(image, dtype=np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        interpreter = get_interpreter()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Process results
        class_names = ['DR', 'No_DR']
        predicted_idx = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0]) * 100)
        
        return JSONResponse({
            "prediction": class_names[predicted_idx],
            "confidence": f"{confidence:.2f}%",
            "probabilities": output_data[0].tolist()
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
