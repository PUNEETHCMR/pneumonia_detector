import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from flask import Flask, render_template, request, jsonify
from pathlib import Path
from src.predict import load_model, load_image
import base64
from io import BytesIO
import sys

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model (loaded once at startup)
model = None

def load_app_model():
    """Load the pneumonia detector model."""
    global model
    model_path = 'model/pneumonia_detector.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠ Model not found at {model_path}. Please train the model first.")
        print("Run: python src/train.py --data-dir path/to/dataset --output-model model/pneumonia_detector.h5")


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for pneumonia prediction."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        # Check if image is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'gif', 'bmp'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Allowed: jpg, jpeg, png, gif, bmp'}), 400
        
        # Save the file
        filename = f"{Path(file.filename).stem}_upload.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        probability = model.predict(load_image(filepath))[0][0]
        
        # Read image and convert to base64 for display
        with open(filepath, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()
        
        return jsonify({
            'success': True,
            'pneumonia_probability': float(probability),
            'diagnosis': 'PNEUMONIA DETECTED' if probability > 0.5 else 'NORMAL',
            'confidence': float(probability * 100 if probability > 0.5 else (1 - probability) * 100),
            'image_base64': f'data:image/png;base64,{img_base64}'
        })
    
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })


if __name__ == '__main__':
    load_app_model()
    print("\n" + "="*50, flush=True)
    print("🫁 Pneumonia Detector Web Interface", flush=True)
    print("="*50, flush=True)
    print("Starting Flask server...", flush=True)
    port = int(os.environ.get('PORT', 5000))
    print(f"→ Open http://localhost:{port} in your browser", flush=True)
    print("="*50 + "\n", flush=True)
    sys.stdout.flush()
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
