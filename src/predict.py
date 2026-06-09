import argparse
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image
from .model import build_model


def load_model(model_path: str) -> tf.keras.Model:
    """Load model from various formats: H5, SavedModel, or weights."""
    model_path_obj = Path(model_path)
    
    # Strategy 1: Try loading as full saved model (H5 or SavedModel directory)
    try:
        if model_path_obj.is_dir():
            # Try SavedModel format (directory)
            model = tf.keras.models.load_model(str(model_path_obj))
            return model
        elif model_path_obj.suffix == '.h5':
            # Try full H5 model
            try:
                model = tf.keras.models.load_model(str(model_path_obj))
                return model
            except Exception as e:
                print(f"Could not load as full model ({type(e).__name__}), trying weights...")
                # Might be weights only, rebuild architecture and load weights
                model = build_model()
                model.load_weights(str(model_path_obj))
                return model
    except Exception as e:
        print(f"Attempting alternative loading strategies...")
    
    # Strategy 2: Check for SavedModel in parent directory
    saved_model_dir = model_path_obj.parent / model_path_obj.stem
    if saved_model_dir.is_dir():
        try:
            model = tf.keras.models.load_model(str(saved_model_dir))
            return model
        except:
            pass
    
    # Strategy 3: Rebuild model and load weights
    try:
        print("Rebuilding model architecture and loading weights...")
        model = build_model()
        model.load_weights(str(model_path_obj))
        return model
    except Exception as e:
        raise RuntimeError(f"Could not load model from {model_path}: {e}")


def load_image(img_path: str, target_size=(224, 224)) -> tf.Tensor:
    img = image.load_img(img_path, target_size=target_size)
    array = image.img_to_array(img)
    normalized = array / 255.0
    return tf.expand_dims(normalized, axis=0)


def predict_image(model: tf.keras.Model, img_path: str) -> float:
    img_tensor = load_image(img_path)
    probability = model.predict(img_tensor)[0][0]
    return float(probability)


def parse_args():
    parser = argparse.ArgumentParser(description='Predict pneumonia from a chest X-ray image.')
    parser.add_argument('--model', required=True, help='Path to the trained model file.')
    parser.add_argument('--image', required=True, help='Path to the chest X-ray image.')
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model)
    probability = predict_image(model, args.image)
    label = 'Pneumonia' if probability >= 0.5 else 'Normal'
    print(f'Prediction: {label} (score={probability:.4f})')


if __name__ == '__main__':
    main()
