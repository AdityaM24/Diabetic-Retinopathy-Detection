import tensorflow as tf

def convert_model(model_path='mobilenetv2_dr_imp.h5', output_path='model.tflite'):
    try:
        print(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        print("Converting model to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size and latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()

        print(f"Saving TFLite model to {output_path}...")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
            
        print("Conversion successful!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    convert_model()
