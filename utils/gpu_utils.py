import os
import tensorflow as tf

def configure_gpu():
    # Set XLA GPU flags
    os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/apps/cuda/11.2.2'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

    print("TensorFlow Version:", tf.__version__)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs Available: {len(gpus)}")

    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Use both GPUs with memory limit
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=25000)]
                )
            
            # Select primary GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            
            print("GPU(s) configured successfully")
            return True
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
            return False
    else:
        print("No GPU available. Running on CPU")
        return False