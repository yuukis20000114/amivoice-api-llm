#!/usr/bin/env python3
"""
GPU availability checker for PyTorch, TensorFlow, CatBoost, and XGBoost
"""

import sys
from typing import Dict, Any

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_result(library: str, available: bool, details: str = ""):
    """Print formatted result"""
    status = "✅ Available" if available else "❌ Not Available"
    print(f"{library:<15}: {status}")
    if details:
        print(f"                {details}")

def check_pytorch() -> Dict[str, Any]:
    """Check PyTorch GPU availability"""
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        current_device = torch.cuda.current_device() if cuda_available else None
        device_name = torch.cuda.get_device_name(0) if cuda_available else None
        
        details = ""
        if cuda_available:
            details = f"Devices: {device_count}, Current: {current_device}, Name: {device_name}"
            # Simple GPU test
            try:
                x = torch.tensor([1.0, 2.0]).cuda()
                y = x * 2
                details += f", Test: {y.cpu().numpy().tolist()}"
            except Exception as e:
                details += f", Test failed: {str(e)}"
        
        return {
            'available': cuda_available,
            'details': details,
            'version': torch.__version__
        }
    except ImportError:
        return {'available': False, 'details': 'PyTorch not installed', 'version': None}
    except Exception as e:
        return {'available': False, 'details': f'Error: {str(e)}', 'version': None}

def check_tensorflow() -> Dict[str, Any]:
    """Check TensorFlow GPU availability"""
    try:
        import tensorflow as tf
        
        # Suppress TensorFlow warnings
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        gpu_available = len(gpus) > 0
        
        details = ""
        if gpu_available:
            details = f"GPUs found: {len(gpus)}"
            for i, gpu in enumerate(gpus):
                details += f", GPU{i}: {gpu.name}"
            
            # Simple GPU test
            try:
                with tf.device('/GPU:0'):
                    a = tf.constant([1.0, 2.0])
                    b = a * 2
                    details += f", Test: {b.numpy().tolist()}"
            except Exception as e:
                details += f", Test failed: {str(e)}"
        
        return {
            'available': gpu_available,
            'details': details,
            'version': tf.__version__
        }
    except ImportError:
        return {'available': False, 'details': 'TensorFlow not installed', 'version': None}
    except Exception as e:
        return {'available': False, 'details': f'Error: {str(e)}', 'version': None}

def check_catboost() -> Dict[str, Any]:
    """Check CatBoost GPU availability"""
    try:
        import catboost
        from catboost import CatBoostClassifier
        
        # Test GPU availability by trying to create a GPU-enabled model
        try:
            model = CatBoostClassifier(
                iterations=1,
                task_type="GPU",
                devices='0',
                verbose=False
            )
            # Try a simple fit to test GPU
            import numpy as np
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([0, 1, 0])
            model.fit(X, y, verbose=False)
            
            return {
                'available': True,
                'details': 'GPU training successful',
                'version': catboost.__version__
            }
        except Exception as gpu_error:
            # Try CPU version to check if CatBoost works at all
            try:
                model_cpu = CatBoostClassifier(iterations=1, verbose=False)
                return {
                    'available': False,
                    'details': f'GPU failed: {str(gpu_error)[:50]}..., but CPU works',
                    'version': catboost.__version__
                }
            except Exception:
                return {
                    'available': False,
                    'details': f'GPU not available: {str(gpu_error)[:50]}...',
                    'version': catboost.__version__
                }
                
    except ImportError:
        return {'available': False, 'details': 'CatBoost not installed', 'version': None}
    except Exception as e:
        return {'available': False, 'details': f'Error: {str(e)}', 'version': None}

def check_xgboost() -> Dict[str, Any]:
    """Check XGBoost GPU availability"""
    try:
        import xgboost as xgb
        
        # Test GPU availability
        try:
            import numpy as np
            from sklearn.datasets import make_classification
            
            # Create sample data
            X, y = make_classification(n_samples=100, n_features=4, random_state=42)
            dtrain = xgb.DMatrix(X, label=y)
            
            # Try GPU training
            params = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'verbosity': 0
            }
            
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=1,
                verbose_eval=False
            )
            
            return {
                'available': True,
                'details': 'GPU training successful with gpu_hist',
                'version': xgb.__version__
            }
            
        except Exception as gpu_error:
            # Try CPU version
            try:
                params_cpu = {
                    'tree_method': 'hist',
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'verbosity': 0
                }
                model_cpu = xgb.train(
                    params_cpu,
                    dtrain,
                    num_boost_round=1,
                    verbose_eval=False
                )
                return {
                    'available': False,
                    'details': f'GPU failed: {str(gpu_error)[:50]}..., but CPU works',
                    'version': xgb.__version__
                }
            except Exception:
                return {
                    'available': False,
                    'details': f'GPU not available: {str(gpu_error)[:50]}...',
                    'version': xgb.__version__
                }
                
    except ImportError:
        return {'available': False, 'details': 'XGBoost not installed', 'version': None}
    except Exception as e:
        return {'available': False, 'details': f'Error: {str(e)}', 'version': None}

def main():
    """Main function to check all libraries"""
    print_header("GPU Availability Checker")
    
    # System info
    print(f"Python version: {sys.version}")
    
    # Check each library
    libraries = {
        "PyTorch": check_pytorch,
        "TensorFlow": check_tensorflow, 
        "CatBoost": check_catboost,
        "XGBoost": check_xgboost
    }
    
    results = {}
    
    for name, check_func in libraries.items():
        print_header(f"Checking {name}")
        result = check_func()
        results[name] = result
        
        if result['version']:
            print(f"Version: {result['version']}")
        
        print_result(name, result['available'], result['details'])
    
    # Summary
    print_header("Summary")
    gpu_count = sum(1 for r in results.values() if r['available'])
    total_installed = sum(1 for r in results.values() if r['version'] is not None)
    
    print(f"Libraries installed: {total_installed}/4")
    print(f"GPU-enabled libraries: {gpu_count}/{total_installed}")
    
    for name, result in results.items():
        if result['version']:
            status = "🟢" if result['available'] else "🔴"
            print(f"{status} {name} v{result['version']}")

if __name__ == "__main__":
    main()