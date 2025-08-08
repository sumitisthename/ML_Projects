import mlflow
import os

# Replace with your actual token from https://dagshub.com/user/settings/tokens
DAGSHUB_TOKEN = "a9243f39c64d2d6812f764347c1e2d10485836b8"

os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/sumitisthename/MachineLearningPipeline.mlflow'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'sumitisthename' 
os.environ['MLFLOW_TRACKING_PASSWORD'] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

try:
    # Test basic connection
    experiments = mlflow.search_experiments()
    print("✅ Connection successful!")
    print(f"Found {len(experiments)} experiments")
    
    # Test creating a run
    with mlflow.start_run():
        mlflow.log_metric("test_metric", 0.95)
        print("✅ Run creation successful!")
        
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Check your token at: https://dagshub.com/user/settings/tokens")
    print("2. Make sure the repository exists")
    print("3. Verify your username is correct")