import yaml
from pathlib import Path
import subprocess
import sys
from utils import setup_logging

class DataPreparationPipeline:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.load_config()
        self.logger = setup_logging("Data_Prep_Pipeline")
        
    def load_config(self):
        """Load configuration from params.yaml"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def run_step(self, step_name, script_name):
        """Run a single pipeline step"""
        self.logger.info(f"Running {step_name}...")
        
        script_path = Path(__file__).parent / script_name
        
        if not script_path.exists():
            self.logger.error(f"Script not found: {script_path}")
            return False
        
        self.logger.info(f"Executing: {script_path}")
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        
        if result.returncode == 0:
            self.logger.info(f"   {step_name} completed successfully!")
            # Print the output of the step
            if result.stdout:
                # Log only the last few lines to avoid clutter
                lines = result.stdout.strip().split('\n')
                if len(lines) > 5:
                    self.logger.info("Output (last 5 lines):")
                    for line in lines[-5:]:
                        if line.strip():
                            self.logger.info(f"  {line}")
                else:
                    for line in lines:
                        if line.strip():
                            self.logger.info(f"  {line}")
            return True
        else:
            self.logger.error(f"❌ {step_name} failed!")
            self.logger.error(f"Error output: {result.stderr}")
            # Also print stdout for debugging
            if result.stdout:
                self.logger.info(f"Script output: {result.stdout}")
            return False
    
    def run(self):
        """Run the complete pipeline"""
        self.logger.info("Starting Data Preparation Pipeline...")
        
        steps = [
            ("COCO to YOLO Conversion", "01_convert_coco_to_yolo.py"),
            ("Dataset Filtering", "02_filter_dataset.py"),
            ("Annotation Cleaning", "03_clean_annotations.py"), 
            ("YOLO Preprocessing", "04_preprocess_yolo.py"),
            ("Dataset Verification", "05_verify_dataset.py")
        ]
        
        successful_steps = 0
        for step_name, script_name in steps:
            success = self.run_step(step_name, script_name)
            if not success:
                self.logger.error(f"🚨 Pipeline failed at {step_name}!")
                self.logger.error(f"Completed {successful_steps}/{len(steps)} steps successfully.")
                return False
            successful_steps += 1
        
        self.logger.info("🎉 Data Preparation Pipeline completed successfully!")
        self.logger.info(f"All {len(steps)} steps completed without errors.")
        self.logger.info("All dataset versions have been tracked with DVC and pushed to remote!")
        return True

def main():
    # More robust path handling
    current_script = Path(__file__)
    project_root = current_script.parents[2]  # Adjust based on your actual structure
    
    # Try multiple possible locations for the config file
    possible_config_paths = [
        project_root / 'configs' / 'params.yaml',
        project_root / 'params.yaml',
        current_script.parents[3] / 'configs' / 'params.yaml',
    ]
    
    config_path = None
    for path in possible_config_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path is None:
        # If no config found, try to find it by searching
        config_files = list(project_root.rglob('params.yaml'))
        if config_files:
            config_path = config_files[0]
        else:
            raise FileNotFoundError("Could not find params.yaml config file")
    
    print(f"Using config file: {config_path}")
    pipeline = DataPreparationPipeline(config_path)
    success = pipeline.run()
    
    if success:
        print("\n" + "="*60)
        print("   PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Your dataset is now ready for model training!")
        print("Next steps:")
        print("1. Check the generated dataset in data/processed/")
        print("2. Start training your YOLO model")
        print("3. Monitor training progress with your preferred tool")
    else:
        print("\n" + "="*60)
        print("   PIPELINE FAILED!")
        print("="*60)
        print("Check the logs above to identify the failing step.")
        sys.exit(1)

if __name__ == "__main__":
    main()