from pathlib import Path

def debug_project_root():
    """Debug the project root path calculation"""
    current_file = Path(__file__)
    print(f"Current file location: {current_file}")
    print(f"Current working directory: {Path.cwd()}")
    
    # Test different parent levels
    for i in range(6):
        try:
            parent = current_file.parents[i]
            print(f"Parent {i}: {parent}")
            
            # Check if configs directory exists here
            configs_dir = parent / 'configs'
            if configs_dir.exists():
                print(f"✅ FOUND CONFIGS at level {i}: {configs_dir}")
                params_file = configs_dir / 'params.yaml'
                if params_file.exists():
                    print(f"✅ FOUND params.yaml: {params_file}")
                else:
                    print(f"❌ params.yaml not found in: {configs_dir}")
        except IndexError:
            print(f"❌ No parent at level {i}")
            break
    
    print("\n" + "="*50)
    
    # Alternative: search from current directory
    print("Searching from current directory:")
    current_dir = Path.cwd()
    for path in current_dir.parents:
        configs_dir = path / 'configs'
        if configs_dir.exists():
            print(f"✅ Found configs in: {path}")
            break

if __name__ == "__main__":
    debug_project_root()