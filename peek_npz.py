import zipfile
import re
import numpy as np

def peek_npz_shape(npz_path):
    print(f"\nInspecting headers: {npz_path}")
    try:
        with zipfile.ZipFile(npz_path, 'r') as z:
            for name in z.namelist()[:5]: # Peek at first few files
                if not name.endswith('.npy'):
                    continue
                try:
                    with z.open(name) as f:
                        header_bytes = f.read(1024)
                        header_str = header_bytes.decode('ascii', errors='ignore')
                        match = re.search(r"\{'descr':.*?\}", header_str, re.DOTALL)
                        if match:
                            print(f"--- {name} ---")
                            print(match.group(0))
                        else:
                            print(f"Could not find valid numpy dictionary header in {name}")
                except Exception as e:
                    print(f"Error reading {name}: {e}")
    except Exception as e:
        print(f"Error opening ZIP {npz_path}: {e}")

def list_keys(npz_path):
    print(f"\nLoading metadata: {npz_path}")
    try:
        data = np.load(npz_path)
        keys = list(data.keys())
        print(f"Total keys: {len(keys)}")
        print(f"Sample keys: {keys[:10]}")
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")

if __name__ == '__main__':
    for path in [r"C:\Users\zeyuj\Downloads\expanded_dataset_500.npz", 
                 r"C:\Users\zeyuj\Downloads\expanded_dataset_1000.npz"]:
        print("\n" + "="*60)
        list_keys(path)
        peek_npz_shape(path)
