import zipfile
import re

def peek_npz_shape(npz_path):
    print(f"Inspecting: {npz_path}")
    with zipfile.ZipFile(npz_path, 'r') as z:
        for name in z.namelist():
            if not name.endswith('.npy'):
                continue
            with z.open(name) as f:
                # Read the first 1KB which definitely contains the header
                header_bytes = f.read(1024)
                
                # NumPy headers contain a string representation of a Python dict like:
                # {'descr': '<f4', 'fortran_order': False, 'shape': (1000, 2016, 2016), }
                # Convert bytes to string (ignoring decode errors for the actual binary data after the header)
                header_str = header_bytes.decode('ascii', errors='ignore')
                
                # Extract the dictionary string
                match = re.search(r"\{'descr':.*?\}", header_str)
                if match:
                    print(f"\n--- {name} ---")
                    print(match.group(0))
                else:
                    print(f"Could not find valid numpy dictionary header in {name}")

if __name__ == '__main__':
    peek_npz_shape(r"C:\Users\zeyuj\Downloads\expanded_dataset_1000.npz")
