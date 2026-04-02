import zipfile
import re

def peek_npz_limited(npz_path):
    print(f"\nPeeking: {npz_path}")
    try:
        with zipfile.ZipFile(npz_path, 'r') as z:
            names = z.namelist()
            print(f"Total entries in ZIP: {len(names)}")
            
            # Look at first few .npy files to get shapes
            npy_files = [n for n in names if n.endswith('.npy')]
            print(f"Total .npy files: {len(npy_files)}")
            
            for name in npy_files[:3]:
                with z.open(name) as f:
                    header_bytes = f.read(1024)
                    header_str = header_bytes.decode('ascii', errors='ignore')
                    match = re.search(r"\{'descr':.*?\}", header_str, re.DOTALL)
                    if match:
                        print(f"[{name}] {match.group(0)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    peek_npz_limited(r"C:\Users\zeyuj\Downloads\expanded_dataset_500.npz")
    print("="*60)
    peek_npz_limited(r"C:\Users\zeyuj\Downloads\expanded_dataset_1000.npz")
