import os
import lzma
import shutil
import argparse

def decompress_xz_to_npy(xz_file_path, destination_folder):
    # Ensure destination exists
    os.makedirs(destination_folder, exist_ok=True)

    # Check input
    if not xz_file_path.endswith('.xz'):
        raise ValueError("Input file must end with .xz")
    if not os.path.isfile(xz_file_path):
        raise FileNotFoundError(f"File not found: {xz_file_path}")

    # Derive output filename
    npy_output_path = os.path.join(destination_folder, "images.npy")

    # Decompress
    print(f"[INFO] Decompressing '{xz_file_path}' to '{npy_output_path}'...")
    with lzma.open(xz_file_path, 'rb') as f_in, open(npy_output_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"[SUCCESS] Decompressed to: {npy_output_path}")
    return npy_output_path


def main():
    parser = argparse.ArgumentParser(description="Decompress a .xz-compressed .npy file.")
    parser.add_argument("xz_file", help="Path to the .xz file (e.g., data.npy.xz)")
    parser.add_argument("destination", help="Destination folder to extract the .npy file")
    args = parser.parse_args()

    decompress_xz_to_npy(args.xz_file, args.destination)


if __name__ == "__main__":
    main()
