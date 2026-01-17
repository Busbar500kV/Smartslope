#!/usr/bin/env python3
"""
Bundle run outputs into a ZIP file.

Creates a ZIP archive of all files in a run directory for easy download and sharing.
Uses only Python stdlib (zipfile module) for maximum portability.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import List


def get_files_to_bundle(run_dir: Path, include_npz: bool = False) -> List[Path]:
    """
    Get list of files to include in the bundle.
    
    Args:
        run_dir: Run directory path
        include_npz: Whether to include .npz files (large binary data)
    
    Returns:
        List of file paths to bundle
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    if not run_dir.is_dir():
        raise ValueError(f"Not a directory: {run_dir}")
    
    files_to_bundle = []
    
    # Get all files in the directory (non-recursive)
    for item in run_dir.iterdir():
        if item.is_file():
            # Skip .npz files unless explicitly requested
            if item.suffix == '.npz' and not include_npz:
                print(f"  Skipping: {item.name} (use --include-npz to include)")
                continue
            
            # Skip existing ZIP files to avoid recursion
            if item.suffix == '.zip':
                print(f"  Skipping: {item.name} (existing ZIP)")
                continue
            
            files_to_bundle.append(item)
    
    return files_to_bundle


def create_bundle(
    run_dir: Path,
    zip_name: str,
    include_npz: bool = False
) -> Path:
    """
    Create a ZIP bundle of run outputs.
    
    Args:
        run_dir: Run directory path
        zip_name: Name of the ZIP file to create
        include_npz: Whether to include .npz files
    
    Returns:
        Path to created ZIP file
    """
    zip_path = run_dir / zip_name
    
    # Get files to bundle
    files_to_bundle = get_files_to_bundle(run_dir, include_npz=include_npz)
    
    if not files_to_bundle:
        raise ValueError(f"No files found to bundle in {run_dir}")
    
    print(f"\nBundling {len(files_to_bundle)} file(s)...")
    
    # Create ZIP file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in files_to_bundle:
            # Store with just the filename (no directory structure)
            arcname = file_path.name
            zf.write(file_path, arcname=arcname)
            print(f"  Added: {arcname}")
    
    return zip_path


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    size = float(size_bytes)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bundle Smartslope run outputs into a ZIP file"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run output directory (e.g., outputs/run_20260117_123456)",
    )
    parser.add_argument(
        "--zip-name",
        type=str,
        default=None,
        help="Name of ZIP file to create (default: <run_id>.zip)",
    )
    parser.add_argument(
        "--include-npz",
        action="store_true",
        help="Include .npz files in the bundle (may be large)",
    )
    
    args = parser.parse_args()
    
    run_dir = args.run_dir.resolve()
    
    # Default ZIP name: use run directory name + .zip
    if args.zip_name is None:
        zip_name = f"{run_dir.name}.zip"
    else:
        zip_name = args.zip_name
    
    print("=" * 60)
    print("  Smartslope Run Bundler")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"ZIP name: {zip_name}")
    print(f"Include NPZ: {args.include_npz}")
    
    try:
        zip_path = create_bundle(run_dir, zip_name, include_npz=args.include_npz)
        
        # Get file size
        size_bytes = zip_path.stat().st_size
        size_str = format_size(size_bytes)
        
        print("\n" + "=" * 60)
        print("  ✓ Bundle created successfully!")
        print("=" * 60)
        print(f"ZIP file: {zip_path}")
        print(f"Size: {size_str} ({size_bytes:,} bytes)")
        
        return 0
        
    except (FileNotFoundError, ValueError) as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
