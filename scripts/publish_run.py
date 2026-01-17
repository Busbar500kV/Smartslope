#!/usr/bin/env python3
"""
Smartslope artifact publisher.

Publishes selected artifacts from a run to a tracked git directory for traceability.
Uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return "unknown"


def is_git_clean() -> bool:
    """Check if git working tree is clean."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet"],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        
        result = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


def check_git_safety(force: bool = False) -> tuple[bool, str]:
    """
    Check if it's safe to publish.
    
    Returns:
        (is_safe, reason_if_not_safe)
    """
    branch = get_git_branch()
    
    # Check if on main branch (or allow with --force)
    if branch not in ["main", "master"] and not force:
        return False, f"Not on main branch (current: {branch}). Use --force to override."
    
    # Check for uncommitted changes
    if not is_git_clean():
        return False, "Git working tree has uncommitted changes. Commit or stash them first."
    
    return True, ""


def copy_artifacts(run_id: str, outputs_dir: Path, artifacts_dir: Path) -> List[str]:
    """
    Copy selected artifacts from outputs to artifacts directory.
    
    Returns list of relative paths (from artifacts_dir) of copied files.
    """
    run_output_dir = outputs_dir / run_id
    if not run_output_dir.exists():
        raise FileNotFoundError(f"Run output directory not found: {run_output_dir}")
    
    run_artifacts_dir = artifacts_dir / "runs" / run_id
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = []
    
    # Copy PNG files (plots)
    for png_file in run_output_dir.glob("*.png"):
        dest = run_artifacts_dir / png_file.name
        shutil.copy2(png_file, dest)
        copied_files.append(f"runs/{run_id}/{png_file.name}")
        print(f"  Copied: {png_file.name}")
    
    # Copy TXT files (summaries)
    for txt_file in run_output_dir.glob("*.txt"):
        dest = run_artifacts_dir / txt_file.name
        shutil.copy2(txt_file, dest)
        copied_files.append(f"runs/{run_id}/{txt_file.name}")
        print(f"  Copied: {txt_file.name}")
    
    # Copy JSON files if any (config, metadata)
    for json_file in run_output_dir.glob("*.json"):
        dest = run_artifacts_dir / json_file.name
        shutil.copy2(json_file, dest)
        copied_files.append(f"runs/{run_id}/{json_file.name}")
        print(f"  Copied: {json_file.name}")
    
    if not copied_files:
        raise ValueError(f"No artifacts found to copy in {run_output_dir}")
    
    return copied_files


def create_manifest(run_id: str, artifacts_dir: Path, copied_files: List[str], notes: str = "") -> None:
    """Create manifest.json for this run."""
    manifest_path = artifacts_dir / "runs" / run_id / "manifest.json"
    
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit_hash": get_git_commit_hash(),
        "git_branch": get_git_branch(),
        "files_published": copied_files,
        "notes": notes,
    }
    
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Created: manifest.json")


def update_index(artifacts_dir: Path, run_id: str) -> None:
    """Update or create the global index.json with all published runs."""
    index_path = artifacts_dir / "index.json"
    
    # Load existing index or create new one
    if index_path.exists():
        with index_path.open("r") as f:
            index = json.load(f)
    else:
        index = {"published_runs": []}
    
    # Load manifest for this run
    manifest_path = artifacts_dir / "runs" / run_id / "manifest.json"
    with manifest_path.open("r") as f:
        manifest = json.load(f)
    
    # Check if run already in index (update it)
    existing_idx = None
    for i, entry in enumerate(index["published_runs"]):
        if entry["run_id"] == run_id:
            existing_idx = i
            break
    
    run_entry = {
        "run_id": run_id,
        "timestamp": manifest["timestamp"],
        "git_commit_hash": manifest["git_commit_hash"],
        "git_branch": manifest["git_branch"],
        "file_count": len(manifest["files_published"]),
    }
    
    if existing_idx is not None:
        index["published_runs"][existing_idx] = run_entry
    else:
        index["published_runs"].append(run_entry)
    
    # Sort by timestamp (newest first)
    index["published_runs"].sort(key=lambda x: x["timestamp"], reverse=True)
    
    with index_path.open("w") as f:
        json.dump(index, f, indent=2)
    
    print(f"  Updated: index.json")


def git_commit_and_push(run_id: str, artifacts_dir: Path, dry_run: bool = False) -> bool:
    """
    Commit and push the artifacts to git.
    
    Returns True if successful.
    """
    commit_msg = f"Smartslope: publish artifacts for {run_id}"
    
    if dry_run:
        print("\nDRY RUN - Would execute:")
        print(f"  git add {artifacts_dir}")
        print(f"  git commit -m '{commit_msg}'")
        print("  git push origin <current-branch>")
        return True
    
    try:
        # Git add
        subprocess.run(
            ["git", "add", str(artifacts_dir)],
            check=True,
            capture_output=True,
        )
        
        # Git commit
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            check=True,
            capture_output=True,
        )
        
        # Git push
        branch = get_git_branch()
        result = subprocess.run(
            ["git", "push", "origin", branch],
            capture_output=True,
            text=True,
            check=True,
        )
        
        print(f"\n✓ Successfully committed and pushed to origin/{branch}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Git operation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish Smartslope run artifacts to tracked git directory"
    )
    parser.add_argument(
        "run_id",
        help="Run ID to publish (e.g., run_20260117_123456)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (default: auto-detect)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force publish even if not on main branch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Optional notes to include in manifest",
    )
    
    args = parser.parse_args()
    
    # Determine repo root
    if args.repo_root:
        repo_root = args.repo_root.resolve()
    else:
        # Auto-detect: this script is in scripts/, so repo_root is parent
        repo_root = Path(__file__).resolve().parent.parent
    
    outputs_dir = repo_root / "outputs"
    artifacts_dir = repo_root / "artifacts"
    
    print("=" * 60)
    print("  Smartslope Artifact Publisher")
    print("=" * 60)
    print(f"Run ID: {args.run_id}")
    print(f"Repo root: {repo_root}")
    print(f"Git branch: {get_git_branch()}")
    print(f"Git commit: {get_git_commit_hash()[:8]}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]")
    
    # Safety checks
    print("\n--- Safety checks ---")
    is_safe, reason = check_git_safety(force=args.force)
    if not is_safe:
        print(f"✗ {reason}")
        return 1
    print("✓ Git safety checks passed")
    
    # Copy artifacts
    print("\n--- Copying artifacts ---")
    try:
        copied_files = copy_artifacts(args.run_id, outputs_dir, artifacts_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ {e}")
        return 1
    
    print(f"\n✓ Copied {len(copied_files)} file(s)")
    
    # Create manifest
    print("\n--- Creating manifest ---")
    create_manifest(args.run_id, artifacts_dir, copied_files, notes=args.notes)
    
    # Update index
    update_index(artifacts_dir, args.run_id)
    
    # Show what will be committed
    print("\n--- Files to be committed ---")
    for f in copied_files:
        print(f"  artifacts/{f}")
    print(f"  artifacts/runs/{args.run_id}/manifest.json")
    print("  artifacts/index.json")
    
    # Commit and push
    print("\n--- Committing and pushing ---")
    if not git_commit_and_push(args.run_id, artifacts_dir, dry_run=args.dry_run):
        return 1
    
    print("\n" + "=" * 60)
    print("  ✓ Publish complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
