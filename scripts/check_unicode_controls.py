#!/usr/bin/env python3
"""Check for Unicode control characters in source files.

This script scans Python and configuration files for potentially dangerous
Unicode control characters that could be used for obfuscation attacks.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Unicode control characters to detect (excluding common whitespace)
DANGEROUS_UNICODE = [
    '\u200B',  # Zero Width Space
    '\u200C',  # Zero Width Non-Joiner
    '\u200D',  # Zero Width Joiner
    '\u200E',  # Left-to-Right Mark
    '\u200F',  # Right-to-Left Mark
    '\u202A',  # Left-to-Right Embedding
    '\u202B',  # Right-to-Left Embedding
    '\u202C',  # Pop Directional Formatting
    '\u202D',  # Left-to-Right Override
    '\u202E',  # Right-to-Left Override
    '\uFEFF',  # Zero Width No-Break Space
]


def check_file(file_path: Path) -> List[Tuple[int, str]]:
    """Check a file for dangerous Unicode control characters.
    
    Args:
        file_path: Path to file to check
    
    Returns:
        List of (line_number, character_name) tuples for any violations
    """
    violations = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                for char in DANGEROUS_UNICODE:
                    if char in line:
                        char_name = f"U+{ord(char):04X}"
                        violations.append((line_num, char_name))
    except UnicodeDecodeError:
        # Skip binary files
        pass
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
    
    return violations


def main() -> int:
    """Main entry point."""
    repo_root = Path(__file__).resolve().parents[1]
    
    # File patterns to check
    patterns = ['**/*.py', '**/*.json', '**/*.md', '**/*.yml', '**/*.yaml']
    
    # Directories to exclude
    exclude_dirs = {'.venv', '__pycache__', '.git', 'node_modules', 'dist', 'build'}
    
    all_violations = {}
    
    for pattern in patterns:
        for file_path in repo_root.glob(pattern):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue
            
            # Skip if not a file
            if not file_path.is_file():
                continue
            
            violations = check_file(file_path)
            if violations:
                rel_path = file_path.relative_to(repo_root)
                all_violations[rel_path] = violations
    
    # Report results
    if all_violations:
        print("✗ Unicode control character violations found:\n")
        for file_path, violations in all_violations.items():
            print(f"{file_path}:")
            for line_num, char_name in violations:
                print(f"  Line {line_num}: {char_name}")
        print(f"\nTotal: {sum(len(v) for v in all_violations.values())} violations in {len(all_violations)} files")
        return 1
    else:
        print("✓ No Unicode control character violations found")
        return 0


if __name__ == '__main__':
    sys.exit(main())
