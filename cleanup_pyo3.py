#!/usr/bin/env python3
"""
Remove all PyO3/Python binding code from frames.rs while keeping pure Rust logic.
"""

import re
from typing import List, Tuple

def find_function_end(lines: List[str], start_idx: int) -> int:
    """Find the end of a function starting at start_idx by tracking braces."""
    brace_count = 0
    found_open_brace = False

    for i in range(start_idx, len(lines)):
        line = lines[i]

        # Count braces
        for char in line:
            if char == '{':
                brace_count += 1
                found_open_brace = True
            elif char == '}':
                brace_count -= 1

        # If we've seen an opening brace and are back to zero, function is done
        if found_open_brace and brace_count == 0:
            return i

    return len(lines) - 1

def is_python_function(lines: List[str], start_idx: int, end_idx: int) -> bool:
    """Check if a function uses Python types."""
    function_text = ''.join(lines[start_idx:end_idx+1])

    python_indicators = [
        'PyReadonlyArray',
        'PyResult',
        'Python<',
        'Bound<',
        'PyArray1',
        'pyo3::exceptions',
        'PyArray1::from_slice_bound',
        '.as_slice()?',
    ]

    return any(indicator in function_text for indicator in python_indicators)

def should_remove_function(func_name: str) -> bool:
    """Check if a function should be removed based on its name."""
    remove_patterns = [
        'py_new',
        'py_from_',
        'py_to_',
        'py_position',
        'py_velocity',
        '__repr__',
        '__str__',
        'from_orbital_elements_py',
        'py_epoch',
    ]

    # Special case: get_position and get_velocity with Python types should be removed
    # but get_obstime that returns Epoch should be kept
    python_getters = ['get_position', 'get_velocity']

    for pattern in remove_patterns:
        if pattern in func_name:
            return True

    if func_name in python_getters:
        return True

    return False

def process_file(input_path: str, output_path: str):
    """Process the Rust file to remove Python bindings."""

    with open(input_path, 'r') as f:
        lines = f.readlines()

    result_lines = []
    i = 0
    removed_count = 0
    removed_attrs = 0

    while i < len(lines):
        line = lines[i]

        # Check for orphaned Python attributes (including #[pyo3(...)])
        if re.match(r'^\s*#\[(pyo3|new|getter|staticmethod)', line):
            print(f"Removing orphaned attribute at line {i+1}: {line.strip()}")
            removed_attrs += 1
            i += 1
            continue

        # Check if this line starts a function definition
        func_match = re.match(r'^(\s*)(?:pub )?fn (\w+)', line)

        if func_match:
            indent = func_match.group(1)
            func_name = func_match.group(2)

            # Find the end of this function
            func_end = find_function_end(lines, i)

            # Check if it should be removed
            if should_remove_function(func_name) or is_python_function(lines, i, func_end):
                print(f"Removing function: {func_name} (lines {i+1}-{func_end+1})")
                removed_count += 1
                i = func_end + 1
                continue

        result_lines.append(line)
        i += 1

    # Clean up multiple blank lines (more than 2 consecutive)
    final_lines = []
    blank_count = 0
    for line in result_lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                final_lines.append(line)
        else:
            blank_count = 0
            final_lines.append(line)

    with open(output_path, 'w') as f:
        f.writelines(final_lines)

    print(f"\nProcessing complete!")
    print(f"Original lines: {len(lines)}")
    print(f"Final lines: {len(final_lines)}")
    print(f"Lines removed: {len(lines) - len(final_lines)}")
    print(f"Functions removed: {removed_count}")
    print(f"Attributes removed: {removed_attrs}")

if __name__ == '__main__':
    process_file('src/coordinates/frames.rs', 'src/coordinates/frames.rs')
