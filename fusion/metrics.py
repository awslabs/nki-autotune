import numpy as np


def check_correctness(desired, actual, atol, rtol):
    abs_diff = np.abs(actual - desired)
    # Avoid division by zero in relative difference calculation
    rel_diff = np.divide(abs_diff, np.abs(desired), out=np.zeros_like(abs_diff), where=np.abs(desired) != 0)

    # Calculate tolerance threshold using numpy's allclose formula
    tolerance = atol + rtol * np.abs(desired)
    mismatches = abs_diff > tolerance
    total_mismatches = np.sum(mismatches)
    total_elements = desired.size

    if total_mismatches > 0:
        # Calculate statistics
        mismatch_percentage = (total_mismatches / total_elements) * 100
        max_abs_diff = np.max(abs_diff)
        max_rel_diff = np.max(rel_diff)

        # Generate error message with statistics and mismatch regions
        regions_summary = generate_mismatch_summary(mismatches, desired, actual)

        err_msg = (
            f"Mismatched elements:\n"
            f"{total_mismatches} / {total_elements} ({mismatch_percentage:.6f}%)\n"
            f"Max absolute difference: {max_abs_diff}\n"
            f"Max relative difference: {max_rel_diff}\n"
            f"{regions_summary}"
        )

        raise AssertionError(err_msg)


def generate_mismatch_summary(mismatches, desired, actual):
    """Generate a summary of contiguous regions with mismatches."""
    if len(mismatches.shape) == 2:  # For 2D arrays
        return summarize_2d_mismatches(mismatches, desired, actual)
    else:
        # For other dimensions
        return summarize_nd_mismatches(mismatches, desired, actual)


def summarize_2d_mismatches(mismatches, desired, actual):
    """Summarize mismatches in 2D arrays as contiguous regions, sorted by size."""
    total_mismatches = np.sum(mismatches)

    if total_mismatches == 0:
        return "No mismatches found."

    if total_mismatches == 1:
        row, col = np.where(mismatches)
        r, c = row[0], col[0]
        return f"Only element [{r}, {c}] is wrong.\n  Desired: {desired[r, c]}\n  Actual:  {actual[r, c]}"

    # Find contiguous regions
    region_info = []  # Will store (size, r_start, c_start, r_end, c_end) tuples
    rows, cols = mismatches.shape

    # Process the array to find rectangular regions
    visited = np.zeros_like(mismatches, dtype=bool)

    for r in range(rows):
        for c in range(cols):
            if mismatches[r, c] and not visited[r, c]:
                # Find the largest rectangle starting at (r,c)
                max_r, max_c = r, c

                # Extend rows
                while max_r + 1 < rows and mismatches[max_r + 1, c]:
                    max_r += 1

                # Find the maximum width for this range of rows
                width = 1
                while c + width < cols:
                    can_extend = True
                    for row_idx in range(r, max_r + 1):
                        if not mismatches[row_idx, c + width]:
                            can_extend = False
                            break
                    if can_extend:
                        width += 1
                    else:
                        break

                # Mark this region as visited
                visited[r : max_r + 1, c : c + width] = True

                # Calculate region size
                region_size = (max_r - r + 1) * width

                # Add region info: (size, r_start, c_start, r_end, c_end)
                region_info.append((region_size, r, c, max_r, c + width - 1))

    # Sort regions by size (descending) and then by coordinates (ascending)
    # For ties in size, sort by row_start, then col_start
    region_info.sort(key=lambda x: (-x[0], x[1], x[2]))

    # Format region strings with values for top regions
    region_strings = []
    num_regions_with_values = min(5, len(region_info))  # Show values for top 5 regions

    for i, (size, r_start, c_start, r_end, c_end) in enumerate(region_info):
        # Only display top 10 regions if there are more than 10
        if i >= 10 and len(region_info) > 10:
            remaining = len(region_info) - 10
            region_strings.append(f"... {remaining} more regions not shown")
            break

        if r_start == r_end and c_start == c_end:
            region_str = f"\nRegion {i+1}: [{r_start}, {c_start}] (size: {size})"
            if i < num_regions_with_values:
                region_str += f"\n  Desired: {desired[r_start, c_start]}"
                region_str += f"\n  Actual:  {actual[r_start, c_start]}"
        else:
            region_str = f"\nRegion {i+1}: [{r_start}:{r_end+1}, {c_start}:{c_end+1}] (size: {size})"
            if i < num_regions_with_values:
                # Extract the region
                desired_region = desired[r_start : r_end + 1, c_start : c_end + 1]
                actual_region = actual[r_start : r_end + 1, c_start : c_end + 1]

                # Format the values - show a sample if the region is large
                height = r_end - r_start + 1
                width = c_end - c_start + 1

                if height <= 3 and width <= 4:
                    # Show full region if small
                    region_str += "\n  Desired:\n" + _format_array(desired_region, "    ")
                    region_str += "\n  Actual:\n" + _format_array(actual_region, "    ")
                else:
                    # Show a sample for large regions
                    region_str += f" (showing sample of {height}x{width} region)"
                    sample_h = min(3, height)
                    sample_w = min(4, width)

                    # Show top-left corner
                    desired_sample = desired_region[:sample_h, :sample_w]
                    actual_sample = actual_region[:sample_h, :sample_w]

                    region_str += "\n  Desired (top-left corner):\n" + _format_array(
                        desired_sample, "    ", show_ellipsis=(height > 3 or width > 4)
                    )
                    region_str += "\n  Actual (top-left corner):\n" + _format_array(
                        actual_sample, "    ", show_ellipsis=(height > 3 or width > 4)
                    )

        region_strings.append(region_str)

    if len(region_strings) == 1:
        return f"Elements {region_strings[0]} are wrong."
    else:
        total_regions = len(region_info)
        header = f"Found {total_regions} mismatched regions, sorted by size (largest first):"
        return f"{header}\n" + "".join(region_strings)


def _format_array(arr, indent="", show_ellipsis=False):
    """Format a numpy array for display with proper indentation."""
    with np.printoptions(precision=6, suppress=True, threshold=100):
        lines = str(arr).split("\n")
        formatted = []
        for i, line in enumerate(lines):
            if i == 0:
                formatted.append(indent + line)
            else:
                formatted.append(indent + line)
        result = "\n".join(formatted)
        if show_ellipsis:
            result = result.replace("]", ", ...]")
        return result


def summarize_nd_mismatches(mismatches, desired, actual):
    """Handle mismatches in arrays with dimensions other than 2."""
    total_mismatches = np.sum(mismatches)
    if total_mismatches == 1:
        coords = np.where(mismatches)
        coord_str = ", ".join(str(dim[0]) for dim in coords)
        idx = tuple(dim[0] for dim in coords)
        return f"Only element [{coord_str}] is wrong.\n  Desired: {desired[idx]}\n  Actual:  {actual[idx]}"

    # For higher dimensions, just report the total and some examples with values
    coords = np.where(mismatches)
    # Get up to 5 examples
    examples = []
    for i in range(min(5, total_mismatches)):
        coord_list = [str(dim[i]) for dim in coords]
        idx = tuple(dim[i] for dim in coords)
        example = f"\n  [{', '.join(coord_list)}]: Desired={desired[idx]}, Actual={actual[idx]}"
        examples.append(example)

    example_str = "".join(examples)
    if total_mismatches > 5:
        example_str += f"\n  ... {total_mismatches - 5} more mismatches not shown"

    return f"Found {total_mismatches} mismatches. Examples:{example_str}"
