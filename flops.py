import re


def count_flops(line):
    # Lowercase to catch everything case-insensitive
    line_lower = line.lower()

    # Count operations with regex for word boundaries (avoid partial matches)
    adds = len(re.findall(r"\+ ", line))  # + signs, simple but effective
    adds += len(re.findall(r" -", line))  # + signs, simple but effective
    adds += len(re.findall(r"\(-", line))  # + signs, simple but effective
    mults = len(re.findall(r" \* ", line))
    divs = len(re.findall(r" / ", line))

    sqrt_count = len(re.findall(r"(?<!r)sqrt\(", line_lower))
    rsqrt_count = len(re.findall(r"rsqrt", line_lower))
    fma_count = len(re.findall(r"fma", line_lower))

    if "dotV3" in line:
        adds += 2
        mults += 3

    if "divV3" in line:
        divs += 3

    if "subV3" in line:
        adds += 3

    if "crossV3" in line:
        mults += 6
        adds += 3

    if "_mm256_mul_pd" in line:
        mults += 4
    if "_mm256_add_pd" in line:
        adds += 4
    if "_mm256_div_pd" in line:
        divs += 4
    if "_mm256_sqrt_pd" in line:
        sqrt_count += 4
    if "_mm256_fmadd_pd" in line:
        fma_count += 4
    if "_mm256_hadd_pd" in line:
        adds += 4

    # Adjust counts based on your rules
    divs += rsqrt_count  # rsqrt counts as one division
    sqrt_count += rsqrt_count  # and one square root

    adds += fma_count  # each FMA counts as one add
    mults += fma_count  # and one mult

    return adds, mults, divs, sqrt_count


def process_c_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    output_lines = []

    exports_started = False

    sadds = 0
    smults = 0
    sdivs = 0
    ssqrts = 0

    for line in lines:
        if "FLOPS" in line and exports_started:
            # Skip lines that already have FLOPS info
            continue

        if "DLL_EXPORT" in line:
            exports_started = True

        # Count operations on the line
        adds, mults, divs, sqrts = count_flops(re.split(r"//", line, 1)[0].rstrip())

        # Prepare the FLOPS annotation line
        flops_line = f"FLOPS({adds}, {mults}, {divs}, {sqrts}, complete_function);\n"

        # Insert the FLOPS annotation above the original line

        output_lines.append(line)

        if exports_started and "DLL_EXPORT" not in line:
            if ";" not in line:
                sadds += adds
                smults += mults
                sdivs += divs
                ssqrts += sqrts
            else:
                if adds + mults + divs + sqrts > 0:
                    output_lines.append(flops_line)

                if sadds + smults + sdivs + ssqrts > 0:
                    flops_line = f"FLOPS({sadds}, {smults}, {sdivs}, {ssqrts}, complete_function);\n"
                    output_lines.append(flops_line)
                    sadds = 0
                    smults = 0
                    sdivs = 0
                    ssqrts = 0

    return output_lines


def main():
    import sys

    if len(sys.argv) != 2:
        print("Usage: python flops_counter.py <filename.c>")
        sys.exit(1)

    filename = sys.argv[1]
    output_lines = process_c_file(filename)

    # Overwrite the original file or create a new one
    with open("test.c", "w") as f:
        f.writelines(output_lines)

    print(f"Processed file '{filename}' with FLOPS annotations added.")


if __name__ == "__main__":
    main()
