import pandas as pd
import re

def parse_dct(dct_path):
    variables = []
    widths = []
    with open(dct_path, 'r') as f:
        for line in f:
            # Match lines like: "str15     caseid    %15s"
            match = re.match(r'\s*\w+\s+(\w+)\s+%(\d+)[a-z]', line)
            if match:
                var = match.group(1)
                width = int(match.group(2))
                variables.append(var)
                widths.append(width)
    return variables, widths

def parse_do_labels(do_path):
    labels = {}
    with open(do_path, 'r') as f:
        for line in f:
            # Match lines like: label variable caseid   "Case Identification"
            match = re.match(r'label variable (\w+)\s+"(.+)"', line)
            if match:
                labels[match.group(1)] = match.group(2)
    return labels

def convert_dat_to_csv(dat_path, dct_path, do_path, output_csv):
    variables, widths = parse_dct(dct_path)
    labels = parse_do_labels(do_path)
    # Read the fixed-width file
    df = pd.read_fwf(dat_path, widths=widths, names=variables)
    # Optionally, rename columns to labels
    df.rename(columns=labels, inplace=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")

# Example usage:
dct_file = "C:\\Users\\acer\\Desktop\\maternal mortality\\maternalMotality (2)\\maternalMotality\\NPGR82FL.DCT"
do_file = "C:\\Users\\acer\\Desktop\\maternal mortality\\maternalMotality (2)\\maternalMotality\\NPGR82FL.DO"
dat_file = "C:\\Users\\acer\\Desktop\\maternal mortality\\maternalMotality (2)\\maternalMotality\\NPGR82FL.DAT"
output_csv = "C:\\Users\\acer\\Desktop\\maternal mortality\\maternalMotality (2)\\maternalMotality\\NPGR82FL_output.csv"

convert_dat_to_csv(dat_file, dct_file, do_file, output_csv)