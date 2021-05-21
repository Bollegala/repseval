"""
This is a helper script for converting published datasets to a format required by repseval.

Danushka Bollegala
6th Feb 2019
"""

def process(input_fname, output_fname):
    with open(input_fname) as infile:
        with open(output_fname, 'w') as outfile:
            for line in infile:
                p = line.strip().split(',')
                outfile.write("%s\t%s\t%s\n" % (p[0], p[1], p[2]))

if __name__ == '__main__':
    process("../benchmarks/MTURK-771.csv", "../benchmarks/mtruk-771-pairs")

