import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lf', '--load_file', type=str, default='updated_iris.csv',
                        help='Specify file to load from input file')
    parser.add_argument('-p1', '--part_1', action='store_true',
                        help='Part 1 in PDF: Perform Data Cleansing using Class Means on the data '
                             'and output to output folder')
    parser.add_argument('-p2', '--part_2', action='store_true',
                        help='Part 2 in PDF: Perform Feature Generation using PCA on the data'
                             ' and output to output folder')
    parser.add_argument('-p3', '--part_3', action='store_true',
                        help='Part 3 in PDF: Perform Feature Preprocessing using Confidence Interval method on the data'
                             ' and output to output folder')
    parser.add_argument('-p4', '--part_4', action='store_true',
                        help='Part 4 in PDF: Perform Feature Ranking using Fisher LDA on the data'
                             ' and output to output folder')
    parser.add_argument('-p5', '--part_5', action='store_true',
                        help='Part 5 in PDF: Perform Dimensionality Reduction using PCA on the data'
                             ' and output to output folder')
    parser.add_argument('-p6a', '--part_6a', action='store_true',
                        help='Part 6a in PDF: Perform Expectation Maximization on the data'
                             ' and output to output folder')
    parser.add_argument('-p6b', '--part_6b', action='store_true',
                        help='Part 6b in PDF: Perform Fisher Linear Discriminant on the data'
                             ' and output to output folder')

    args = parser.parse_args()

    return args
