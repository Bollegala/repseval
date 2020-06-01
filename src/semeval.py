"""
This module implements the necessary functions to load and evaluate 
on the SemEval-2012 Task 2 dataset (https://sites.google.com/site/semeval2012task2/)

Danushka Bollegala
23/12/2014
"""

import string
import subprocess
import os

class SemEval:

    def __init__(self, pkg_dir):
        #self.data_path = os.path.join(pkg_dir, "../benchmarks/semeval")
        self.data_path = pkg_dir
        self.load_dataset()
        pass


    def load_dataset(self):
        """
        Load the word-pairs for each relation. 
        """
        # Load sub-categories and paradigms.
        self.data = []
        subcat_file = open("%s/../benchmarks/semeval/subcategories-paradigms.txt" % self.data_path)
        for line in subcat_file:
            relation = {}
            p = [x.strip() for x in line.strip().split(',')]
            relation["filename"] = "%s%s" % (p[0], p[1])
            relation["category"] = p[2]
            relation["sub-category"] = p[3]
            relation["paradigms"] = [tuple(x.split(':')) for x in p[4:]]
            self.data.append(relation)
        subcat_file.close()
        # load word pairs for each relation.
        for Q in self.data:
            wpair_file = open("%s/../benchmarks/semeval/Phase1Answers/Phase1Answers-%s.txt" % (self.data_path, Q["filename"]))
            Q["wpairs"] = [tuple([x.strip() for x in line.strip().replace('"', '').split(':')]) for line in wpair_file]
            wpair_file.close()
        pass


    def get_accuracy(self, fname, file_id):
        """
        Evaluate the result. 
        """
        acc = None
        #print("sh %s/semeval.sh %s %s %s/../benchmarks/semeval > /dev/null" % (self.data_path, fname, file_id, self.data_path))
        #subprocess.call("sh %s/semeval.sh %s %s %s > /dev/null" % (self.pkg_dir, fname, file_id, self.data_path), shell=True)

        subprocess.call("sh %s/semeval.sh %s %s %s > /dev/null" % (self.data_path, fname, file_id, self.data_path), shell=True)
        F = open("%s/../work/semeval-tmp/MaxDiffFinal-%s.txt" % (self.data_path, file_id))
        for line in F:
            if line.startswith("Overall Accuracy:"):
                acc = float(line.strip().split(':')[1].split('%')[0])
        F.close()
        if acc is None:
            raise("Could not read accuracy from file = %s" % fname, ValueError)
        return acc


def process():
    S = SemEval("../benchmarks/semeval")
    print(S.data[0]["wpairs"])
    pass


if __name__ == "__main__":
    process()



