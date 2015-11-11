# README #

## Requirements ##
- Python 2.7 or above
- numpy
- scipy

## Execution ##

### Evaluating using Semantic Similarity Benchmarks ###

To evaluate on semantic similarity benchmarks, go to the src directory and execute
```
python eval.py -m lex -d noOfDimensions -i wordRepsFile -o result.csv
```

* -m option specifies the mode of operation and 'lex' indicates that we will be performing evaluations on semantic similarity benchmark datasets. 

* -d option is used to specify the dimensionality of the word representations.

* -i specifies the input file from which we will read word representations. The format of this file is as follows.
Each line represents the word vector for a particular word. First element in each line is the word and subsequent elements
(in total the number of columns corresponding to the dimensionality specified using the -d option) contains the value of
each dimension of the representation.

* -o is the name of the output file into which we will write the Pearson correlation coefficients and their significance values.
This is a csv file.

### Evaluating using Word Analogy Benchmarks ###

To evaluate on word analogy benchmarks, go to the src directory and execute
```
python eval.py -m ana -d noOfDimensions -i wordRepsFile -o result.csv
```

* -m option specifies the mode of operation. In this case it must be 'ana' (representing word analogy benchmarks). 

* We will use several benchmarks such as
    * SAT (accuracy of SAT questions correctly answered)
    * Google dataset (Mikolov et al. 2013 Google dataset consisting of semantic and syntactic analogies). Accuracy is used as the evaluation measure for (man, king), (woman, ?) type of proportional analogy questions, where queen is the correct answer in this case. The system must rank queen as the top result among all other candidates.
    * [SemEval 2012 Task 2 (https://sites.google.com/site/semeval2012task2/)] (MaxDiff is used as the evaluation measure)