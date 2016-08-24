#! /usr/bin/python -u
"""
Perform evaluations of the word representations using three analogy datasets:
Mikolov (Google + MSRA), SAT, and SemEval.
and various semantic similarity datasets such as WS353, RG, MC, SCWC, RW, MEN.
"""

__author__ = "Danushka Bollegala"
__licence__ = "BSD"
__version__ = "1.0"

import numpy
import scipy.stats
import sys
import collections
import argparse
import os

pkg_dir = os.path.dirname(os.path.abspath(__file__))

VERBOSE = False


class WordReps:

    def __init__(self):
        self.vocab = None 
        self.vects = None 
        self.vector_size = None
        pass


    def read_model(self, fname, dim, words=None, HEADER=False,):
        """
        Read the word vectors where the first token is the word.
        """
        res = {}
        F = open(fname)
        if HEADER:
            res["method"] = F.readline().split('=')[1].strip()
            res["input"] = F.readline().split('=')[1].strip()
            res["rank"] = int(F.readline().split('=')[1])
            res["itermax"] = int(F.readline().split('=')[1])
            res["vertices"] = int(F.readline().split('=')[1])
            res["edges"] = int(F.readline().split('=')[1])
            res["labels"] = int(F.readline().split('=')[1])
            R = res["rank"]
        R = dim
        # read the vectors.
        vects = {}
        vocab = []
        line = F.readline()
        while len(line) != 0:
            p = line.split()
            word = p[0]
            if words is None or word in words:
                v = numpy.zeros(R, float)
                for i in range(0, R):
                    v[i] = float(p[i+1])
                vects[word] = normalize(v)
                vocab.append(word)
            line = F.readline()
        F.close()
        self.vocab = vocab
        self.vects = vects
        self.dim = R
        pass


    def read_w2v_model_text(self, fname, dim):
        """
        Read the word vectors where the first token is the word.
        """
        F = open(fname)
        R = dim
        # read the vectors.
        vects = {}
        vocab = []
        line = F.readline()  # vocab size and dimensionality 
        assert(int(line.split()[1]) == R)
        line = F.readline()
        while len(line) != 0:
            p = line.split()
            word = p[0]
            v = numpy.zeros(R, float)
            for i in range(0, R):
                v[i] = float(p[i+1])
            vects[word] = normalize(v)
            vocab.append(word)
            line = F.readline()
        F.close()
        self.vocab = vocab
        self.vects = vects
        self.dim = R
        pass


    def read_w2v_model_binary(self, fname, dim):
        """
        Given a model file (fname) produced by word2vect, read the vocabulary list 
        and the vectors for each word. We will return a dictionary of the form
        h[word] = numpy.array of dimensionality.
        """
        F = open(fname, 'rb')
        header = F.readline()
        vocab_size, vector_size = map(int, header.split())
        vocab = []
        vects = {}
        print "Vocabulary size =", vocab_size
        print "Vector size =", vector_size
        assert(dim == vector_size)
        binary_len = numpy.dtype(numpy.float32).itemsize * vector_size
        for line_number in xrange(vocab_size):
            # mixed text and binary: read text first, then binary
            word = ''
            while True:
                ch = F.read(1)
                if ch == ' ':
                    break
                if ch != '\n':
                    word += ch
            word = word.lower()
            vocab.append(word)
            vector = numpy.fromstring(F.read(binary_len), numpy.float32)
            vects[word] = vector        
        F.close()
        self.vocab = vocab
        self.vects = vects
        self.dim = vector_size
        pass


    def get_vect(self, word):
        if word not in self.vocab:
            return numpy.zeros(self.vector_size, float)
        return self.vects[word]


    def normalize_all(self):
        """
        L2 normalizes all vectors.
        """
        for word in self.vocab:
            self.vects[word] = normalize(self.vects[word])
        pass


    def test_model(self):
        A = self.get_vect("man")
        B = self.get_vect("king")
        C = self.get_vect("woman")
        D = self.get_vect("queen")
        x = B - A + C
        print cosine(x, D)
        pass   


def cosine(x, y):
    """
    Compute the cosine similarity between two vectors x and y. 
    We must L2 normalize x and y before we use this function.
    """
    #return numpy.dot(x,y.T) / (numpy.linalg.norm(x) * numpy.linalg.norm(y))
    norm = numpy.linalg.norm(x) * numpy.linalg.norm(y)
    return 0 if norm == 0 else (numpy.dot(x, y) / norm)


def normalize(x):
    """
    L2 normalize vector x. 
    """
    norm_x = numpy.linalg.norm(x)
    return x if norm_x == 0 else (x / norm_x)


def get_embedding(word, WR):
    """
    If we can find the embedding for the word in vects, we will return it.
    Otherwise, we will check if the lowercase version of word appears in vects
    and if so we will return the embedding for the lowercase version of the word.
    Otherwise we will return a zero vector.
    """
    if word in WR.vects:
        return WR.vects[word]
    elif word.lower() in WR.vects:
        return WR.vects[word.lower()]
    else:
        return numpy.zeros(WR.dim, dtype=float)


def eval_SemEval(WR, method):
    """
    Answer SemEval questions. 
    """
    from semeval import SemEval
    S = SemEval(os.path.join(pkg_dir, "../benchmarks/semeval"))
    total_accuracy = 0
    print "Total no. of instances in SemEval =", len(S.data)
    for Q in S.data:
        scores = []
        for (first, second) in Q["wpairs"]:
            val = 0
            for (p_first, p_second) in Q["paradigms"]:
                #if (first in WR.vects) and (second in WR.vects) and (p_first in WR.vects) and (p_second in WR.vects):
                if 1:
                    # print first, second, p_first, p_second
                    va = get_embedding(first, WR)
                    vb = get_embedding(second, WR)
                    vc = get_embedding(p_first, WR)
                    vd = get_embedding(p_second, WR)
                    val += scoring_formula(va, vb, vc, vd, method)
            val /= float(len(Q["paradigms"]))
            scores.append(((first, second), val))
        # sort the scores and write to a file. 
        scores.sort(lambda x, y: -1 if x[1] > y[1] else 1)
        score_fname = os.path.join(pkg_dir, "../work/semeval/%s.txt" % Q["filename"])
        score_file = open(score_fname, 'w')
        for ((first, second), score) in scores:
            score_file.write('%f "%s:%s"\n' % (score, first, second))
        score_file.close()
        total_accuracy += S.get_accuracy(score_fname, Q["filename"])
    acc = total_accuracy / float(len(S.data))
    print "SemEval Accuracy =", acc
    return {"acc": acc}



def eval_SAT_Analogies(WR, method):
    """
    Solve SAT word analogy questions using the vectors. 
    """
    from sat import SAT
    S = SAT()
    questions = S.getQuestions()
    corrects = total = skipped = 0
    for Q in questions:
        total += 1
        (q_first, q_second) = Q['QUESTION']
        if q_first['word'] in WR.vects and q_second['word'] in WR.vects:
            va = get_embedding(q_first['word'], WR)
            vb = get_embedding(q_second['word'], WR)
            max_sim = -100
            max_cand = -100
            for (i, (c_first, c_second)) in enumerate(Q["CHOICES"]):
                sim = 0
                if c_first['word'] in WR.vects and c_second['word'] in WR.vects:
                    vc = get_embedding(c_first['word'], WR)
                    vd = get_embedding(c_second['word'], WR)
                    sim = scoring_formula(va, vb, vc, vd, method)
                    # print q_first['word'], q_second['word'], c_first['word'], c_second['word'], sim
                    # sim = numpy.random.random()
                    if max_sim < sim:
                        max_sim = sim 
                        max_cand = i
            if max_cand == Q['ANS']:
                corrects += 1
                # print "CORRECT:", q_first['word'], q_second['word'], c_first['word'], c_second['word'], sim
        else:
            skipped += 1
    acc = float(100 * corrects) / float(total)
    coverage = 100.0 - (float(100 * skipped) / float(total))
    print "SAT Accuracy = %f (%d / %d)" % (acc, corrects, total)
    print "Qustion coverage = %f (skipped = %d)" % (coverage, skipped) 
    return {"acc":acc, "coverage":coverage}


def eval_Google_Analogies(WR, method):
    """
    Evaluate the accuracy of the learnt vectors on the analogy task. 
    We consider the set of fourth words in the test dataset as the
    candidate space for the correct answer.
    """
    analogy_file = open(os.path.join(pkg_dir, "../benchmarks/google-analogies.txt"))
    cands = []
    questions = collections.OrderedDict()
    total_questions = {}
    corrects = {}
    while 1:
        line = analogy_file.readline()
        if len(line) == 0:
            break
        if line.startswith(':'):  # This is a label 
            label = line.split(':')[1].strip()
            questions[label] = []
            total_questions[label] = 0
            corrects[label] = 0
        else:
            p = line.strip().split()
            total_questions[label] += 1
            questions[label].append((p[0], p[1], p[2], p[3]))
            if p[3] not in cands:
                cands.append(p[3])

            #if (p[0] in vects) and (p[1] in vects) and (p[2] in vects) and (p[3] in vects):
            #    questions[label].append((p[0], p[1], p[2], p[3]))
            #if (p[3] in vects) and (p[3] not in cands):
            #    cands.append(p[3])

    analogy_file.close()
    valid_questions = sum([len(questions[label]) for label in questions])
    print "== Google Analogy Dataset =="
    print "Total no. of question types =", len(questions) 
    print "Total no. of candidates =", len(cands)
    print "Valid questions =", valid_questions
    
    # predict the fourth word for each question.
    count = 1
    for label in questions:
        for (a,b,c,d) in questions[label]:
            if count % 100 == 0:
                print "%d%% (%d / %d)" % ((100 * count) / float(valid_questions), count, valid_questions), "\r", 
            count += 1
            # set of candidates for the current question are the fourth
            # words in all questions, except the three words for the current question.
            scores = []
            va = get_embedding(a, WR)
            vb = get_embedding(b, WR)
            vc = get_embedding(c, WR)
            for cand in cands:
                if cand not in [a,b,c]:
                    y = get_embedding(cand, WR)
                    scores.append((cand, scoring_formula(va, vb, vc, y, method)))

            scores.sort(lambda p, q: -1 if p[1] > q[1] else 1)
            if scores[0][0] == d:
                corrects[label] += 1
    
    # Compute accuracy
    n = semantic_total = syntactic_total = semantic_corrects = syntactic_corrects = 0
    for label in total_questions:
        n += total_questions[label]
        if label.startswith("gram"):
            syntactic_total += total_questions[label]
            syntactic_corrects += corrects[label]
        else:
            semantic_total += total_questions[label]
            semantic_corrects += corrects[label]
    print "Percentage of questions attempted = %f (%d / %d)" % ((100 * valid_questions) /float(n), valid_questions, n)
    for label in questions:
        acc = float(100 * corrects[label]) / float(total_questions[label])
        print "%s = %f (correct = %d, attempted = %d, total = %d)" % (
            label, acc, corrects[label], len(questions[label]), total_questions[label])
    semantic_accuracy = float(100 * semantic_corrects) / float(semantic_total)
    syntactic_accuracy = float(100 * syntactic_corrects) / float(syntactic_total)
    total_corrects = semantic_corrects + syntactic_corrects
    accuracy = float(100 * total_corrects) / float(n)
    print "Semantic Accuracy =", semantic_accuracy 
    print "Syntactic Accuracy =", syntactic_accuracy
    print "Total accuracy =", accuracy
    return {"semantic": semantic_accuracy, "syntactic":syntactic_accuracy, "total":accuracy}


def eval_MSR_Analogies(WR, method):
    """
    Evaluate the accuracy of the learnt vectors on the analogy task using MSR dataset. 
    We consider the set of fourth words in the test dataset as the
    candidate space for the correct answer.
    """
    analogy_file = open(os.path.join(pkg_dir, "../benchmarks/msr-analogies.txt"))
    cands = []
    questions = []
    total_questions = 0
    corrects = 0
    while 1:
        line = analogy_file.readline()
        if len(line) == 0:
            break
        p = line.strip().split()
        total_questions += 1
        questions.append((p[0], p[1], p[2], p[3]))
        if p[3] not in cands:
            cands.append(p[3])

            #if (p[0] in vects) and (p[1] in vects) and (p[2] in vects) and (p[3] in vects):
            #    questions[label].append((p[0], p[1], p[2], p[3]))
            #if (p[3] in vects) and (p[3] not in cands):
            #    cands.append(p[3])

    analogy_file.close()
    print "== MSR Analogy Dataset =="
    print "Total no. of questions =", len(questions)
    print "Total no. of candidates =", len(cands)
    
    # predict the fourth word for each question.
    count = 1
    for (a,b,c,d) in questions:
        if count % 100 == 0:
            print "%d / %d" % (count, len(questions)), "\r", 
        count += 1
        # set of candidates for the current question are the fourth
        # words in all questions, except the three words for the current question.
        scores = []
        va = get_embedding(a, WR)
        vb = get_embedding(b, WR)
        vc = get_embedding(c, WR)
        for cand in cands:
            if cand not in [a,b,c]:
                y = get_embedding(cand, WR)
                scores.append((cand, scoring_formula(va, vb, vc, y, method)))

        scores.sort(lambda p, q: -1 if p[1] > q[1] else 1)
        if scores[0][0] == d:
            corrects += 1
    accuracy = float(corrects) / float(len(questions))
    print "MSR accuracy =", accuracy
    return {"accuracy": accuracy}


############### SCORING FORMULAS ###################################################
def scoring_formula(va, vb, vc, vd, method):
    """
    Call different scoring formulas. 
    """
    t = numpy.copy(vb)
    vb = vc
    vc = t

    if method == "CosSub":
        return subt_cos(va, vb, vc, vd)
    elif method == "PairDiff":
        return PairDiff(va, vb, vc, vd)
    elif method == "CosMult":
        return mult_cos(va, vb, vc, vd)
    elif method == "CosAdd":
        return add_cos(va, vb, vc, vd)
    elif method == "DomFunc":
        return domain_funct(va, vb, vc, vd)
    elif method == "EleMult":
        return elementwise_multiplication(va, vb, vc, vd)
    else:
        raise ValueError


def mult_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    log(cos(vb, vd)) + log(cos(vc,vd)) - log(cos(va,vd))
    """
    first = (1.0 + cosine(vb, vd)) / 2.0
    second = (1.0 + cosine(vc, vd)) / 2.0
    third = (1.0 + cosine(va, vd)) / 2.0
    score = numpy.log(first) + numpy.log(second) - numpy.log(third)
    return score


def add_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(vb - va + vc, vd)
    """
    x = normalize(vb - va + vc)
    return cosine(x, vd)


def domain_funct(va, vb, vc, vd):
    """
    Uses the Formula proposed by Turney in Domain and Function paper.
    """
    return numpy.sqrt((1.0 + cosine(va, vc))/2.0 * (1.0 + cosine(vb, vd))/2.0)


def elementwise_multiplication(va, vb, vc, vd):
    """
    Represent the first word-pair by the elementwise multiplication of va and vb.
    Do the same for vc and vd. Finally measure the cosine similarity between the
    two resultant vectors.
    """
    return cosine(va * vb, vc * vd)


def subt_cos(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(va - vc, vb - vd)
    """
    return cosine(normalize(va - vc), normalize(vb - vd))


def PairDiff(va, vb, vc, vd):
    """
    Uses the following formula for scoring:
    cos(vd - vc, vb - va)
    """
    return cosine(normalize(vd - vc), normalize(vb - va))
####################################################################################



def batch_process_analogy(model_fname, dim, output_fname):
    res_file = open(output_fname, 'w')
    res_file.write("# Method, semantic, syntactic, all, SAT, SemEval, MSR\n")
    methods = ["CosAdd", "CosMult", "CosSub", "PairDiff"]
    settings = [(model_fname, dim)]
    words = set()
    with open("../benchmarks/all_words.txt") as F:
        for line in F:
            words.add(line.strip())
    for (model, dim) in settings:
        WR = WordReps()
        WR.read_model(model, dim, words)
        for method in methods:
            print model, dim, method
            res_file.write("%s+%s, " % (model, method))
            res_file.flush()
            Google_res = eval_Google_Analogies(WR.vects, method)
            res_file.write("%f, %f, %f, " % (Google_res["semantic"], Google_res["syntactic"], Google_res["total"]))
            res_file.flush()
            SAT_res = eval_SAT_Analogies(WR.vects, method)
            res_file.write("%f, " % SAT_res["acc"])
            res_file.flush()
            SemEval_res = eval_SemEval(WR.vects, method)
            res_file.write("%f, " % SemEval_res["acc"])
            res_file.flush()
            MSR_res = eval_MSR_Analogies(WR.vects, method)
            res_file.write("%f\n" % MSR_res["accuracy"])
    res_file.close()
    pass


def get_words_in_benchmarks():
    """
    Get the set of words in benchmarks.
    """
    words = set()
    benchmarks = ["ws", "rg", "mc", "rw", "scws", "men", "simlex"]
    for bench in benchmarks:
        with open("../benchmarks/%s_pairs.txt" % bench) as F:
            for line in F:
                p = line.strip().split()
                words.add(p[0])
                words.add(p[1])

    # Get words in Google analogies.
    analogy_file = open("../benchmarks/google-analogies.txt")
    while 1:
        line = analogy_file.readline()
        if len(line) == 0:
            break
        if line.startswith(':'):  # This is a label 
            label = line.split(':')[1].strip()
        else:
            p = line.strip().split()
            words.add(p[0])
            words.add(p[1])
            words.add(p[2])
            words.add(p[3])
    analogy_file.close()

    # Get words in MSR analogies.
    analogy_file = open("../benchmarks/msr-analogies.txt")
    while 1:
        line = analogy_file.readline()
        p = line.strip().split()
        if len(p) == 0:
            break
        words.add(p[0])
        words.add(p[1])
        words.add(p[2])
        words.add(p[3])
    analogy_file.close()

    # Get SAT words.
    from sat import SAT
    S = SAT()
    questions = S.getQuestions()
    for Q in questions:
        (q_first, q_second) = Q['QUESTION']
        words.add(q_first['word'])
        words.add(q_second['word'])
        for (i, (c_first, c_second)) in enumerate(Q["CHOICES"]):
            words.add(c_first['word'])
            words.add(c_second['word'])

    # Get SemEval words.
    from semeval import SemEval
    S = SemEval("../benchmarks/semeval")
    for Q in S.data:
        for (first, second) in Q["wpairs"]:
            words.add(first)
            words.add(second)
            for (p_first, p_second) in Q["paradigms"]:
                words.add(p_first)
                words.add(p_second)

    with open("../benchmarks/all_words.txt", 'w') as G:
        for word in words:
            G.write("%s\n" % word)
    pass


def get_correlation(dataset_fname, vects, corr_measure):
    """
    Measure the cosine similarities for words in the dataset using their representations 
    given in vects. Next, compute the correlation coefficient. Specify method form
    spearman and pearson.
    """
    ignore_missing = False
    global VERBOSE
    if VERBOSE:
        if ignore_missing:
            sys.stderr.write("Ignoring missing pairs\n")
        else:
            sys.stderr.write("Not ignoring missing pairs\n")
    mcFile = open(dataset_fname)
    mcPairs = {}
    mcWords = set()
    for line in mcFile:
        p = line.strip().split()
        mcPairs[(p[0], p[1])] = float(p[2])
        mcWords.add(p[0])
        mcWords.add(p[1])
    mcFile.close()
    #print "Total no. of unique words in the dataset =", len(mcWords)
    found = mcWords.intersection(set(vects.keys()))
    #print "Total no. of words found =", len(found)
    missing = []
    for x in mcWords:
        if x not in vects:
            missing.append(x)
    human = []
    computed = []
    found_pairs = False
    missing_count = 0
    for wp in mcPairs:
        (x, y) = wp
        if (x in missing or y in missing):
            missing_count += 1
            if ignore_missing:
                continue
            else:
                comp = 0
        else:
            found_pairs = True
            comp = cosine(vects[x], vects[y])
        rating = mcPairs[wp]
        human.append(rating) 
        computed.append(comp)       
        #print "%s, %s, %f, %f" % (x, y, rating, comp)
    if VERBOSE:    
        sys.stderr.write("Missing pairs = %d (out of %d)\n" % (missing_count, len(mcPairs)))

    if found_pairs is False:
        #print "No pairs were scored!"
        return (0, 0)
    if corr_measure == "pearson":
        return scipy.stats.pearsonr(computed, human)
    elif corr_measure == "spearman":
        return scipy.stats.spearmanr(computed, human)
    else:
        raise ValueError
    pass


def batch_process_lexical(model_fname, dim, output_fname):
    """
    This function shows how to evaluate a trained word representations
    on semantic similarity benchmarks.
    """
    WR = WordReps()
    # We will load vectors only for the words in the benchmarks.
    words = set()
    with open("../benchmarks/all_words.txt") as F:
        for line in F:
            words.add(line.strip())
    WR.read_model(model_fname, dim, words)
    benchmarks = ["ws", "rg", "mc", "rw", "scws", "men", "simlex"]
    output_file = open(output_fname, 'w')
    output_file.write("# Benchmark, Spearman, Significance\n")
    for bench in benchmarks:
        (corr, sig) = get_correlation("../benchmarks/%s_pairs.txt" % bench, WR.vects, "spearman")
        output_file.write("%s, %s, %s\n" % (bench, str(corr), str(sig)))
        print "Benchmark = %s,\t Spearman = %s,\t Significance = %s" % (bench, str(corr), str(sig))
    pass


def evaluate_embeddings(embed_fname, dim):
    """
    This function can be used to evaluate an embedding.
    """
    res = {}
    WR = WordReps()
    # We will load vectors only for the words in the benchmarks.
    words = set()
    with open(os.path.join(pkg_dir, "../benchmarks/all_words.txt")) as F:
        for line in F:
            words.add(line.strip())
    WR.read_model(embed_fname, dim, words)

    # semantic similarity benchmarks.
    benchmarks = ["ws", "rg", "mc", "rw", "scws", "men", "simlex"]  
    for bench in benchmarks:
        (corr, sig) = get_correlation(os.path.join(pkg_dir, "../benchmarks/%s_pairs.txt" % bench), WR.vects, "spearman")
        res[bench] = corr

    # word analogy benchmarks.
    scoring_method = "CosMult"
    res["scoring_method"] = scoring_method
    res["Google_res"] = eval_Google_Analogies(WR, scoring_method)
    #res["SAT_res"] = eval_SAT_Analogies(WR, scoring_method)
    res["MSR_res"] = eval_MSR_Analogies(WR, scoring_method)
    res["SemEval_res"] = eval_SemEval(WR, scoring_method)

    res_file = open("../work/res.csv", 'w')
    res_file.write("#RG, MC, WS, RW, SCWS, MEN, SimLex, sem, syn, total, SemEval, MSR\n")
    res_file.write("%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % (res["rg"], res["mc"], res["ws"], res["rw"], res["scws"], 
            res["men"], res["simlex"], res["Google_res"]["semantic"], res["Google_res"]["syntactic"], 
            res["Google_res"]["total"], res["SemEval_res"]["acc"], res["MSR_res"]["accuracy"]))
    res_file.close()
    return res


def main():
    """
    Catch the arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate pre-trained word representation on semantic similarity and word analogy tasks.")
    parser.add_argument("-mode", type=str, help="specify the mode of evaluation. 'lex' for semantic similarity evaluation, 'ana' for word analogy evaluation. If none specified we will evaluate both.")
    parser.add_argument("-dim", type=int, help="specify the dimensionality of the word representations as an integer.")
    parser.add_argument("-input", type=str, help="specify the input file from which to read word representations.")
    parser.add_argument("-output", type=str, help="specify the csv formatted output file to which the evaluation result to be written.")
    parser.add_argument("-verbose", action="store_true", help="if set, we will display debug info.")
    args = parser.parse_args()
    global VERBOSE
    if args.verbose:
        VERBOSE = True
    if args.mode.lower() == "lex":
        batch_process_lexical(args.input, args.dim, args.output) 
    elif args.mode.lower() == "ana":
        batch_process_analogy(args.input, args.dim, args.output)
    elif args.mode.lower() == "full":
        evaluate_embeddings(args.input, args.dim)
    else:
        print parser.print_help()
        sys.stderr.write("Invalid option for mode. It must be either lex or ana\n")
    pass


if __name__ == "__main__":
    main()
    #get_words_in_benchmarks()
    #evaluate_embeddings("../../../../embeddings/glove.42B.300d.txt", 300)
    #evaluate_embeddings("../../../work/glove+sg.concat", 600)
    #evaluate_embeddings("../../../work/glove+sg.coemb", 300)
    
   
