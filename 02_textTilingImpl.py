import argparse
import logging
import math
import sys
import statistics
import string
import csv
import numpy as np
from pathlib import Path

from collections import Counter
import nltk
from nltk.metrics import windowdiff
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
# from sentence_transformers import SentenceTransformer, util
nltk.download("stopwords")
print("okay0", file=sys.stderr)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
print("okay1", file=sys.stderr)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)
print("okay2", file=sys.stderr)

def read_csv(file_name):
    print(file_name)
    sentences = []
    labels = []

    with open(f"./{file_name}", "r") as f:
        data = []
        reader = csv.reader(f)
        for line in reader:
            data.append(line)
        for line in data:
            if len(line) == 2:
                sentence, label = line
                labels.append(label)
            else:
                sentence = line[0]
            punctuation = string.punctuation + r"""”“’—"""
            sentence_stripped = sentence.lower().translate(
                str.maketrans("", "", punctuation)
            )
            sentences.append(sentence_stripped)

    return sentences, labels


def preprocess(sentence_list):
    english_stopwords = stopwords.words("english")
    # lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("english")
    sentences = []
    for sentence in sentence_list:
        temp_sentence = ""
        for word in sentence.split():
            if word in english_stopwords:
                pass
            else:
                # word = lemmatizer.lemmatize(word)
                word = stemmer.stem(word)
                if len(temp_sentence) == 0:
                    temp_sentence = word
                else:
                    temp_sentence = f"{temp_sentence} {word}"
        sentences.append(temp_sentence)

    return sentences


def score_block_comparison(sentence_list, block_size=2):
    lexical_gap_scores = []

    if block_size < 2:
        block_size = 2
        print("BLOCK SIZE must be at least 2", file=sys.stderr)
    
    for i in range(len(sentence_list) - 1): #this loop will hit on every possible sentence gap as a boundary, while shifting the blocks

        if i < (block_size - 1): #if it's near the beginning of the list, where block 1 will be shortened
            # print(i,file=sys.stderr)
            block1 = []
            for j in range(0,i+1):
                block1.append(sentence_list[j])
            block2 = []
            for k in range(1, block_size+1):
                block2.append(sentence_list[i+k])

        elif i < len(sentence_list) - block_size: #if it's near the middle of the list, 2 equal blocks
            # print(i, file=sys.stderr)
            block1 = []
            for j in range(0, block_size):
                block1.append(sentence_list[i-j])


            block2 = []
            for k in range(1, block_size+1):
                block2.append(sentence_list[i + k])

        else: #if its near the end of the list, block 2 will be shortened
            # print(i, file=sys.stderr)
            block1 = []
            for j in range(0, block_size):
                block1.append(sentence_list[i - j])

            block2 = []
            for sent in sentence_list[i+1:]:
                block2.append(sent)

        block1_counter = Counter()
        block2_counter = Counter()
        # print(block1, block2, file=sys.stderr)
        #add our words from block1 into block1_counter, words from block2 into block2_counter
        for sentence in block1:
            block1_counter += Counter(sentence.split())
        for sentence in block2:
            block2_counter += Counter(sentence.split())

        # make sure all the keys are in both counters
        # we need vectors of same length to take the inner product of the counts
        for key in block1_counter.keys():
            if key not in block2_counter:
                block2_counter[key] = 0

        for key in block2_counter.keys():
            if key not in block1_counter:
                block1_counter[key] = 0

        #numerator is the dot product of the two vectors
        inner_product = sum(block1_counter[key] * block2_counter[key] for key in block1_counter.keys())

        #denominator is the square root of multiplying the two normalizations together
        #had to add 1 or else the stopword removal can break the lexical overlap if there are no words in a sentence
        block1_norm = sum([x ** 2 for x in block1_counter.values()])
        block2_norm = sum([x ** 2 for x in block2_counter.values()])
        # print(block1_norm, block2_norm, file=sys.stderr)
        denominator = math.sqrt(block1_norm * block2_norm)

        lexical_gap_scores.append(inner_product / denominator)

        # old code
        # block1_weight = sum(block1_counter.values())
        # block2_weight = sum(block2_counter.values())
        #
        # numerator = block1_weight + block2_weight
        # denominator = math.sqrt((block1_weight ** 2) * (block2_weight ** 2))
        #
        # lexical_gap_scores.append(numerator / denominator)


    # for i in range(len(sentence_list))
    #
    #
    #     block1_counter = Counter()
    #     block2_counter = Counter()
    #
    #     for sentence in block1:
    #         block1_counter += Counter(sentence.split())
    #     for sentence in block2:
    #         block2_counter += Counter(sentence.split())
    #
    #     block1_weight = sum(block1_counter.values())
    #     block2_weight = sum(block2_counter.values())
    #
    #     numerator = block1_weight + block2_weight
    #     denominator = math.sqrt((block1_weight**2) * (block2_weight**2))
    #
    #     lexical_gap_scores.append(numerator / denominator)

    return lexical_gap_scores


def score_vocabulary_introduction(sentence_list):
    lexical_gap_scores = []
    new_words_a = set()
    new_words_b = set()

    for i in range(len(sentence_list) - 1):
        sent1 = sentence_list[i]
        sent2 = sentence_list[i + 1]
        sent1_new = 0
        sent2_new = 0

        if i == 0:
            new_words_b = set(sent1.split())

        for word in sent1.split():
            if word not in new_words_a:
                sent1_new += 1
                new_words_a.add(word)

        for word in sent2.split():
            if word not in new_words_b:
                sent2_new += 1
                new_words_b.add(word)

        score = (sent1_new + sent2_new) / (len(sent1) + len(sent2))
        lexical_gap_scores.append(score)
        i += 1

    return lexical_gap_scores


def score_embedding_similarity(sentence_list, model):
    embedding_gap_scores = []

    for i in range(len(sentence_list) - 1):
        sent1 = sentence_list[i]
        sent2 = sentence_list[i + 1]
        sent1_embedding = model.encode(sent1)
        sent2_embedding = model.encode(sent2)

        similarity = float(util.dot_score(sent1_embedding, sent2_embedding))
        embedding_gap_scores.append(similarity)
        print("Similarity:", similarity)

    return embedding_gap_scores

def score_embedding_similarity_blocked(sentence_list, model, block_size = 2):
    print("BLOCK SIZE", block_size, file=sys.stderr)
    embedding_gap_scores = []

    for i in range(len(sentence_list) - 1): #this loop will hit on every possible sentence gap as a boundary, while shifting the blocks

        if i < (block_size - 1): #if it's near the beginning of the list, where block 1 will be shortened
            # print(i,file=sys.stderr)
            block1 = []
            for j in range(0,i+1):
                block1.append(sentence_list[j])
            block2 = []
            for k in range(1, block_size+1):
                block2.append(sentence_list[i+k])

        elif i < len(sentence_list) - block_size: #if it's near the middle of the list, 2 equal blocks
            # print(i, file=sys.stderr)
            block1 = []
            for j in range(0, block_size):
                block1.append(sentence_list[i-j])


            block2 = []
            for k in range(1, block_size+1):
                block2.append(sentence_list[i + k])

        else: #if its near the end of the list, block 2 will be shortened
            # print(i, file=sys.stderr)
            block1 = []
            for j in range(0, block_size):
                block1.append(sentence_list[i - j])

            block2 = []
            for sent in sentence_list[i+1:]:
                block2.append(sent)

        #concatenate blocks into string to feed to model
        block1 = " ".join(block1)
        block2 = " ".join(block2)
        sent1_embedding = model.encode(block1)
        sent2_embedding = model.encode(block2)

        similarity = float(util.dot_score(sent1_embedding, sent2_embedding))
        embedding_gap_scores.append(similarity)
        print("Similarity:", similarity)

    return embedding_gap_scores

def boundary_identification(lexical_gap_scores):
    depth_scores = []

    for i in range(len(lexical_gap_scores)):
        if i == 0:
            depth_score = (0 - lexical_gap_scores[i]) + (
                lexical_gap_scores[i + 1] - lexical_gap_scores[i]
            )
        elif i == (len(lexical_gap_scores) - 1):
            depth_score = (lexical_gap_scores[i - 1] - lexical_gap_scores[i]) + (
                0 - lexical_gap_scores[i]
            )
        else:
            depth_score = (lexical_gap_scores[i - 1] - lexical_gap_scores[i]) + (
                lexical_gap_scores[i + 1] - lexical_gap_scores[i]
            )
        depth_scores.append(depth_score)

    return depth_scores


def smoothing(depth_scores):
    smoothed_scores = [depth_scores[0]]
    for i in range(len(depth_scores)):
        if (len(depth_scores) - 2) >= i >= 1:
            smoothed_score = (
                depth_scores[i - 1] + depth_scores[i] + depth_scores[i + 1]
            ) / 3
            smoothed_scores.append(smoothed_score)
        elif i == (len(depth_scores) - 1):
            smoothed_scores.append(depth_scores[-1])

    return smoothed_scores


def paragraph_numbers(depth_scores):
    paragraph_starts = []
    average = sum(depth_scores) / len(depth_scores)
    standard_dev = statistics.stdev(depth_scores)

    for score in depth_scores:
        if score > (average + (standard_dev / 2)):
            paragraph_starts.append(1)
        else:
            paragraph_starts.append(0)

    return paragraph_starts


def translate_gaps_to_sentences(scores):
    i = 1
    para_nums = [i]
    for score in scores:
        if score > 0:
            i += 1
        para_nums.append(i)
    return para_nums


if __name__ == "__main__":
    print("Hello", file=sys.stderr)
    parser = argparse.ArgumentParser(description="argument parser for text tiling")
    parser.add_argument("-f", "--file", required=False, help="file to parse, each line: sent, label")
    parser.add_argument(
        "-sc", "--score", required=True, help="scoring type, `block` or `vocab` or `embedding`"
    )
    parser.add_argument(
        "-m", "--model", required=False, help="embedding_model", default="multi-qa-MiniLM-L6-cos-v1"
    )
    parser.add_argument(
        "--smooth", required=False, help="smoothing toggle", action="store_true"
    )
    parser.add_argument(
        "--eval", required=False, help="windowdiff eval", action="store_true"
    )
    parser.add_argument(
        "--print",
        required=False,
        help="print output one line at a time for copy paste",
        action="store_true",
    )
    parser.add_argument("--folder")

    parser.add_argument("-nb", "--num_blocks", help="number of blocks for lexical overlap score", default=2)

    args = parser.parse_args()
    if args.file:
        logger.info(f"Reading in file: {args.file}")
    else:
        logger.info(f"Folder: {args.folder}")

    if args.folder:
        pathlist = Path(args.folder).rglob('*.csv')
        for path in pathlist:
            path_in_str = str(path)
            sentences, labels = read_csv(path_in_str)
    if args.file:
        sentences, labels = read_csv(args.file)

    logger.info(f"File length: {len(sentences)}")

    sentences = preprocess(sentences)
    # logger.info(sentences)

    if args.score == "block":
        logger.info("scoring with block method")
        gap_scores = score_block_comparison(sentences, block_size = int(args.num_blocks))
    elif args.score == "vocab":
        logger.info("scoring with vocab introduction method")
        gap_scores = score_vocabulary_introduction(sentences)
    elif args.score == "embedding":
        logger.info("scoring with embedding similarity method")
        multi_model = SentenceTransformer(args.model)
        gap_scores = score_embedding_similarity_blocked(sentences, multi_model, block_size = int(args.num_blocks))

    gap_scores = boundary_identification(gap_scores)

    if args.smooth:
        gap_scores = smoothing(gap_scores)

    paragraph_boundaries = paragraph_numbers(gap_scores)

    output = translate_gaps_to_sentences(paragraph_boundaries)

    # convert output to 0,1 format (1 indicates the start of a subtopic)
    windowdiff_tiles="1"
    for i in range(len(output)-1):
        current_score = output[i]
        next_score = output[i+1]
        if current_score == next_score:
            windowdiff_tiles += "0"
        else:
            windowdiff_tiles += "1"

    print("System Tiles:",windowdiff_tiles,file=sys.stderr)

    if args.eval:
        label_string = ''.join(labels)
        print("Gold Tiles:", label_string)
        print("Windowdiff Score:", windowdiff(labels, windowdiff_tiles, 3))
    

    print(f"OUTPUT: {windowdiff_tiles}")
    if args.print:
        for item in windowdiff_tiles:
            print(item)