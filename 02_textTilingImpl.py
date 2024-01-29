import argparse
import logging
import math
import sys
import statistics
import string

from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def read_csv(file_name):
    sentences = []

    with open(f"./{file_name}", "r") as f:
        file_text = f.read()
        file_text = file_text.strip().split("\n")
        for sentence in file_text:
            punctuation = string.punctuation + r"""”“’—"""
            sentence_stripped = sentence.lower().translate(
                str.maketrans("", "", punctuation)
            )
            sentences.append(sentence_stripped)

    return sentences


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


def score_block_comparison(sentence_list, block_size=5):
    lexical_gap_scores = []

    for i in range(len(sentence_list) - block_size):
        block1 = sentence_list[i : (i + block_size)]
        block1_counter = Counter()

        for sentence in block1:
            block1_counter += Counter(sentence.split())

        block2 = sentence_list[(i + 1) : (i + block_size + 1)]
        block2_counter = Counter()

        for sentence in block2:
            block2_counter += Counter(sentence.split())

        block1_weight = sum(block1_counter.values())
        block2_weight = sum(block2_counter.values())

        numerator = block1_weight + block2_weight
        denominator = math.sqrt((block1_weight**2) * (block2_weight**2))

        lexical_gap_scores.append(numerator / denominator)

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
        if score > (average - (standard_dev / 2)):
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
    parser = argparse.ArgumentParser(description="argument parser for text tiling")
    parser.add_argument("-f", "--file", required=True, help="file to parse")
    parser.add_argument(
        "-sc", "--score", required=True, help="scoring type, `block` or `vocab`"
    )
    parser.add_argument(
        "-m", "--model", required=False, help="embedding_model", default="multi-qa-MiniLM-L6-cos-v1"
    )
    parser.add_argument(
        "--smooth", required=False, help="smoothing toggle", action="store_true"
    )
    parser.add_argument(
        "--print",
        required=False,
        help="print output one line at a time for copy paste",
        action="store_true",
    )

    args = parser.parse_args()
    logger.info(f"Reading in file: {args.file}")

    sentences = read_csv(args.file)
    logger.info(f"File length: {len(sentences)}")

    sentences = preprocess(sentences)
    # logger.info(sentences)

    if args.score == "block":
        logger.info("scoring with block method")
        gap_scores = score_block_comparison(sentences)
    elif args.score == "vocab":
        logger.info("scoring with vocab introduction method")
        gap_scores = score_vocabulary_introduction(sentences)
    elif args.score == "embedding":
        logger.info("scoring with embedding similarity method")
        multi_model = SentenceTransformer(args.model)
        gap_scores = score_embedding_similarity(sentences, multi_model)

    gap_scores = boundary_identification(gap_scores)

    if args.smooth:
        gap_scores = smoothing(gap_scores)

    paragraph_boundaries = paragraph_numbers(gap_scores)

    output = translate_gaps_to_sentences(paragraph_boundaries)
    print(f"OUTPUT: {output}")
    if args.print:
        for item in output:
            print(item)
