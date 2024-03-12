"""
Main file to utilize the indices in indices folder and create a search engine
for the given queries.
"""

from string import punctuation
import math # for log applications.
import argparse
import pandas as pd

def set_arguments() -> argparse.ArgumentParser:
    """This function will set the arguments are return a parser object.

    Returns:
        argparse.ArgumentParser: A parser object from which arguments can be
                                extracted.
    """

    parser = argparse.ArgumentParser(
        prog="Urdu Indexer Scoring and Evaluation",
        description="You can apply scoring function on your indexer and evaluate them",
    )
    parser.add_argument(
        "--score",
        metavar="SCORING_FUNC",
        help="Pass a scoring function from given list of choices: [tfidf, bm25, dirichlet]",
        type=str,
        choices=["tfidf", "bm25", "dirichlet"]
        )
    parser.add_argument(
        "--evaluate",
        help="Pass this argument to notify the program to run the evaluator",
        action="store_true" # Used to store true if evaluate is passed.
    )

    return parser

def read_query_rel(filename : str) -> dict[int, list[dict[int, int], int]]:
    """This file reads the query relavance file and and returns a dictionary
       with the information in that file.

    Args:
        filename (str): the filename containing query relavance information.

    Returns:
        dict[int, list[dict[int, int], int]]: return object in the shape:
        {
            query_id : [{docid : relevancy_score}, total_rel_docs]
        }
        relevancy_score will be mapped between 0 and 1,
        with mapping scheme 1-2 : 0, 3-4 : 1
    """

    rel_dict = {}
    qrel_dict = pd.read_excel(filename, sheet_name=None) # Reading all sheets in the form of a dict.

    for sheet, sheet_df in qrel_dict.items(): # Looping over every dataframe to extract data.
        print(f"Processing sheet: {sheet}")
        topicid = sheet_df.loc[sheet_df.index[0], "Topic Id"] # The query/topic ID.
        topicid = int(topicid)
        total_reldocs = 0 # Count for total relevant docs.
        rel_dict[topicid] = [{}, 0] # Setting the tuple object.
        for idx in sheet_df.index:
            docid = sheet_df.loc[idx, "Doc Id"] # Doc ID
            docid = int(docid)
            relevancy = sheet_df.loc[idx, "Doc Relevancy"] # relevancy b/w 1-4
            relevancy = 0 if (int(relevancy) <= 2) else 1
            # This will increment if document is relevant only, as relevant docs have value 1.
            total_reldocs += relevancy
            # Setting the relevancy score for each topicid and docid.
            rel_dict[topicid][0][docid] = relevancy
        rel_dict[topicid][1] = total_reldocs # Setting the relevance value.

    return rel_dict

def get_queries(filename : str) -> list[str]:
    """Function that reads queries from the given filename and returns them.

    Args:
        filename (str): the file containing the queries separated by newline.

    Returns:
        list[str]: list of queries.
    """

    queries = [] # list of queries.
    with open(filename, "r", encoding="utf-8") as _f:
        for line in _f.readlines(): # looping over every line.
            # appending after stripping to remove any trailing spaces or newline characters.
            queries.append(line.strip())

    return queries

def preprocess_string(query : str, stopword_file : str) -> list[str]:
    """This function will preprocess the given query and return its tokens.

    Args:
        query (str): the given query to preprocess.
        stopword_file (str): The file containing newline separated stopwords.

    Returns:
        list[str]: list of tokens
    """

    stopwords = []
    with open(stopword_file, encoding="utf-8") as _f: # opening file containing stopwords.
        stopwords.extend(
            [stopword.strip() for stopword in _f] # Reading each line in the file and stripping it.
        )

    # Replacing \n with ' ' so that we can tokenize easily.
    content = query.replace("\n", " ")
    # Removing all english characters.
    # Adding urdu punctuation marks.
    urdu_punc = "؛،٫؟۔“”’‘٪" + ','
    all_punc = urdu_punc
    # Replacing all punctuations with space.
    for punc in all_punc:
        content = content.replace(punc, " ")
    # Tokenizing on spaces and \t as well.
    tokens = content.strip().replace("\t", " ").split(" ")

    tokens = [token for token in tokens if token and token not in stopwords] # Removing all empty strings as tokens.

    return tokens

def get_ids(filename : str) -> dict[str, int]:
    """This function is used to read ids from the given filename.

    Args:
        filename (str): The filename containing the ids(either for doc or term).

    Returns:
        dict[str, int]: Resultant dictionary containing the following pair:
                        {term/doc : ID}
    """
    with open(filename, encoding='utf-8') as _f:
        # Reading each line.
        id_list = {}
        for line in _f.readlines():
            # As the file structure is ID\tOBJECT, parsing in this format.
            # Removing all prevailing \n with replace.
            try:
                # List comprehension is removing useless \t, as it is splitting and
                # then removing empty strings.
                obj_id, obj = [s for s in line.replace("\n", "").split("\t") if s]
                # Setting the dictionary.
                id_list[obj] = int(obj_id)
            except ValueError: # Case if the line doesn't have both id and its respective term/doc.
                pass

    return id_list

def get_term_info(filename : str) -> dict[int, dict]:
    """This function is used to read term info from the given filename

    Args:
        filename (str): The filename containing term info.

    Returns:
        dict[int, dict]: The resultant object with the following key-value pair:
                        {
                            termID : {
                                "Offset" : file_offset,
                                "Total" : total_count,
                                "Total Docs" : doc_count
                            }
                        }
    """
    with open(filename, encoding="utf-8") as _f:
        # Reading each line.
        term_info = {}
        for line in _f.readlines():
            # As the file file structure is ID\tOFFSET\tTOTAL\tTOTALDOC,
            # parsing in this format.
            # Removing all prevailing \n with strip
            term_id, line_offset, total_count, doc_count = line.strip().split('\t')
            # Setting the dictionary.
            term_info[int(term_id)] = {
                "Offset" : int(line_offset),
                "Total" : int(total_count),
                "Total Docs" : int(doc_count)
            }

    return term_info

def get_index_entry(filename : str, line_offset : int) -> dict[int, list]:
    """This function grabs the index entry from the inverted index file at that given offset.

    Args:
        filename (str): The file containing the inverted index.
        line_offset (int): The offset to start reading from

    Returns:
        dict[int, list]: The returning object in the following format:
                        {docid : [list of postings]}
    """
    index_entry = {}
    with open(filename, encoding="utf-8") as _f:
        # Jumping to that offset.
        _f.seek(line_offset)
        # Reading the desired termID line.
        line = _f.readline()
        # Ignoring the first split on \t as that is termid.
        _, *postings = line.strip().split("\t")
        # Storing the first entry as is, as it is not delta encoded.
        p_docid, position = postings[0].split(":")
        index_entry[int(p_docid)] = [int(position)]
        # Previous docid for delta decoding, and previous posting for decoding as well.
        prev_docid = int(p_docid)
        prev_posting = int(position)
        for posting in postings[1:]:
            p_docid, position = posting.split(":")
            p_docid, position = int(p_docid), int(position)
            # Docid, hence position was delta encoded.
            if p_docid == 0:
                position += prev_posting
            # Setting the docid key if it doesn't already exist.
            p_docid += prev_docid
            index_entry.setdefault(p_docid , [])
            index_entry[p_docid].append(position)
            # Set previous for the current position and p_docid
            prev_docid, prev_posting = p_docid, position

    return index_entry

def get_doc_lengths(doc_entry_file : str) -> dict[int, int]:
    """This function will return each docids' total tokens.

    Args:
        doc_entry_file (str): The file containing the forward index.

    Returns:
        dict[int, int]: The returning object in the form:
        {
            docid : total tokens.
        }
    """

    doc_lengths = {} # Returning object.

    with open(doc_entry_file, encoding="utf-8") as _f:
        # Looping over every line until desired docid is found.
        for line in _f:
            curr_docid, _, *curr_postings = line.strip().split("\t")
            curr_docid = int(curr_docid)
            try: # If curr_docid is already a key in doc_lengths, this try will run.
                doc_lengths[curr_docid] += len(curr_postings)
            except KeyError: # curr_docid doesn't exist as a key.
                doc_lengths[curr_docid] = len(curr_postings)

    return doc_lengths

def tfidf_scorer(query_tokens : list[str], termids : dict[str, int],
                docids : dict[str, int], term_info : dict[int, dict],
                inverted_idx_file : str) -> dict[int, float]:
    """This function will return the query's tfidf score with each document in docids.

    Args:
        query_tokens (list[str]): The preprocessed tokens of the query to find tfidf score with 
                                  each document in docids.
        termids (dict[str, int]): The termids dictionary, in the format {term : id}
        docids (dict[str, int]): Dictionary with docids, format: {doc : id}
        term_info (dict[int, dict]): the term information dictionary in the format:
        {
            termID : {
                "Offset" : file_offset,
                "Total" : total_count,
                "Total Docs" : doc_count
            }
        }
        inverted_idx_file (str): The file containing the inverted index.

    Returns:
        dict[int, float]: returning object in the format:
        {
            docid : tfidf score
        }
    """
    uniq_tokens = list(set(query_tokens)) # Extracting all unique tokens.
    qtokens_tf = [] # Term frequency vector for the query.
    total_docs = len(docids)

    for token in uniq_tokens:
        token_count = query_tokens.count(token) # Finding the count of the token.
        qtokens_tf.append(1 + math.log10(token_count)) # log normalizing and appending the count.

    doc_vector = {} # Dictionary for the document tfidf vector.
    doc_tfidf = {} # Dictionary of each doc's tfidf score, the variable to return.

    for docid in docids.values(): # Loop to populte doc_vector
        docid = int(docid)
        # Populating each vector with 0s equal to the length of query's tf vector.
        # These 0s will be replaced with tfidf value if the token exists in the document,
        # Otherwise can stay 0, keeping the vector's lengths same.
        doc_vector[docid] = [0] * len(qtokens_tf)

        doc_tfidf[docid] = 0 # Initialize with 0, tfidf score will sum here.

    for i, token in enumerate(uniq_tokens):
        try:
            token_id = termids[token]
        except KeyError: # In case token doesn't exist in the termids dictionary
            print(f"Out of Vocabulary word spotted: {token}")
            continue
        token_info = term_info[token_id] # Getting the token info.
        offset = token_info["Offset"]
        index_entry = get_index_entry(inverted_idx_file, offset) # Getting the index entry.

        for docid, postings in index_entry.items():
            # Looping over each returned docid to populate its document vector.
            # Postings is the log normalized count of the term in that document.
            _tf = 1 + math.log10(len(postings))
            # Finding the idf for the term.
            idf = 1 + math.log10((total_docs / token_info["Total Docs"]))
            doc_vector[docid][i] = _tf * idf # Updating the tf-idf value for that document.

            # Multiplying the tf of query with tfidf of document and accumulating it. This
            # step can also be performed in an external loop.
            doc_tfidf[docid] += (doc_vector[docid][i] * qtokens_tf[i])


    return doc_tfidf

def bm25_scorer(query_tokens : list[str], termids : dict[str, int],
                docids : dict[str, int], term_info : dict[int, dict],
                inverted_idx_file : str, doc_lengths : dict[int, int], total_tokens : int,
                bm25_k1 : float, bm25_k2 : int, bm25_b : float) -> dict[int, float]:
    """This function will return the query's Okapi BM25 score with each document in docids.

    Args:
        query_tokens (list[str]): The preprocessed tokens of the query to find Okapi BM25 score with 
                                  each document in docids.
        termids (dict[str, int]): The termids dictionary, in the format {term : id}
        docids (dict[str, int]): Dictionary with docids, format: {doc : id}
        term_info (dict[int, dict]): the term information dictionary in the format:
        {
            termID : {
                "Offset" : file_offset,
                "Total" : total_count,
                "Total Docs" : doc_count
            }
        }
        inverted_idx_file (str): The file containing the inverted index.
        doc_lengths (dict[int, int]): The total terms in each docid.
        total_tokens (int) : Total tokens in the entire corpus.
        bm25_k1 (float): BM25 constant.
        bm25_k2 (int): BM25 constant.
        bm25_b (float): BM25 constant.

    Returns:
        dict[int, float]: returning object in the format:
        {
            docid : Okapi BM25 score
        }
    """
    uniq_tokens = list(set(query_tokens))
    total_docs = len(docids)

    bm25_score = {}

    for docid in docids.values():
        bm25_score[docid] = 0 # Initialize each docid with 0 score.

    for token in uniq_tokens:
        try:
            token_id = termids[token]
        except KeyError: # In case token doesn't exist in the termids dictionary
            print(f"Out of Vocabulary word spotted: {token}")
            continue
        token_info = term_info[token_id] # Getting the token info.
        offset = token_info["Offset"]
        index_entry = get_index_entry(inverted_idx_file, offset) # Getting the index entry.

        for docid, postings in index_entry.items():
            # Calculating the entire score for the term as per the formula
            bm25_K = bm25_k1 * (
                (1 - bm25_b) + bm25_b * (
                    doc_lengths[docid] / (total_tokens / total_docs)
                )
            )

            score = math.log10(
                                (total_docs + 0.5) / (token_info["Total Docs"] + 0.5)
                                )
            score *= (
                ((1 + bm25_k1) * len(postings)) / (bm25_K + len(postings))
            )

            score *= (
                ((1 + bm25_k2) * query_tokens.count(token)) / (
                    bm25_k2 + query_tokens.count(token)
                )
            )

            bm25_score[docid] += score # Accumulating the score in the resulting dictionary.


    return bm25_score

def dirichlet_scorer(query_tokens : list[str], termids : dict[str, int],
                docids : dict[str, int], term_info : dict[int, dict],
                inverted_idx_file : str, doc_lengths : dict[int, int], total_tokens : int):
    """This function will apply dirichlet smoothing and return the query
       tokens' summed probabilities for scoring.

    Args:
        query_tokens (list[str]): The preprocessed tokens of the query to find Okapi BM25 score with 
                                  each document in docids.
        termids (dict[str, int]): The termids dictionary, in the format {term : id}
        docids (dict[str, int]): Dictionary with docids, format: {doc : id}
        term_info (dict[int, dict]): the term information dictionary in the format:
        {
            termID : {
                "Offset" : file_offset,
                "Total" : total_count,
                "Total Docs" : doc_count
            }
        }
        inverted_idx_file (str): The file containing the inverted index.
        doc_lengths (dict[int, int]): The total terms in each docid.
        total_tokens (int) : Total tokens in the entire corpus.
    """
    total_docs = len(docids)
    dirichlet_score = {}
    _mu = total_tokens / total_docs

    for docid in docids.values(): # Looping for every docid.
        _n = doc_lengths[docid]
        dirichlet_score[docid] = 0 # Default value.
        for token in query_tokens: # Looping for every token.
            try:
                token_id = termids[token]
            except KeyError: # In case token doesn't exist in the termids dictionary
                print(f"Out of Vocabulary word spotted: {token}")
                continue
            token_info = term_info[token_id] # Getting the token info.
            offset = token_info["Offset"]
            index_entry = get_index_entry(inverted_idx_file, offset) # Getting the index entry.

            # Finding the Maximum Likelihood Estimate
            try:
                score = (_n / (_n + _mu)) * (len(index_entry[docid]) / _n)
                score += (_mu / (_n + _mu)) * (token_info["Total"] / total_tokens)

                dirichlet_score[docid] += score
            except KeyError: # The docid has no entry for the given term.
                pass


    return dirichlet_score

def run_scorer(scorer : str):
    """Function to run the provided scorer and display ranked documents for each query.

    Args:
        scorer (str): The scorer to run, among the choices: [tfidf, bm25, dirichlet]
    """

    # FILENAMES
    queries_file = "topics.txt"
    stopwords_file = "Urdu stopwords.txt"
    termid_file = "indices/termids.txt"
    docid_file = "indices/docids.txt"
    term_info_file = "indices/term_info.txt"
    inverted_idx_file = "indices/term_index.txt"
    forward_index_file = "indices/doc_index.txt"

    queries = get_queries(queries_file)
    query_tokens = [] # A 2d list containing each query's tokens.

    print("Processing Queries...")
    for query in queries: # Preprocessing each query and appending them to a list.
        query_tokens.append(preprocess_string(query, stopwords_file))

    print("Loading metadata for the indexer...")
    termids = get_ids(termid_file)
    docids = get_ids(docid_file)
    term_info = get_term_info(term_info_file)
    print("Loading Document lengths...")
    doc_lengths = get_doc_lengths(forward_index_file)
    print("Document Lengths loaded...")
    # Getting the total term count for the corpus
    # which should be the total sum of all docids' lengths.
    total_terms = sum(doc_lengths.values())

    inv_docids = {} # Dictionary mapping of docid to doc name. Used in scorer.
    for docname, docid in docids.items():
        inv_docids[docid] = docname

    if scorer == 'tfidf':
        queries_tfidf = {} # Dictionary for each query's tfidf
        tfidf_ranking = [] # Dictionary to output ranking to a file.
        for query_id, query_token in enumerate(query_tokens, 1):
            tfidf = tfidf_scorer(query_token, termids, docids, term_info, inverted_idx_file)
            queries_tfidf[query_id] = tfidf

        ranked_docs = {} # Documents ranked by their score.
        for query_id, query_docs in queries_tfidf.items():
            ranked_docs[query_id] = list(query_docs.keys()) # Storing all the keys.
            # Sorting the keys on the basis of their values.
            ranked_docs[query_id].sort(key=lambda k : query_docs[k], reverse=True)

        print("TF-IDF RANKING: ")
        for query_id, ranking in ranked_docs.items():
            for rank, docid in enumerate(ranking, 1):
                # If the scorer returned 0 for current doc, don't print it.
                if queries_tfidf[query_id][docid]:
                    print(f"{query_id}\t{inv_docids[docid]}\t"\
                        f"{rank}\t{queries_tfidf[query_id][docid]}\t  run1")
                tfidf_ranking.append(f"{query_id}\t{inv_docids[docid]}\t"\
                        f"{rank}\t{queries_tfidf[query_id][docid]}\t  run1")

        # Writing the output ranking to a file.
        with open("TF-IDF Ranking.txt", "w", encoding="utf-8") as _f:
            _f.write('\n'.join(tfidf_ranking))
    elif scorer == 'bm25':
        queries_bm25 = {} # Dictionary for each query's BM25
        bm25_ranking = []
        ### BM25 Constants
        bm25_k1 = 1.2
        bm25_k2 = 1000
        bm25_b = 0.75

        for query_id, query_token in enumerate(query_tokens, 1):
            bm25 = bm25_scorer(query_token, termids, docids, term_info,
                            inverted_idx_file, doc_lengths, total_terms, bm25_k1, bm25_k2, bm25_b)
            queries_bm25[query_id] = bm25

        ranked_docs = {} # Documents ranked by their score.
        for query_id, query_docs in queries_bm25.items():
            ranked_docs[query_id] = list(query_docs.keys()) # Storing all the keys.
            # Sorting the keys on the basis of their values.
            ranked_docs[query_id].sort(key=lambda k : query_docs[k], reverse=True)

        print("Okapi BM25 RANKING: ")
        for query_id, ranking in ranked_docs.items():
            for rank, docid in enumerate(ranking, 1):

                if queries_bm25[query_id][docid]:
                    print(f"{query_id}\t{inv_docids[docid]}\t"\
                        f"{rank}\t{queries_bm25[query_id][docid]}\trun1")
                bm25_ranking.append(f"{query_id}\t{inv_docids[docid]}\t"\
                        f"{rank}\t{queries_bm25[query_id][docid]}\trun1")

        # Writing the output ranking to a file.
        with open("Okapi BM25 Ranking.txt", "w", encoding="utf-8") as _f:
            _f.write('\n'.join(bm25_ranking))

    elif scorer == 'dirichlet':
        queries_dirichlet = {} # Dictionary for each query's Dirichlet Smoothing LM
        dirichlet_ranking = []

        for query_id, query_token in enumerate(query_tokens, 1):
            print(f"Running Dirichlet Scorer for Query {query_id}")
            dirichlet = dirichlet_scorer(query_token, termids, docids, term_info,
                            inverted_idx_file, doc_lengths, total_terms)
            queries_dirichlet[query_id] = dirichlet

        ranked_docs = {} # Documents ranked by their score.
        for query_id, query_docs in queries_dirichlet.items():
            ranked_docs[query_id] = list(query_docs.keys()) # Storing all the keys.
            # Sorting the keys on the basis of their values.
            ranked_docs[query_id].sort(key=lambda k : query_docs[k], reverse=True)

        print("Language Model using Dirichlet Smoothing RANKING: ")
        for query_id, ranking in ranked_docs.items():
            for rank, docid in enumerate(ranking, 1):

                if queries_dirichlet[query_id][docid]:
                    print(f"{query_id}\t{inv_docids[docid]}\t"\
                        f"{rank}\t{queries_dirichlet[query_id][docid]}\trun1")

                dirichlet_ranking.append(f"{query_id}\t{inv_docids[docid]}\t"\
                        f"{rank}\t{queries_dirichlet[query_id][docid]}\trun1")

        # Writing the output ranking to a file.
        with open("LM with Dirichlet Ranking.txt", "w", encoding="utf-8") as _f:
            _f.write('\n'.join(dirichlet_ranking))

    else:
        print(f"Wrong scorer function passed: {scorer}, "\
              "please choose among: [tfidf, bm25, dirichlet]")

def evaluator():
    """This function will be invoked if --evaluate is called.
       It will evaluate all 3 scoring methods and generate a report table.
    """
    # FILENAMES
    qrel_filename = "qrels.xlsx"
    queries_file = "topics.txt"
    stopwords_file = "Urdu stopwords.txt"
    termid_file = "indices/termids.txt"
    docid_file = "indices/docids.txt"
    term_info_file = "indices/term_info.txt"
    inverted_idx_file = "indices/term_index.txt"
    forward_index_file = "indices/doc_index.txt"
    bm25_k1 = 1.2
    bm25_k2 = 1000
    bm25_b = 0.75

    # VARIABLES USED.
    rel_dict = read_query_rel(qrel_filename)
    queries = get_queries(queries_file)
    query_tokens = [] # A 2d list containing each query's tokens.

    print("Processing Queries...")
    for query in queries: # Preprocessing each query and appending them to a list.
        query_tokens.append(preprocess_string(query, stopwords_file))

    print("Loading metadata for the indexer...")
    termids = get_ids(termid_file)
    docids = get_ids(docid_file)
    term_info = get_term_info(term_info_file)
    print("Loading Document lengths...")
    doc_lengths = get_doc_lengths(forward_index_file)
    print("Document Lengths loaded...")
    # Getting the total term count for the corpus
    # which should be the total sum of all docids' lengths.
    total_terms = sum(doc_lengths.values())

    queries_tfidf = {} # Dictionary for each query's tfidf
    queries_bm25 = {} # Dictionary for each query's BM25
    queries_dirichlet = {} # Dictionary for each query's Dirichlet Smoothing LM

    for query_id, query_token in enumerate(query_tokens, 1):
        print(f"\nPROCESSING QUERY: {query_id}")
        print("Running TF-IDF Scorer...")
        tfidf = tfidf_scorer(query_token, termids, docids, term_info, inverted_idx_file)
        queries_tfidf[query_id] = tfidf
        print("Running Okapi BM25 Scorer...")
        bm25 = bm25_scorer(query_token, termids, docids, term_info,
                           inverted_idx_file, doc_lengths, total_terms, bm25_k1, bm25_k2, bm25_b)
        queries_bm25[query_id] = bm25
        print("Running Dirichlet Scorer...")
        dirichlet = dirichlet_scorer(query_token, termids, docids, term_info,
                           inverted_idx_file, doc_lengths, total_terms)
        queries_dirichlet[query_id] = dirichlet

    tfidf_ranked_docs = {} # TFIDF scoring ranking.
    bm25_ranked_docs = {} # BM25 scoring ranking.
    dirichlet_ranked_docs = {} # LM using Dirichlet Smoothing ranking.

    for query_id, tfidf_docs in queries_tfidf.items():
        # Storing all docids for all of the 3 methods.
        tfidf_ranked_docs[query_id] = list(tfidf_docs.keys())
        bm25_ranked_docs[query_id] = list(queries_bm25[query_id].keys())
        dirichlet_ranked_docs[query_id] = list(queries_dirichlet[query_id].keys())

        # Ranking on the basis of score provided.
        tfidf_ranked_docs[query_id].sort(key = lambda k : tfidf_docs[k], reverse=True)
        bm25_ranked_docs[query_id].sort(key = lambda k : queries_bm25[query_id][k], reverse=True)
        dirichlet_ranked_docs[query_id].sort(
            key = lambda k : queries_dirichlet[query_id][k], reverse=True
            )

    # We can safely delete the scores after we have the ranking,
    # to free up the memory, they are deleted.
    del queries_bm25
    del queries_tfidf
    del queries_dirichlet

    inv_docids = {} # Dictionary mapping of docid to doc name. Used in evaluation.
    for docname, docid in docids.items():
        inv_docids[docid] = docname

    # Creating a data dictionary for evaluation metrics.
    evaluation = {}

    for query_id in range(1, 11): # Looping over each query, there are total 10 queries.
        tfidf_rel_count = 0
        bm25_rel_count = 0
        dirichlet_rel_count = 0
        evaluation[query_id] = []
        for i in range(5): # Calculating relevant documents @ 5
            tfidf_doc = tfidf_ranked_docs[query_id][i] # Finding the ith doc for tfidf.
            bm25_doc = bm25_ranked_docs[query_id][i] # Finding the ith doc for tfidf.
            dirichlet_doc = dirichlet_ranked_docs[query_id][i] # Finding the ith doc for tfidf.

            tfidf_doc = int(inv_docids[tfidf_doc]) # Converting docid to docname.
            bm25_doc = int(inv_docids[bm25_doc]) # Converting docid to docname.
            dirichlet_doc = int(inv_docids[dirichlet_doc]) # Converting docid to docname.

            # 1 will be added if relevant, otherwise 0 will be added.
            # If doc exists, its score, otherwise assume its non relevant.
            tfidf_rel_count += rel_dict[query_id][0].get(tfidf_doc, 0)
            bm25_rel_count += rel_dict[query_id][0].get(bm25_doc, 0)
            dirichlet_rel_count += rel_dict[query_id][0].get(dirichlet_doc, 0)

        # Relevant docs at 5 found.
        evaluation[query_id].append(tfidf_rel_count / 5) # P@5 for TF-IDF
        evaluation[query_id].append(bm25_rel_count / 5) # P@5 for Okapi BM25
        evaluation[query_id].append(dirichlet_rel_count / 5) # P@5 for Dirichlet Smoothing.

        for i in range(5, 10): # Calculating relevant documents @ 10
            tfidf_doc = tfidf_ranked_docs[query_id][i] # Finding the ith doc for tfidf.
            bm25_doc = bm25_ranked_docs[query_id][i] # Finding the ith doc for tfidf.
            dirichlet_doc = dirichlet_ranked_docs[query_id][i] # Finding the ith doc for tfidf.

            tfidf_doc = int(inv_docids[tfidf_doc]) # Converting docid to docname.
            bm25_doc = int(inv_docids[bm25_doc]) # Converting docid to docname.
            dirichlet_doc = int(inv_docids[dirichlet_doc]) # Converting docid to docname.

            # 1 will be added if relevant, otherwise 0 will be added.
            # If doc exists, its score, otherwise assume its non relevant.
            tfidf_rel_count += rel_dict[query_id][0].get(tfidf_doc, 0)
            bm25_rel_count += rel_dict[query_id][0].get(bm25_doc, 0)
            dirichlet_rel_count += rel_dict[query_id][0].get(dirichlet_doc, 0)

        # Relevant docs at 10 found.
        evaluation[query_id].append(tfidf_rel_count / 10) # P@10 for TF-IDF
        evaluation[query_id].append(bm25_rel_count / 10) # P@10 for Okapi BM25
        evaluation[query_id].append(dirichlet_rel_count / 10) # P@10 for Dirichlet Smoothing.

        for i in range(10, 20): # Calculating relevant documents @ 20
            tfidf_doc = tfidf_ranked_docs[query_id][i] # Finding the ith doc for tfidf.
            bm25_doc = bm25_ranked_docs[query_id][i] # Finding the ith doc for tfidf.
            dirichlet_doc = dirichlet_ranked_docs[query_id][i] # Finding the ith doc for tfidf.

            tfidf_doc = int(inv_docids[tfidf_doc]) # Converting docid to docname.
            bm25_doc = int(inv_docids[bm25_doc]) # Converting docid to docname.
            dirichlet_doc = int(inv_docids[dirichlet_doc]) # Converting docid to docname.

            # 1 will be added if relevant, otherwise 0 will be added.
            # If doc exists, its score, otherwise assume its non relevant.
            tfidf_rel_count += rel_dict[query_id][0].get(tfidf_doc, 0)
            bm25_rel_count += rel_dict[query_id][0].get(bm25_doc, 0)
            dirichlet_rel_count += rel_dict[query_id][0].get(dirichlet_doc, 0)

        # Relevant docs at 20 found.
        evaluation[query_id].append(tfidf_rel_count / 20) # P@20 for TF-IDF
        evaluation[query_id].append(bm25_rel_count / 20) # P@20 for Okapi BM25
        evaluation[query_id].append(dirichlet_rel_count / 20) # P@20 for Dirichlet Smoothing.

        for i in range(20, 30): # Calculating relevant documents @ 30
            tfidf_doc = tfidf_ranked_docs[query_id][i] # Finding the ith doc for tfidf.
            bm25_doc = bm25_ranked_docs[query_id][i] # Finding the ith doc for tfidf.
            dirichlet_doc = dirichlet_ranked_docs[query_id][i] # Finding the ith doc for tfidf.

            tfidf_doc = int(inv_docids[tfidf_doc]) # Converting docid to docname.
            bm25_doc = int(inv_docids[bm25_doc]) # Converting docid to docname.
            dirichlet_doc = int(inv_docids[dirichlet_doc]) # Converting docid to docname.

            # 1 will be added if relevant, otherwise 0 will be added.
            # If doc exists, its score, otherwise assume its non relevant.
            tfidf_rel_count += rel_dict[query_id][0].get(tfidf_doc, 0)
            bm25_rel_count += rel_dict[query_id][0].get(bm25_doc, 0)
            dirichlet_rel_count += rel_dict[query_id][0].get(dirichlet_doc, 0)

        # Relevant docs at 30 found.
        evaluation[query_id].append(tfidf_rel_count / 30) # P@30 for TF-IDF
        evaluation[query_id].append(bm25_rel_count / 30) # P@30 for Okapi BM25
        evaluation[query_id].append(dirichlet_rel_count / 30) # P@30 for Dirichlet Smoothing.

        # Resetting this value for MAP.
        tfidf_rel_count = 0
        bm25_rel_count = 0
        dirichlet_rel_count = 0

        # MAP summation variables.
        tfidf_map = 0
        bm25_map = 0
        dirichlet_map = 0

        for rank, docid in enumerate(tfidf_ranked_docs[query_id], 1):
            docid = int(inv_docids[docid]) # Converting docid to docname.
            if rel_dict[query_id][0].get(docid, 0) == 1: # The document is relevant.
                tfidf_rel_count += 1 # Increment counter.
                # Getting the precision at that point and dividing.
                tfidf_map += (tfidf_rel_count / rank)

        for rank, docid in enumerate(bm25_ranked_docs[query_id], 1):
            docid = int(inv_docids[docid]) # Converting docid to docname.
            if rel_dict[query_id][0].get(docid, 0) == 1: # The document is relevant.
                bm25_rel_count += 1 # Increment counter.
                # Getting the precision at that point and dividing.
                bm25_map += (bm25_rel_count / rank)

        for rank, docid in enumerate(dirichlet_ranked_docs[query_id], 1):
            docid = int(inv_docids[docid]) # Converting docid to docname.
            if rel_dict[query_id][0].get(docid, 0) == 1: # The document is relevant.
                dirichlet_rel_count += 1 # Increment counter.
                # Getting the precision at that point and dividing.
                dirichlet_map += (dirichlet_rel_count / rank)

        # Adding the MAP values to the evaluation dict after dividing them with total relevant docs.
        evaluation[query_id].append(tfidf_map / rel_dict[query_id][1]) # MAP for TF-IDF
        evaluation[query_id].append(bm25_map / rel_dict[query_id][1]) # MAP for Okapi BM25
        # MAP for Dirichlet Smoothing.
        evaluation[query_id].append(dirichlet_map / rel_dict[query_id][1])

    # Dictionary to calculate total evaluation for all queries for tfidf
    tfidf_sum = {
        "P@5" : 0,
        "P@10" : 0,
        "P@20" : 0,
        "P@30" : 0,
        "MAP" : 0,
    }

    # Dictionary to calculate total evaluation for all queries for BM25
    bm25_sum = {
        "P@5" : 0,
        "P@10" : 0,
        "P@20" : 0,
        "P@30" : 0,
        "MAP" : 0,
    }

    # Dictionary to calculate total evaluation for all queries for Dirichlet
    dirichlet_sum = {
        "P@5" : 0,
        "P@10" : 0,
        "P@20" : 0,
        "P@30" : 0,
        "MAP" : 0,
    }

    for query_id, eval_scores in evaluation.items():
        tfidf_sum["P@5"] += eval_scores[0]
        bm25_sum["P@5"] += eval_scores[1]
        dirichlet_sum["P@5"] += eval_scores[2]
        tfidf_sum["P@10"] += eval_scores[3]
        bm25_sum["P@10"] += eval_scores[4]
        dirichlet_sum["P@10"] += eval_scores[5]
        tfidf_sum["P@20"] += eval_scores[6]
        bm25_sum["P@20"] += eval_scores[7]
        dirichlet_sum["P@20"] += eval_scores[8]
        tfidf_sum["P@30"] += eval_scores[9]
        bm25_sum["P@30"] += eval_scores[10]
        dirichlet_sum["P@30"] += eval_scores[11]
        tfidf_sum["MAP"] += eval_scores[12]
        bm25_sum["MAP"] += eval_scores[13]
        dirichlet_sum["MAP"] += eval_scores[14]

    evaluation["Total Average"] = [] # The total averages of all metrics.

    for metric, tfidf_metric_score in tfidf_sum.items(): # Populating the evaluation dictionary.
        evaluation["Total Average"].append(
            tfidf_metric_score / 10
        )
        evaluation["Total Average"].append(
            bm25_sum[metric] / 10
        )
        evaluation["Total Average"].append(
            dirichlet_sum[metric] / 10
        )

    data_index = [] # Building the index.
    for metric in tfidf_sum:
        data_index.append(f"TF-IDF {metric}")
        data_index.append(f"Okapi BM25 {metric}")
        data_index.append(f"LM Using Dirichlet Smoothing {metric}")

    evaluate_df = pd.DataFrame(evaluation, index=data_index)

    evaluate_df.to_csv("Evaluation Results.csv")
    print("Evaluation Results written to 'Evaluation Results.csv'")

def main():
    """Main function to run the entire code.
    """

    parser = set_arguments()
    args = parser.parse_args()
    if args.evaluate:
        evaluator()
    else:
        run_scorer(args.score)

def _main():
    # FILENAMES
    qrel_filename = "qrels.xlsx"
    queries_file = "topics.txt"
    stopwords_file = "Urdu stopwords.txt"
    termid_file = "indices/termids.txt"
    docid_file = "indices/docids.txt"
    term_info_file = "indices/term_info.txt"
    inverted_idx_file = "indices/term_index.txt"
    forward_index_file = "indices/doc_index.txt"
    bm25_k1 = 1.2
    bm25_k2 = 1000
    bm25_b = 0.75

    # VARIABLES USED.
    rel_dict = read_query_rel(qrel_filename)
    queries = get_queries(queries_file)
    query_tokens = [] # A 2d list containing each query's tokens.

    print("Processing Queries...")
    for query in queries: # Preprocessing each query and appending them to a list.
        query_tokens.append(preprocess_string(query, stopwords_file))

    print("Loading metadata for the indexer...")
    termids = get_ids(termid_file)
    docids = get_ids(docid_file)
    term_info = get_term_info(term_info_file)
    print("Loading Document lengths...")
    doc_lengths = get_doc_lengths(forward_index_file)
    print("Document Lengths loaded...")
    # Getting the total term count for the corpus
    # which should be the total sum of all docids' lengths.
    total_terms = sum(doc_lengths.values())

    queries_tfidf = {} # Dictionary for each query's tfidf
    queries_bm25 = {} # Dictionary for each query's BM25
    queries_dirichlet = {} # Dictionary for each query's Dirichlet Smoothing LM

    for query_id, query_token in enumerate(query_tokens, 1):
        tfidf = tfidf_scorer(query_token, termids, docids, term_info, inverted_idx_file)
        for i, key in enumerate(tfidf):
            print(key, tfidf[key])
            if i == 10:
                break
        if query_id == 1:
            break

    print(rel_dict[1][0].values())


if __name__ == "__main__":
    main()
