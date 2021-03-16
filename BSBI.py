import os
import pickle as pkl
import contextlib
from collections import defaultdict, Counter
import heapq
import math

from invIndex import *
from utils import *

class BSBIIndex:
    """ 
    Attributes
    ----------
    term_id_map(IdMap): For mapping terms to termIDs
    doc_id_map(IdMap): For mapping relative paths of documents (eg 
        0/3dradiology.stanford.edu_) to docIDs
    data_dir(str): Path to data
    output_dir(str): Path to output index files
    index_name(str): Name assigned to index
    postings_encoding: Encoding used for storing the postings.
        The default (None) implies UncompressedPostings
    """
    def __init__(self, data_dir, output_dir, index_name = "BSBI"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.doc_vectors = VectorIdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = CompressedPostings
        self.doc_freqencies = {}
        self.idfs = {}

        # Stores names of intermediate indices
        self.intermediate_indices = []


    def save(self):
        """Dumps doc_id_map and term_id_map into output directory"""
        
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pkl.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pkl.dump(self.doc_id_map, f)
        with open(os.path.join(self.output_dir, 'doc_vectors.dict'), 'wb') as f:
            pkl.dump(self.doc_vectors, f)
        with open(os.path.join(self.output_dir, 'doc_freq.dict'), 'wb') as f:
            pkl.dump(self.doc_freqencies, f)
    

    def load(self):
        """Loads doc_id_map and term_id_map from output directory"""
        
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pkl.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pkl.load(f)
        with open(os.path.join(self.output_dir, 'doc_vectors.dict'), 'rb') as f:
            self.doc_vectors = pkl.load(f)
        with open(os.path.join(self.output_dir, 'doc_freq.dict'), 'rb') as f:
            self.doc_freqencies = pkl.load(f)


    def index(self):
        """Base indexing code
        
        This function loops through the data directories, 
        calls parse_block to parse the documents
        calls invert_write, which inverts each block and writes to a new index
        then saves the id maps and calls merge on the intermediate indices
        """
        for block_dir_relative in sorted(next(os.walk(self.data_dir))[1]):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, directory=self.output_dir, 
                                     postings_encoding=
                                     self.postings_encoding) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
        self.save()
        with InvertedIndexWriter(self.index_name, directory=self.output_dir, 
                                 postings_encoding=
                                 self.postings_encoding) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexIterator(index_id, 
                                          directory=self.output_dir, 
                                          postings_encoding=
                                          self.postings_encoding)) 
                 for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


    def parse_block(self, block_dir_relative):
        """Parses a tokenized text file into termID-docID pairs
        
        Args
        
            block_dir_relative : str
                Relative Path to the directory that contains the files for the block
        
        Returns
            List[Tuple[Int, Int]]
                Returns all the td_pairs extracted from the block
        
        Should use self.term_id_map and self.doc_id_map to get termIDs and docIDs.
        These persist across calls to parse_block
        """
        td_pairs = []
        for file_dir in sorted(os.listdir(os.path.join(self.data_dir, block_dir_relative))):
            with open(os.path.join(self.data_dir, block_dir_relative, file_dir), 'r') as f:
                content = f.read().strip().split() # list of tokens
                doc_id = self.doc_id_map[os.path.join(block_dir_relative, file_dir)]
                tokens = set(content)
                for token in tokens:
                    if not self.doc_freqencies.get(token):
                        self.doc_freqencies[token] =1
                    else:
                        self.doc_freqencies[token] +=1 
                for token in content:
                    self.doc_vectors.add_term_occurance(doc_id, token)
                    term_id = self.term_id_map[token]
                    td_pairs.append([term_id, doc_id])
        return td_pairs


    def invert_write(self, td_pairs, index):
        """Inverts td_pairs into postings_lists and writes them to the given index
        
        Args
        
            td_pairs: List[Tuple[Int, Int]]
                List of termID-docID pairs
            index: InvertedIndexWriter
                Inverted index on disk corresponding to the block       
        """
        td_dict = defaultdict(list)
        for t, d in td_pairs:
            td_dict[t].append(d)
        for t in sorted(td_dict.keys()):
            p_list = sorted(td_dict[t])
            index.append(t, sorted(p_list))
    

    def merge(self, indices, merged_index):
        """Merges multiple inverted indices into a single index
        
        Args
        
            indices: List[InvertedIndexIterator]
                A list of InvertedIndexIterator objects, each representing an
                iterable inverted index for a block
            merged_index: InvertedIndexWriter
                An instance of InvertedIndexWriter object into which each merged 
                postings list is written out one at a time
        """
        last_term = last_posting = None
        for curr_term, curr_postings in heapq.merge(*indices):
            if curr_term != last_term:
                if last_term:
                    last_posting = list(sorted(set(last_posting)))
                    merged_index.append(last_term, last_posting)
                last_term = curr_term
                last_posting = curr_postings
            else:
                last_posting += curr_postings
        if last_term:
            last_posting = list(sorted(set(last_posting)))
            merged_index.append(last_term, last_posting) 

    
    def retrieve(self, query):
        """Retrieves the documents corresponding to the conjunctive query
        
        Args
            query: str
                Space separated list of query tokens
            
        Returns
            List[str]
                Sorted list of documents which contains each of the query tokens. 
                Should be empty if no documents are found.
        
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()
        query = self.convert_query(query)
        with InvertedIndexMapper(self.index_name, directory=self.output_dir, 
                                 postings_encoding=
                                 self.postings_encoding) as mapper:
            result = None
            for term in query.split():
                term_id = self.term_id_map.str_to_id.get(term)
                if not term_id:
                    return []
                r = mapper[term_id]
                if result is None:
                    result = r
                else:
                    result = sorted_intersect(result, r)
        
        tmp = [self.doc_id_map[r] for r in result]
        result = []
        for link in tmp:
            start = link.find('()')
            result.append(f"reddit.com{link[start:].replace('()', '/')}")

        return result


    def generate_vector(self, prevec):
        """
        Args
            prevec: Iterable
                String to be generate tfidf weights for
        Returns
            dict
            tfidf weighed vector
        """
        if isinstance(prevec, str):
            prevec = Counter(prevec.split())
        vector = {}
        for token, cnt in prevec.items():
            if not self.idfs.get(token):
                idf = len(self.doc_vectors) / self.doc_freqencies[token]
                self.idfs[token] = 1 + math.log(idf)
            tfidf = self.idfs[token] * cnt
            vector[token] = tfidf
        return vector

    @staticmethod
    def convert_query(query):
        return " ". join(tokenize_text(clean_text(query)))

    def cosine_sim(self, vector1, vector2):
        import numpy as np
        vec1 = []
        vec2 = []
        if len(vector2) > len(vector1):
            terms = list(vector1.keys())
        else:
            terms = list(vector2.keys())
        for t in terms:
            val1 = vector1.get(t)
            val2 = vector2.get(t)
            if val1 is not None and val2 is not None:
                vec1.append(val1)
                vec2.append(val2)
        return np.array(vec1)@np.array(vec2)


    def tfidf_retrieve(self, query):
        """Retrieves the documents corresponding to the conjunctive query
        
        Args
            query: str
                Sentence to be queried
            
        Returns
            List[str]
                Sorted list of documents which contains all the tokens sorted by
                cosine similarity using tf-idf weights.
        
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0 or \
             len(self.doc_vectors) == 0 or len(self.doc_freqencies) == 0:
            self.load()
        query = self.convert_query(query)
        with InvertedIndexMapper(self.index_name, directory=self.output_dir, 
                                 postings_encoding=
                                 self.postings_encoding) as mapper:
            result = None
            query_vector = self.generate_vector(query)
            # try enforcing docs to have all query terms
            for term in query.split():
                term_id = self.term_id_map.str_to_id.get(term)
                if not term_id:
                    
                    return []
                r = mapper[term_id]
                if result is None:
                    result = r
                else:
                    result = sorted_intersect(result, r)
        document_vectors = [self.generate_vector(self.doc_vectors[r]) for r in result]
        ranked_vectors = []
        for docid, vector in zip(r, document_vectors):
            ranked_vectors.append((self.cosine_sim(query_vector, vector), docid))

        tmp = [self.doc_id_map[docid] for rank, docid in sorted(ranked_vectors, reverse=True)]
        result = []
        for link in tmp:
            start = link.find('()')
            result.append(f"reddit.com{link[start:].replace('()', '/')}")

        return result[:10]

try: 
    os.mkdir('output_dir_compressed')
except FileExistsError:
    pass
    
# BSBI_index = BSBIIndex(data_dir='docs', output_dir = 'output_dir_compressed', index_name='reddit')
# print(BSBI_index.tfidf_retrieve("the viking homeland"))