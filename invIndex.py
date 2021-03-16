from collections import defaultdict, Counter
import pickle as pkl
import os

from utils import CompressedPostings


class InvertedIndex:
    """A class that implements efficient reads and writes of an inverted index 
    to disk
    
    Attributes
    ----------
    postings_dict: Dictionary mapping: termID->(start_position_in_index_file, 
                                                number_of_postings_in_list,
                                               length_in_bytes_of_postings_list)
        This is a dictionary that maps from termIDs to a 3-tuple of metadata 
        that is helpful in reading and writing the postings in the index file 
        to/from disk. This mapping is supposed to be kept in memory. 
        start_position_in_index_file is the position (in bytes) of the postings 
        list in the index file
        number_of_postings_in_list is the number of postings (docIDs) in the 
        postings list
        length_in_bytes_of_postings_list is the length of the byte 
        encoding of the postings list
        
    """
    def __init__(self, index_name, postings_encoding=None, directory=''):
        """
        Parameters
        ----------
        index_name (str): Name used to store files related to the index
        postings_encoding: A class implementing static methods for encoding and 
            decoding lists of integers. Default is None, which gets replaced
            with UncompressedPostings
        directory (str): Directory where the index files will be stored
        """

        self.index_file_path = os.path.join(directory, index_name+'.index')
        self.metadata_file_path = os.path.join(directory, index_name+'.dict')

        if postings_encoding is None:
            self.postings_encoding = CompressedPostings
        else:
            self.postings_encoding = postings_encoding
        self.directory = directory

        self.postings_dict = {}
        self.terms = []

    def __enter__(self):
        """Opens the index_file and loads metadata upon entering the context"""
        # Open the index file
        self.index_file = open(self.index_file_path, 'rb+')

        # Load the postings dict and terms from the metadata file
        with open(self.metadata_file_path, 'rb') as f:
            self.postings_dict, self.terms = pkl.load(f)
            self.term_iter = self.terms.__iter__()                       

        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        """Closes the index_file and saves metadata upon exiting the context"""
        # Close the index file
        self.index_file.close()
        
        # Write the postings dict and terms to the metadata file
        with open(self.metadata_file_path, 'wb') as f:
            pkl.dump([self.postings_dict, self.terms], f)


class IdMap:
    """Helper class to store a mapping from strings to ids."""
    def __init__(self):
        self.str_to_id = {}
        self.id_to_str = []
        
    def __len__(self):
        """Return number of terms stored in the IdMap"""
        return len(self.id_to_str)
        
    def _get_str(self, i):
        """Returns the string corresponding to a given id (`i`)."""
        return self.id_to_str[i]
        

    def _get_id(self, s):
        """Returns the id corresponding to a string (`s`). 
        If `s` is not in the IdMap yet, then assigns a new id and returns the new id.
        """
        if s not in self.str_to_id:
            self.str_to_id[s] = len(self.id_to_str)
            self.id_to_str.append(s)
        return self.str_to_id.get(s)
            

    def __getitem__(self, key):
        """If `key` is a integer, use _get_str; 
           If `key` is a string, use _get_id;"""
        if type(key) is int:
            return self._get_str(key)
        elif type(key) is str:
            return self._get_id(key)
        else:
            raise TypeError

class VectorIdMap(IdMap):

    def __init__(self):
        self.docId_to_vector = defaultdict(Counter)
        self.docId_to_tfidf = {}

    def add_term_occurance(self, docid, term):
        self.docId_to_vector[docid][term] += 1

    def __len__(self):
        return len(self.docId_to_vector)
    
    def __getitem__(self, key):
        return self.docId_to_vector[key]


class InvertedIndexMapper(InvertedIndex):
    def __getitem__(self, key):
        return self._get_postings_list(key)
    
    def _get_postings_list(self, term):
        """Gets a postings list (of docIds) for `term`.
        
        This function should not iterate through the index file.
        I.e., it should only have to read the bytes from the index file
        corresponding to the postings list for the requested term.
        """
        start_position, n_postings, length_in_bytes = self.postings_dict[term]
        self.index_file.seek(start_position)
        return self.postings_encoding.decode(self.index_file.read(length_in_bytes))


class InvertedIndexWriter(InvertedIndex):
    """"""
    def __enter__(self):
        self.index_file = open(self.index_file_path, 'wb+')              
        return self

    def append(self, term, postings_list):
        """Appends the term and postings_list to end of the index file.
        
        This function does three things, 
        1. Encodes the postings_list using self.postings_encoding
        2. Stores metadata in the form of self.terms and self.postings_dict
           Note that self.postings_dict maps termID to a 3 tuple of 
           (start_position_in_index_file, 
           number_of_postings_in_list, 
           length_in_bytes_of_postings_list)
        3. Appends the bytestream to the index file on disk
        
        Args
        
            term:
                term or termID is the unique identifier for the term
            postings_list: List[Int]
                List of docIDs where the term appears
        """
        encoded_postings_list = self.postings_encoding.encode(postings_list)
        start_position_in_index_file = self.index_file.seek(0, 2)
        length_in_bytes_of_postings_list = self.index_file.write(encoded_postings_list)
        self.terms.append(term)
        self.postings_dict[term] = (start_position_in_index_file, len(postings_list), length_in_bytes_of_postings_list)


class InvertedIndexIterator(InvertedIndex):
    """"""
    def __enter__(self):
        """Adds an initialization_hook to the __enter__ function of super class
        """
        super().__enter__()
        self._initialization_hook()
        return self

    def _initialization_hook(self):
        """Use this function to initialize the iterator
        """
        self.curr_term_pos = 0

    def __iter__(self): 
        return self
    
    def __next__(self):
        """Returns the next (term, postings_list) pair in the index.
        
        Note: This function should only read a small amount of data from the 
        index file. In particular, you should not try to maintain the full 
        index file in memory.
        """
        if self.curr_term_pos < len(self.terms):
            term = self.terms[self.curr_term_pos]
            self.curr_term_pos += 1
            start_position, n_postings, length_in_bytes = self.postings_dict[term]
            self.index_file.seek(start_position)
            postings_list = self.postings_encoding.decode(self.index_file.read(length_in_bytes))
            return term, postings_list
        else:
            raise StopIteration

    def delete_from_disk(self):
        """Marks the index for deletion upon exit. Useful for temporary indices
        """
        self.delete_upon_exit = True

    def __exit__(self, exception_type, exception_value, traceback):
        """Delete the index file upon exiting the context along with the
        functions of the super class __exit__ function"""
        self.index_file.close()
        if hasattr(self, 'delete_upon_exit') and self.delete_upon_exit:
            os.remove(self.index_file_path)
            os.remove(self.metadata_file_path)
        else:
            with open(self.metadata_file_path, 'wb') as f:
                pkl.dump([self.postings_dict, self.terms], f)


