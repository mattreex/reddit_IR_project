import array
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string, re


class CompressedPostings:
    @staticmethod
    def VBEncodeNum(n):
        byte = []
        while True:
            byte.append(n % 128)
            if n < 128:
                break
            n //= 128
        byte[0] += 128
        return byte[::-1]

    @staticmethod
    def VBEncode(n_list):
        b = []
        for n in n_list:
            b.extend(CompressedPostings.VBEncodeNum(n))
        return b

    @staticmethod
    def VBDecode(bs):
        n_list = []
        n = 0
        for b in bs:
            if b < 128:
                n = 128*n + b
            else:
                n = 128*n + b - 128
                n_list.append(n)
                n = 0
        return n_list
    

    @staticmethod
    def encode(postings_list):
        """Encodes `postings_list` using gap encoding with variable byte 
        encoding for each gap
        
        Args
        
        postings_list: List[int]
            The postings list to be encoded
        
        Returns
        
        bytes: 
            Bytes reprsentation of the compressed postings list 
            (as produced by `array.tobytes` function)
        """
        p = postings_list.copy()
        for i in range(1, len(p))[::-1]:
            p[i] -= p[i-1]
        vb = CompressedPostings.VBEncode(p)
        return array.array('B', vb).tobytes()
        

    @staticmethod
    def decode(encoded_postings_list):
        """Decodes a byte representation of compressed postings list
        
        Args
        
        encoded_postings_list: bytes
            Bytes representation as produced by `CompressedPostings.encode` 
            
        Returns
        
        List[int]
            Decoded postings list (each posting is a docIds)
        """
        vb = array.array('B')
        vb.frombytes(encoded_postings_list)
        postings_list = CompressedPostings.VBDecode(vb.tolist())
        for i in range(1, len(postings_list)):
            postings_list[i] += postings_list[i-1]
        return postings_list


def sorted_intersect(list1, list2):
    """Intersects two (ascending) sorted lists and returns the sorted result
    
    Args
    
        list1: List[Comparable]
        list2: List[Comparable]
            Sorted lists to be intersected
        
    Returns
    
        List[Comparable]
            Sorted intersection        
    """
    idx1 = idx2 = 0
    intersect = []
    while idx1 < len(list1) and idx2 < len(list2):
        if list1[idx1] < list2[idx2]:
            idx1 += 1
        elif list2[idx2] < list1[idx1]:
            idx2 += 1
        else:
            intersect.append(list1[idx1])
            idx1 += 1
            idx2 += 1
    return intersect


def tokenize_text(text):
    stemmer = SnowballStemmer(language='english')
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in word_tokenize(text) if token not in stop_words and token.isalnum()]
    return [stemmer.stem(token) for token in tokens]


def clean_text(text):
    if type(text) != str or text == "[removed]" or text == 'nan':
        return ""
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub("", text)