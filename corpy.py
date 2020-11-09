import glob
import pickle
import collections
import copy
import numpy as np
import warnings
import random
import os


class Corpy():

    def __init__(self, books, **kwargs):
        self.mode                = kwargs.get('mode',                'word'  )
        self.lower               = kwargs.get('lower',               True    )
        self.one_document        = kwargs.get('one_document',        False   )
        self.threshold           = kwargs.get('threshold',           None    )
        self.threshold_section   = kwargs.get('threshold_section', 'first' ) # 'first', 'all' or int
        self.text_sections       = kwargs.get('text_sections',       (1,)    ) # tuple or list: (train, valid, test, ...)
        self.text_sections_level = kwargs.get('text_sections_level', 'item'  ) # 'book' or 'item'
        self.init_books_seq      = kwargs.get('init_books_seq',      'normal') # 'normal', 'random' or sequence list
        self.punct               = kwargs.get('punct',                "'.,!?«»:;()[]-"   ) # list or string of punctuation to divide words
        
        
        self.ind_books = np.arange(len(books))
        if type(self.init_books_seq) == str:
            if self.init_books_seq == 'normal':
                pass
            elif self.init_books_seq == 'random':
                np.random.shuffle(self.ind_books)
            else:
                warnings.warn('Init_books_seq not recognised. Using \'normal\'', UserWarning)
        elif type(self.init_books_seq) == list:
            self.ind_books = self.init_books_seq
        else:
            raise ValueError('init_books_seq must be a string (\'normal\' or \'random\') or a list of indexes')
            
                
        if self.one_document:
            string = ''
            for b in books:
                string += b + ' '
            self.books = [string]
            self.ind_books = [0]
        else:
            self.books = copy.deepcopy(books) # list of book strings
        self.num_books = len(self.books)
        
        self.books_list_of_items = None # list of book lists containing the single items
        self.books_encoding = None # list of book encodings
        self.items_count = None # dictionary with the number of items (word or char) occourrences
        self.items_freq = None # dictionary with the frequency of items occourrences
        self.ind2item = None # dictionary to translate one item encoding to item
        self.item2ind = None # dictionary to translate an item to the associated encoding
        self.num_items = None # number of different items
        
        self.book_ind  = [0] * len(self.text_sections)
        self.chunk_ind = [0] * len(self.text_sections)
        
        self._build()
        
    def _build(self):
        all_items = []
        self.books_list_of_items = []
        
        # Reading the full text
        for k in self.ind_books:
            b = self.books[k]
            if self.lower:
                self.books[k] = self.books[k].lower()
            if self.mode == 'word':
                for p in self.punct:
                    self.books[k] = self.books[k].replace(p, " "+p+" ")
                while self.books[k].find('  ') > -1:
                    self.books[k] = self.books[k].replace('  ', ' ')
                self.books[k] = self.books[k].strip()
                self.books_list_of_items.append(self.books[k].split(' '))
            elif self.mode == 'char':
                self.books_list_of_items.append(list(self.books[k]))
            #all_items += self.books_list_of_items[-1]
        
        # Building the sections
        self._sections_building()
        
        # Calculating the items distribution
        if type(self.threshold_section) == int:
            for kb, b in enumerate(self.sections_book[self.threshold_section]):
                #print(b)
                fr, to = self.sections_text[self.threshold_section][kb]
                all_items += self.books_list_of_items[b][fr:to]
        elif type(self.threshold_section) == str:
            if self.threshold_section == 'first':
                for kb, b in enumerate(self.sections_book[0]):
                    fr, to = self.sections_text[0][b]
                    all_items += self.books_list_of_items[kb][fr:to]
            elif self.threshold_section == 'all':
                for b in self.books_list_of_items:
                    all_items += b
                
        self.items_count = collections.Counter(all_items)
        self.items_count = {k: v for k, v in sorted(self.items_count.items(), key=lambda item: item[1], reverse=True)}
        self.num_items = len(self.items_count)
        tot_items = len(all_items)
        self.items_freq = dict()
        for k in self.items_count:
            self.items_freq[k] = self.items_count[k] / tot_items
        
        
        
        # Cutting the distribution
        if self.threshold is not None:
            if self.mode == 'char':
                warnings.warn('Char mode. Threshold should not be used.', UserWarning)
            key2remove = []
            if self.threshold > 1:
                self.threshold = int(self.threshold)
                for k, (key, value) in enumerate(self.items_count.items()):
                    if k >= self.threshold:
                        key2remove.append(key)
            elif self.threshold < 1. and self.threshold > 0.:
                cumv = 0.0
                for key, value in self.items_freq.items():
                    cumv += value
                    if cumv >= self.threshold:
                        key2remove.append(key)
            else:
                raise ValueError('max_items must be positive!')
            for key in key2remove:
                del self.items_freq[key]
                del self.items_count[key]
            self.num_items = len(self.items_count)
        
        # Building the dictionaries for connecting items to code
        self.ind2item = dict()
        self.item2ind = dict()
        for k, key in enumerate(self.items_count):
            self.ind2item[k] = key
            self.item2ind[key] = k
        self.item2ind = collections.OrderedDict(sorted(self.item2ind.items()))
        
        # Updating the text after cutting distribution
        self.books_encoding = []
        for kb, b in enumerate(self.books_list_of_items):
            self.books_encoding.append([])
            for k, w in enumerate(b):
                code = self.item2ind.get(w, None)
                self.books_encoding[-1].append(code)
                self.books_list_of_items[kb][k] = self.ind2item.get(code)
        
        
        
        # Removing the unnecessary data
        del self.books
        del self.num_items
        del self.items_count
        del self.items_freq
    
    def _sections_building(self):
        ss_sum = 0.0
        self.text_sections = list(self.text_sections)
        for ss in self.text_sections:
            ss_sum += ss
        for k, ss in enumerate(self.text_sections):
            self.text_sections[k] /= ss_sum
        full_text_len = 0
        for kb in range(len(self.books_list_of_items)):
            full_text_len += len(self.books_list_of_items[kb])
        number_of_items_per_section = []
        for section_k, ts in enumerate(self.text_sections):
            number_of_items_per_section.append(int(ts * full_text_len))
        if sum(number_of_items_per_section) < full_text_len:
            number_of_items_per_section[-1] += full_text_len - sum(number_of_items_per_section)
        
        self.sections_book = [[]]
        self.sections_text = [[]]
        
        if self.text_sections_level == 'item':
            book = 0
            fr = 0

            for section_k, nips in enumerate(number_of_items_per_section):
                nips_rem = number_of_items_per_section[section_k]
                while nips_rem > 0:
                    if len(self.books_list_of_items[book][fr:]) <= nips_rem:
                        self.sections_book[section_k].append(book)
                        self.sections_text[section_k].append([fr,len(self.books_list_of_items[book])])
                        nips_rem -= len(self.books_list_of_items[book]) - fr
                        fr = 0
                        book += 1
                    else:
                        self.sections_book[section_k].append(book)
                        if nips_rem > len(self.books_list_of_items[book]) - fr:
                            to = len(self.books_list_of_items[book])
                            self.sections_text[section_k].append([fr, to])
                            fr = 0
                            book += 1
                            nips_rem -= to - fr
                        else:
                            to = fr + nips_rem
                            self.sections_text[section_k].append([fr, to])
                            nips_rem -= to - fr
                            fr = to                        
                        if len(self.sections_book) < len(self.text_sections):
                            self.sections_book.append([])
                            self.sections_text.append([])
                        
        elif self.text_sections_level == 'book':
            section_k = 0
            temp_len = 0
            for kb, b in enumerate(self.books_list_of_items):
                if len(b) + temp_len < number_of_items_per_section[section_k]:
                    self.sections_book[section_k].append(kb)
                    self.sections_text[section_k].append([0, len(b)])
                    temp_len += len(b)
                else:
                    if (number_of_items_per_section[section_k] - temp_len) > int(len(b) * 0.75):
                        self.sections_book[section_k].append(kb)
                        self.sections_text[section_k].append([0, len(b)])
                        section_k += 1
                        temp_len = 0
                        if len(self.sections_book) < len(self.text_sections):
                            self.sections_book.append([])
                            self.sections_text.append([])
                    else:
                        if len(self.sections_book) < len(self.text_sections):
                            self.sections_book.append([])
                            self.sections_text.append([])
                            section_k += 1
                            self.sections_book[section_k].append(kb)
                            self.sections_text[section_k].append([0, len(b)])
                            temp_len = len(b)
            
            for k, sb in enumerate(self.sections_book):
                if sb == []:
                    del self.sections_book[k]
                    del self.sections_text[k]
            
            if len(self.sections_book) < len(self.text_sections):
                wstr = 'Number of sections lower than requested: {}/{}'.format(len(self.sections_book), len(self.text_sections))
                warnings.warn(wstr, UserWarning)
        
    
    def reset_counter(self, section='all'):
        if type(section) == int:
            self.book_ind[section] = 0
            self.chunk_ind[section] = 0
        elif type(section) == str and section == 'all':
            for s in range(len(self.sections_book)):
                self.book_ind[s] = 0
                self.chunk_ind[s] = 0
    
    def string2code(self, string):
        code = []
        string_list_of_items = []
        if self.mode == 'word':
            #self.punct = "'.,!?«»:;()[]-"
            for p in self.punct:
                string = string.replace(p, " "+p+" ")
            while string.find('  ') > -1:
                string = string.replace('  ', ' ')
            string = string.strip()
            string_list_of_items = string.split(' ')
        elif self.mode == 'char':
            string_list_of_items = list(string)
        for w in string_list_of_items:
            code.append(self.item2ind.get(w))
        return code
    
    def items2code(self, items):
        code = []
        for w in items:
            code.append(self.item2ind.get(w))
        return code
    
    def code2items(self, code):
        items = []
        for c in code:
            items.append(self.ind2item.get(c))
        return items
    
    def code2string(self, code):
        string = ''
        for c in code:
            string += self.ind2item.get(c, 'UNK')
            if self.mode == 'word':
                string += ' '
        string = string.strip()
        return string
    
    def get_chunk(self, **kwargs):
        book_sel     = kwargs.get('book_sel'      , -1        ) # book index relative to the chosen section
        chunk_sel    = kwargs.get('chunk_sel'     , -1        ) # chunk starting index relative to the book in the section
        chunk_len    = kwargs.get('chunk_len'     , 30        )
        chunk_mode   = kwargs.get('chunk_mode'    , 'normal'  ) # 'normal', 'sequential', 'random'
        padding      = kwargs.get('padding'       , 0         )
        last_element = kwargs.get('last_element'  , True      )
        output_mode  = kwargs.get('output_mode'   , 'code'    ) # 'code' list, 'item' list, 'string'
        section      = kwargs.get('section'       , 0         ) # index (int) of the section or 'all'
        unk          = kwargs.get('unk'           , 'max'     ) # In case of output_mode='code': 
                                                                #    'max': insert the code=max item
                                                                #    'none': insert None
                                                                # In case of output_mode='item' or 'string':
                                                                #    insert the passed string
        
        if type(section) == int:
            books = self.sections_book[section] # list of absolute index in the section
        elif type(section) == str and section == 'all':
            books = list(range(self.num_books))
        texts = []
        for kb, b in enumerate(books):
            texts.append(self.sections_text[section][kb]) # list of couples [init,end] of each book in the section
        num_books = len(books)        
        
        if chunk_mode == 'sequential':
            book_sel_rel = self.book_ind[section]
            book_sel_abs = books[book_sel_rel]
            chunk_sel_abs = self.chunk_ind[section] + texts[book_sel_rel][0]
        elif chunk_mode == 'random':
            book_sel_rel = random.randint(0,len(books)-1)
            book_sel_abs = books[book_sel_rel]
            chunk_sel_abs = random.randint(texts[book_sel_rel][0], texts[book_sel_rel][1]-10)
        elif chunk_mode == 'normal':
            if book_sel == -1 or chunk_sel == -1:
                raise ValueError('get_chunk: if sequential=False, book_sel and chunk_sel must be greater than -1')
            else:
                book_sel_rel = book_sel
                book_sel_abs = books[book_sel_rel]
                chunk_sel_abs = chunk_sel
                if chunk_sel >= texts[book_sel_rel][1]:
                    raise ValueError('get_chunk: chunk_sel is greater than the chosen section chunk')
        
        end_book = False
    
        if book_sel_abs < len(self.books_encoding):
            if chunk_sel_abs < texts[book_sel_rel][1]:
                fr = chunk_sel_abs
                to = fr + chunk_len
                if last_element:
                    to += 1
                if to > texts[book_sel_rel][1]:
                    to = texts[book_sel_rel][1]
                    end_book = True
                if output_mode == 'code':
                    output = self.books_encoding[book_sel_abs][fr:to]
                    if unk == 'max':
                        for k, o in enumerate(output):
                            if o is None:
                                output[k] = len(self.ind2item)
                    elif unk == 'none':
                        pass
                elif output_mode == 'item':
                    output = self.books_list_of_items[book_sel_abs][fr:to]
                    for k, o in enumerate(output):
                        if o is None:
                            if self.mode == 'word':
                                if unk=='max':
                                    output[k] = 'UNK'
                                elif unk=='none':
                                    output[k] = None
                                else:
                                    output[k] = unk
                elif output_mode == 'string':
                    output = self.code2string(self.books_encoding[book_sel_abs][fr:to])
            else:
                raise ValueError('get_chunk: chunk_sel greater than book length')
        else:
            raise ValueError('get_chunk: book_sel greater than total books')
            
        if chunk_mode == 'sequential':
            if end_book:
                self.book_ind[section] += 1
                if self.book_ind[section] >= len(books):
                    self.book_ind[section] = 0
                self.chunk_ind[section] = 0
            else:
                if padding < 1:
                    self.chunk_ind[section] += chunk_len
                else:
                    self.chunk_ind[section] += padding
                    if self.chunk_ind[section] >= len(texts[self.book_ind[section]]):
                        self.chunk_ind[section] = 0
                        self.book_ind[section] += 1
                        if self.book_ind[section] >= len(books):
                            self.book_ind[section] = 0
                
        return output
    
    def save(self, namefile=''):
        if namefile == '':
            namefile = './corpy_'
            ind = 0
            while os.path.isfile(namefile + str(ind) + '.pkl'):
                ind += 1
            namefile += str(ind) + '.pkl'
            print('Corpy file saved as: ' + namefile)
        with open(namefile, 'wb') as f:
            pickle.dump(self, f)
            
def load_corpy(namefile):
    c = None
    with open(namefile, 'rb') as f:
        c = pickle.load(f)
    return c