# Corpy
Corpy is a simple manager for textual corpora written in Python

## Features:
- Handling multiple documents
- Dictionary limited by setting a threshold
- Handling of sections (train, test,...) for ML purposes
- Different modalities to get textual chunks

## Usage
Corpy is a class requiring a list of textual documents at the input.

``` corpus = Corpy(_list_of_texts_) ```

creates a textual corpus.

### Parameters
Bold values mean Default
- one_document: concatenates all the documents into one (space separator). Vaues: True, **False**
- mode: encodes the text with words or characters. Values: **'word'**, 'char'
- lower: sets the full text lowercase. Values: **True**, False
- threshold: sets the threshold to limit the dictionary length. Values: **None**: keeps the full dictionary; _float < 1_ cuts the dictionary when the cumulative distribution of frequency is less than _float_; _int_ cuts the dictionary after the _int'th_ item.
- text_sections: divides the full text basing on the values passed through a list or a tuple. The values are normalized to 1. For example: [2,1,1] creates three sections whose length if 50%, 25%, 25% of the full text. Values: _list_ or _tuple_. Default: **(1,)** (only one section).
- text_sections_level: sections can be created by either counting the single items (words or chars) or the number of documents (books). Values: **'item'**, 'book'. If 'book', the documents into the section have at least the 75% of the text in that specific section.
- threshold_section: defines which section is used to cut the dictionary. Values: **'first'** (only the first section), 'all' (threshold is applied to the full text), _int_: the index of the chosen section.
- init_books_seq: order of input documents. Values: **'normal'** (the same order as read from the disk), 'random' (randomized order), _list_: list with the index of the documents. If len(_list_) < len(books), only the documents denoted by _list_ will be taken into account.
- punct: list of punctuation for single items (word mode only). Values: _string_ or _list_. Default: "'.,!?«»:;()[]-\""

## Methods:
#### get_chunk: returns a chunk of text from the corpus. 
Parameters:
- chunk_len: textual chunk length (_int_). Default: 30
- chunk_mode: defines the modality for getting the textual chunk. If **'normal'** it selects the chunk with _book_sel_ and _chunk_sel_ parames (see later). If 'sequential', the chunk is selected sequentially every time the same modality is called. If 'random', the chunk is chosen randomly.
- book_sel: when chunk_mode is 'normal', defines the document to get the chunk from.
- chunk_sel: when chunk_mode is 'normal', defines the chunk starting point in the selected document.
- padding: when chunk_mode is 'sequential', defines how many steps foward the next chunk will start. Default 0 (full chunk length padding).
- last_element: if True, the chunk length is chunk_len+1, but it doesn't affect the padding.
- output_mode: defines the type of output chunk. It can be: 'item' as a list of sequential words or characters (depending on the Corpy mode); 'string' as a concatenated string of items; **'code'** as a list of numbers defined by the items dictionary.
-  section: the section where the chunk is taken from.
-  unk: the symbol used for Unkown items. In case of output_mode='code': it can be  **'max'**: insert the code=max item, or  'none': insert None.  In case of output_mode='item' or 'string': insert the passed string.

#### string2code(string): converts a string to the associated sequence of code

#### items2code(list): converts a list of items to the associated sequence of code

#### code2items(list): converts a list of numerical codes to the associated sequence of items

#### code2string(list): converts a list of numerical codes to the associated string

#### reset_counter: resets the counter in 'sequential' mode.
Parameters:
- section: it can be **'all'**, it resets to 0 all the sections, or a _int_ defining the section to reset.

#### save: saves the Corpy object in a pickle file.
Parameters:
- namefile: path and file name for saving data. If it is an empty string (default), a unique file name is set.


## Other functions:
#### load_corpy(filename)
load a pikle Corpy file and returns a Corpy object.

## Examples:
See corpy.ipynb Jupyter notebook.