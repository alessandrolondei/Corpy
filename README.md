# Corpy
Corpy is a simple manager for textual corpora written in Python

## Features:
- Handling multiple documents
- Dictionary limited by setting a threshold
- Handling of sections (train, test,...) for ML purposes
- Different modalities to get textual chunks

## Usage
Corpy is a class requiring at least a vaid path containing txt documents when created.

``` corpus = Corpy('./texts') ```

creates a textual corpus by reading all the files contained in the './texts' folder.

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
