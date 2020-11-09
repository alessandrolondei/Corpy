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
- one_document: concatenates all the documents into one (space separator). Vaues: True, *False*

