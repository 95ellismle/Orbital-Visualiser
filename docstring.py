#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Will parse the docstring from the individual python files.
"""

import os


class Docstr_Parsing(object):
    """
    Will parse the docstrings on top of the python files and also the README.md
    files in each folder together to make the how_to_edit documentation.

    Inputs:
        * rootFolder => the folder to look for all the python files in

    The docstrings are all stored in `self.docstr_dict' this is divded up into
        directories and then files.

    The docstring are parsed and written into a variable named `self.docstrTxt'
        this is what should be inputted as the *Mov_Mak_Edit* var in the
        How_to_edit.html file.
    """

    def __init__(self, rootFolder):
        # Save as list to use multiple times
        Docstr_Parsing.all_files = list(os.walk(rootFolder))
        Docstr_Parsing._make_docstr_dict(self)
        Docstr_Parsing._get_README_files(self)

        self.docstrTxt = "Still need to create the string. See docstring.py"

    def _get_docstr(self, fname):
        """
        Will get the docstring on the top of one of the python files.
        """
        with open(fname, 'r') as f:
            for line in f:
                if '"""' in line:
                    delim = '"""'
                    break
                elif "'''" in line:
                    delim = "'''"
                    break
            else:
                print("no delim in %s\n\nskipping..." % fname)
                return ""
            docstr = f.read().split(delim)[0]
        return docstr

    @staticmethod
    def _get_README_files(self):
        """
        Will get all of the README files from each folder
        """
        all_folders = [i for i in self.docstr_dict
                       if os.path.isdir(i) and i != '.']
        for fold in all_folders:
            README_file = fold+'/README.md'
            if os.path.isfile(README_file):
                with open(README_file, 'r') as f:
                    self.docstr_dict[fold]['README'] = f.read().replace("\n",
                                                                        "")

    @staticmethod
    def _make_docstr_dict(self):
        """
        Will make the dictionary that holds all the docstrings for each file.
        """
        self.docstr_dict = {}
        for dpath, dnames, fnames in Docstr_Parsing.all_files:
            for fname in fnames:
                if '.py' in fname and 'pyc' not in fname:
                    fpath = dpath + '/' + fname
                    docstr = self._get_docstr(fpath)
                    if dpath not in self.docstr_dict:
                        self.docstr_dict[dpath] = {}
                    self.docstr_dict[dpath][fpath] = docstr
