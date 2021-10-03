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
        Docstr_Parsing._create_docstr_HTML(self)

    def _get_docstr(self, fname):
        """
        Will get the docstring on the top of one of the python files.

        #TODO: change this to just use the python in-built
        """
        with open(fname, 'r') as f:
            ftxt = f.read().split('\n')

        for line in ftxt:
            if '"""' in line:
                delim = '"""'
                break
            elif "'''" in line:
                delim = "'''"
                break
        else:
            print("no delim in %s\n\nskipping..." % fname)
            return ""

        docstr = '\n'.join(ftxt).split(delim)[1]
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

    @staticmethod
    def _create_docstr_HTML(self):
        """
        Will create a nice HTML file from the information in the docstrings
        """
        self.docstrTxt = ''
        for folder in self.docstr_dict:
            fold = folder
            if folder == '.':
                fold = "Root Folder"
            self.docstrTxt += "<div>"
            folderHTML = '<span style="color: darkred;"> %s </span>' % fold
            self.docstrTxt += "<h4> Folder: %s </h4> </br>" % folderHTML
            self.docstrTxt += "<h5> What is contained in the folder </h5>"

            Docstr_Parsing.__createFolderExplan(self, folder)

            self.docstrTxt += "<h5> The files: </h5>"
            self.docstrTxt += "<table id=\"files\" cellspacing=\"0\"> <th>File</th> <th>Explanation</th>"
            for file in self.docstr_dict[folder]:
                if 'README' in file or '__init__.py' in file:
                    continue
                expl = self.docstr_dict[folder][file]
                self.docstrTxt += "<tr>"
                self.docstrTxt += "<td>%s</td>  <td>%s</td>" % (file, expl)
                self.docstrTxt += "</tr>"
#                self.docstrTxt += "<tr><td>. </td><td> </td></tr>"
            self.docstrTxt += "</table>"

            self.docstrTxt += "</div>"
            self.docstrTxt += "</br></br>"

    @staticmethod
    def __createFolderExplan(self, folder):
        """
        Will create the folder explanation
        """
        if folder != '.':
            if 'README' not in self.docstr_dict[folder]:
                keys = self.docstr_dict[folder].keys()
                print("ERROR: README file required in the following folder.")
                print("Can't find the README file in the folder:")
                print("\t'%s'" % folder)
                print("Files Found: [%s]" % ', '.join(keys))
                print("Missing README")
                self.docstrTxt += "No README.md file in folder..."

            else:
                self.docstrTxt += self.docstr_dict[folder]['README']
        else:
            self.docstrTxt += "This is the root folder where you will find"
            self.docstrTxt += " the most important files in running the "
            self.docstrTxt += "code. These are:"
            self.docstrTxt += "<ul><li>main.py</li>"
            self.docstrTxt += "<li>Settings.inp</li>"
            self.docstrTxt += "<li>Create_docs.py</li></ul>"


