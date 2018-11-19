#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Will parse the docstring from the individual python files.
"""

import os

def get_docstr(fname):
    """
    Will get the docstring on the top of one of the python files.
    """
    with open(fname, 'r') as f:
#        txt = f.read()
#        ltxt = txt.split('\n')
#        delim = ltxt[0]
        for line in f:
            if '"""' in line:
                delim = '"""'
                break
            elif "'''" in line:
                delim = "'''"
                break
        else:
            print("no delim in %s"%fname)
        docstr = f.read().split(delim)[0]
    return docstr



all_files = os.walk('.')
for dpath, dnames, fnames in all_files:
    for fname in fnames:
        if '.py' in fname and 'pyc' not in fname:
            fpath = dpath +'/' + fname
            docstr = get_docstr(fpath)
