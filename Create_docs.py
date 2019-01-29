#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This will Parse the defaults.py file into a set of HTML tables for the
documentation. This also fixes the links required for the documentation.

N.B. At the momement this file is quite a mess it definitely needs a lot of
tidying!
"""
from collections import OrderedDict
import os
import re

from src import IO as io
from src import EXCEPT as EXC

import docstring as docstr


class ParseDefaults(object):
    """
    Will parse the default settings file. Stores all the parsed setting in a
    dictionary called self.params. This is divided up by the section header in
    the defaults dict file.

    This class will also create the filepaths for the table HTML files

    Inputs:
        * default settings filepath (filepath to the defaults.py file)
        * HTMLFiles = the dictionary storing the filepaths for the HTML files

    (No Public Functions)
    """
    def __init__(self, filePath):
        self.fPath = filePath
        self.params = OrderedDict()
        self.fTxt = io.open_read(filePath)

        # Ignore the docstring at the top and get only the defaults dictionary
        defaultsTxt = self._getDefaultsDict()
        self._separateSections(defaultsTxt)  # Get sections in defaults dict

        # Parse each line
        remKeys = []
        for section in self.params:
            if section.strip():
                for line in self.params[section]['ltxt']:
                    setting, params = self._parseLine(line)
                    if setting:
                        self.params[section][setting] = params
                self.params[section].pop('ltxt')
            else:
                remKeys.append(section)

        # Remove redundant keys
        for key in remKeys:
            self.params.pop(key)

    def _getDefaultsDict(self):
        """
        Will find the defaults dictionary in the fTxt
        """
        ltxt = self.fTxt.split('\n')
        for linei, line in enumerate(ltxt):

            words = [i.strip() for i in line.split(' ')]
            if all(word in words for word in ('defaults', '=', '{')):
                return ltxt[linei+1:]
        else:
            msg = "\nCouldn't find the declaration of the `defaults' variable!"
            msg += "\n\nMake sure there is a line in the defaults.py file with"
            msg += " the strings 'defaults', '=' and '{'"
            raise SystemExit(msg)

    def _separateSections(self, dictTxt):
        """
        Will find each section in the defaults dict and add these sections to
        the self.params dictionary
        """
        dictTxt = '\n'.join(dictTxt)
        for section in dictTxt.split('##'):
            ltxt = section.split('\n')
            sectionName = ltxt[0].strip()
            self.params[sectionName] = OrderedDict()
            self.params[sectionName]['ltxt'] = ltxt[1:]

    def _parseLine(self, line):
        """
        Will just parse a single line and add the settings to the params dict
        """
        cond = line.strip() and ':' in line
        if cond:
            setting, rest = self.__getSetting(line)
            value, rest = self.__getDefaultValue(rest)
            explan, options, tested = self.__getExplanationOptionsTested(rest)
            paramsDict = {'default': value,
                          'explanation': explan,
                          'options': options,
                          'tested': tested}
            return setting, paramsDict

        return False, False

    def __getSetting(self, line):
        """
        Will get the setting from the line in the defaults dictionary. This is
        the first word behind the ' : ' character.
        """
        # First get the setting
        setting, rest = self.__split_by(line,
                                        ' : ',
                                        2,
                                        True,
                                        2)
        # Remove the string marks from the setting
        setting = setting.strip().strip("'").strip('"')
        return setting, rest

    def __getDefaultValue(self, line2):
        """
        Will get the default value of the setting for a line in the defaults.py
        defaults dict.

        line2 = the line without the setting in (use __getSetting first)
        """
        value, rest = self.__split_by(line2,
                                      ', #',
                                      2,
                                      True,
                                      2)
        return eval(value), rest

    def __getExplanationOptionsTested(self, line3):
        """
        Will parse the explanations of the setting, the Options available to
        set it and whether it's been tested or not.
        """
        tmp = self.__split_by(line3,
                              '|',
                              3,
                              False,
                              1)
        if len(tmp) == 3:
            explan, options, tested = tmp
            tested = 'not' not in tested
            return explan, options, tested

        elif any('DEPRECATED' in word.strip() for word in tmp):
            return False, False, False
        else:
            msg = "Sorry I don't know how to find the explanation, options and"
            msg += " whether it has been tested or not from this:\n\t"
            msg += line3
            raise SystemExit(msg)

    def __split_by(self, line, by, length=2, congeal_last=False, min_lines=2):
        """
        Split a string ('line') by the 'by' variable.

        Inputs:
            * line = line to split
            * by   = the splitter
            * length = the max length the split list should be
            * congeal_last = Whether to join all but the 1st item in the split
                             list
            * min_lines = minimum length of the lines
        """
        split = line.split(by)
        if len(split) < min_lines:
            EXC.ERROR("""

            ERROR: The length of line (%s) split by '%s' is %i, it should be 2.

            This is probably due to something being entered wrong in the
            Templates/defaults file.

            Each line needs to have the format:
            'setting' : 'default' , # Explanation | ['list of accepted settings'] | 'not-tested' or 'tested'

            In the Templates/defaults.py file look for the line:
            \t'%s'

            and check the subtring:
            \t'%s'
            if there.
            """ % (line, by, len(split), line, by))
        if len(split) > length:
            msg = "\n\nWarning docs entry entered incorrectly.\nDetails:"
            msg += "\n\t* Line = %s" % line
            msg += "\n\t*Length after split by %s %i" % (by, len(split))
            EXC.ERROR(msg)
        if congeal_last:
            split = split[0], by.join(split[1:])
        return split


class SideBar(object):
    """
    Will add the parameters to the replacers dictionary (*var* = param[var]).
    Will Create the sidebar text and the sidebar filepaths

    Inputs:
        * defaults = the ParseDefaults object that parses the default settings
                      file
        * replacers = replacer dictionary to add the parameters to
    """
    def __init__(self, defaults, replacers, tablesFolder):
        self.params = defaults.params
        self.replacers = replacers
        self.tablesFolder = tablesFolder

        self._createSidebar()
        self._addParametersToReplacers()

    def _addParametersToReplacers(self):
        """
        Will add the parameters to the replacers dictionary.
        """
        for section in self.params:
            link = self.tableFilePaths[section]

            for param in self.params[section]:
                htmlLink = '<a href="%s"> %s </a>' % (link, param)
                self.replacers['*%s*' % param] = htmlLink

    def _createSidebar(self):
        """
        Create the sidebar text to use in the HTML files. Also creates the dict
        of filepaths.
        """
        self.tableFilePaths = {}

        sidebarTxt = ''
        for sectionName in self.params:
            tableFilePath = '%s%s.html' % (self.tablesFolder,
                                           sectionName)
            sidebarTxt += '<li>\n'
            sidebarTxt += '%s<a href="%s"> %s </a>' % ("\t" * 5,
                                                       tableFilePath,
                                                       sectionName)
            sidebarTxt += '\n</li>\n'
            self.tableFilePaths[sectionName] = tableFilePath

        self.replacers['*sidebar_text*'] = sidebarTxt


class HTMLFile(object):
    """
    Will create a HTML file from the template file given. This will involve
    replacing the *vars* using the replacer dictionary and fixing any links
    using the html filepaths dictionary.

    Inputs:
        * templateFilePath = The path to the template file
                               (e.g. in Templates/HTML/*.html)

        * replacers = the dictionary which determines what the strings that
                       follow the *var* patter are replaced with.

        * htmlFiles = the dictionary containing the links to replace the
                       relative links with
    """
    def __init__(self, templateFilePath, replacers, defaults):
        self.filePath = templateFilePath
        self.replacers = replacers
        self.fileTxt = io.open_read(templateFilePath)
        self.defaults = defaults
        self.params = defaults.params

        self._replaceVars()

    def _replaceVars(self):
        """
        Will replace the variables in the file text (self.fileTxt) with the
        relevant variable in replacers
        """
        allVars = re.findall(r'\*[a-zA-Z_]+.?\*', self.fileTxt)

        # Error Checking
        for var in allVars:
            if var not in self.replacers:
                msg = "Can't find the variable %s" % var
                msg += " in the replacers dictionary.\n\n"
                msg += "This is found in the file %s\n\n\n" % self.filePath
                raise SystemExit(msg)

        for var in allVars:
            self.fileTxt = self.fileTxt.replace(var,
                                                self.replacers[var])


class TableHTMLFile(HTMLFile):
    """
    Special case HTML file creator for the table files. Will create the tables
    in the table files.
    """
    def __init__(self, tableFilePath, sectionParams):
        self.filePath = tableFilePath
        self.fileTxt = io.open_read(self.filePath)
        self.params = sectionParams


section_header = 'h3'
table_tag = "<table id=table1>"

# Define folders
docs_folder = io.folder_correct('./Docs')
templates_folder = io.folder_correct('./Templates/HTML')
static_folder = io.folder_correct(docs_folder+"Static")
tables_folder = io.folder_correct(docs_folder+"Tables")


# The filepaths to the files acting as the templates to the webpages
template_filepaths = {}
for f in os.listdir(templates_folder):
    fName = f.replace('.html', '')
    template_filepaths[fName] = templates_folder + f

# The default settings filepath
defaults_filepath = io.folder_correct('./Templates/') + 'defaults.py'

defaults = ParseDefaults(defaults_filepath)
# A dictionary of things to find in the HTML files and replace
replacers = {"*doc_img_folder*": io.folder_correct(docs_folder+"img"),
             "*docs_folder*": docs_folder,
             "*vendor_folder_path*": io.folder_correct(docs_folder+"vendor"),
             "*css_folder_path*": io.folder_correct(docs_folder+"css"),
             "*quick_start*": io.open_read(templates_folder+"QuickStart.html"),
             "*intro_text*": io.open_read(templates_folder+"Intro.html"),
             "*header_text*": io.open_read(templates_folder+"HeaderTxt.html"),
             "*Misc*": io.open_read(templates_folder+'IntroMisc.html'),
             "*title*": "Movie Maker Documentation",
             "*index_file*": io.folder_correct("./Documentation"),
             "*how_it_works_file*":
                 io.folder_correct("./Docs/Static/HowItWorks.html"),
             "*how_to_edit_file*":
                 io.folder_correct("./Docs/Static/HowToEdit.html"),
             }

# Handle the Sidebar
SideBar(defaults, replacers, tables_folder)
dstr = docstr.Docstr_Parsing('.')

# Must do the TopNav bar first
templateFilePathsOrder = ['TopNav']
for i in template_filepaths:
    if i not in templateFilePathsOrder:
        templateFilePathsOrder.append(i)

filesToWrite = {}
for key in templateFilePathsOrder:
    if 'table' in key:
        for section in defaults.params:
            filesToWrite[section] = TableHTMLFile(template_filepaths[key],
                                                  defaults.params[section]
                                                  )
    else:
        filesToWrite[key] = HTMLFile(template_filepaths[key],
                                     replacers,
                                     defaults)

    if 'TopNav' in key:
        replacers['*top_nav*'] = filesToWrite[key].fileTxt
