#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
This will Parse the defaults.py file into a set of HTML tables for the
documentation. This also fixes the links required for the documentation.

The folder ./Templates/HTML/ contains some skeleton (template) files with the
structure of the HTML files without the details filled in. E.g. A template file
may contain a header tag in the format <h3>*header*<\h3>. Here how the tag
looks is determined by the template file and what the tag says is determined by
the *header* variable. The variable is set in the `replacers' dictionary within
this code. The filepaths for the template files are given in the
`template_filepaths' dictionary. This is auto-generated and shouldn't need
changing.

The important variables here are:
    * replacers => This gives the value of all the variables within the HTML
                    templates.
    * template_filepaths => this specifies where all the template files are
"""
from collections import OrderedDict
import os
import re

from src import IO as io
from src import EXCEPT as EXC

import docstring as docstr

# Define folders
docs_folder = io.folder_correct('./Docs')
templates_folder = io.folder_correct('./Templates/HTML')
static_folder = io.folder_correct(docs_folder+"Static")
tables_folder = io.folder_correct(docs_folder+"Tables")
index_filePath = io.folder_correct('./Documentation.html')


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
                htmlLink = '<code><a href="%s"> %s </a></code>' % (link, param)
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
                                                       sectionName.title())
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

        * getPathOnly = Will only create the filePath for the file
    """
    def __init__(self, templateFilePath,
                 replacers,
                 defaults,
                 getPathOnly=False):

        self.filePath = templateFilePath
        self.replacers = replacers
        self.defaults = defaults
        self.params = defaults.params

        self._getTitle()
        self._determineSavePath(static_folder)

        if getPathOnly is not True:
            self.fileTxt = io.open_read(templateFilePath)
            self.replacers['*topnavStyle*'] = ' '
            self.fileTxt = self._replaceVars(self.fileTxt, self.replacers)

    def _getTitle(self):
        """
        Will get the filename from the full filepath and store it as the title
        """
        self.title = self.filePath[self.filePath.rfind('/') + 1:]
        self.title = self.title.replace(".html", "")

    def _determineSavePath(self, folder):
        """
        Determines where to save the table that is created

        Inputs:
            * folder => the folder to store the HTML files
            * title => the name of the filename
        """
        self.saveFolder = folder
        if not os.path.isdir(self.saveFolder):
            os.makedirs(self.saveFolder)

        fileName = self.title  # .replace(" ", "_")
        self.savePath = self.saveFolder + fileName + '.html'

    def _replaceVars(self, txt, replacers):
        """
        Will replace the variables in the file text (self.fileTxt) with the
        relevant variable in replacers
        """
        allVars = re.findall(r'\*[a-zA-Z_]+.?\*', txt)
        repl = {i: replacers[i] for i in replacers}  # Don't change dictionary

        # Error Checking
        for var in allVars:
            if var not in repl:
                msg = "Can't find the variable %s" % var
                msg += " in the replacers dictionary.\n\n"
                msg += "This is found in the file %s\n\n\n" % self.filePath
                raise SystemExit(msg)

        for var in allVars:
            varInVar = re.findall(r'\*[a-zA-Z_]+.?\*', replacers[var])
            if varInVar:
                repl[var] = self._replaceVars(repl[var],
                                              repl)
            txt = txt.replace(var, repl[var])
        return txt


class TableHTMLFile(HTMLFile):
    """
    Special case HTML file creator for the table files. Will create the tables
    in the table files.

    Inputs:
        * templateFilePath => The filepath of the table template file

        * sectionParams => The parameters in the section of the defaults.py
                            file

        * title => The title of the section of the defaults.py file.

        * replacers => the replacers dictionary

        * getPathOnly = Will only create the filePath for the file
    """
    def __init__(self, templateFilePath,
                 sectionParams,
                 title,
                 replacers,
                 getPathOnly=False):

        self.filePath = templateFilePath  # The template HTML of the table
        self.title = title  # What the section is called
        self.params = sectionParams  # All the parameters in the table

        # Borrowed from Parent (HTMLFile)
        self._determineSavePath(tables_folder)  # Sets the self.savePath

        if getPathOnly is not True:
            self.fileTxt = io.open_read(self.filePath)   # Text in table file
            self._create_table()

            repl = {i: replacers[i] for i in replacers}
            repl['*table_data*'] = self.tableTxt
            repl['*table_name*'] = self.title.title()
            repl['*topnavStyle*'] = 'style="margin-left: 205px; width:100%;"'
            self.fileTxt = self._replaceVars(self.fileTxt, repl)

    def _create_table(self):
        """
        Will create the table (based on the template given in the )
        """
        table_tag = "<table id=table1>\n"

        self.tableTxt = table_tag
        allHeaders = ['Setting', 'Default Value', 'Description', 'Input Type']
        for header in allHeaders:
            self.tableTxt += '\t<th> %s </th>' % header
        for setting in self.params:
            self.tableTxt += "\n<tr>"
            self.tableTxt += "<td> %s </td>" % setting
            for header in self.params[setting]:
                if header != 'tested':
                    self.tableTxt += self._createTd(
                                                   self.params[setting][header]
                                                   )
            self.tableTxt += '\n</tr>\n'
        self.tableTxt += "\n</table>\n</br>\n</br>"

    def _createTd(self, val):
        """
        Will create a table data string for the particular value. E.g. if the
        value was 'bob' then this function would create a <td> bob <\td>
        """
        if any(type(val) == typ for typ in (int, str)):
            strVal = str(val)
        elif type(val) == float:
            strVal = "%.3g" % val
        elif type(val) == list:
            strVal = "<ul>"
            for item in val:
                strVal += "<li> %s </li>\n" % item
            strVal += "</ul>"
        elif type(val) == tuple:
            strVal = "(%s)" % ', '.join([str(i) for i in val])
        elif type(val) == bool:
            strVal = '<span id=false> %s </span>' % val
            if val:
                strVal = r'<span id=true> %s </span>' % val
        elif type(val) == dict:
            strVal = "\n<table>"
            strVal += '<th> Key </th>  <th> Value </th>\n'
            for key in val:
                printer = self._createTd(val[key]).replace('<td>', '')
                printer = printer.replace('</td>', '')
                printer = printer.strip()
                strVal += "\n<tr> <td> %s </td>" % key
                strVal += "<td> %s </td> </tr>" % printer
            strVal += '</table>'
        elif val is None:
            strVal = 'None'
        else:
            raise SystemExit("Sorry Can't decied what to do with type:" +
                             "%s" % type(val))

        return "\t<td> %s </td>" % strVal


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
             "*Misc*": io.open_read(templates_folder+'IntroMisc.html'),
             "*Pagetitle*": "Movie Maker Documentation",
             "*header_text*": io.open_read(templates_folder+'HeaderTxt.html'),
             "*top_nav*": io.open_read(templates_folder+'TopNav.html'),
             }

# Handle the Sidebar
SideBar(defaults, replacers, tables_folder)

# Handle the docstrings on top of the python files
dstr = docstr.Docstr_Parsing('.')
replacers['*Mov_Mak_Edit*'] = dstr.docstrTxt

# Complete the files in order (do the TopNav first)
firstItems = ['TopNav', 'HeaderTxt', 'QuickStart']
lastItems = ['table']
templateFilePathsOrder = firstItems
for i in template_filepaths:
    if i not in templateFilePathsOrder:
        templateFilePathsOrder.append(i)
for i in lastItems:
    templateFilePathsOrder.remove(i)
templateFilePathsOrder += lastItems

FilesToNotWrite = ['IntroMisc', 'TopNav', 'HeaderTxt', 'EditDocumentation', 
                   'QuickStart']

# First create all the correct paths to the files
for key in templateFilePathsOrder:
    if key not in FilesToNotWrite:
        if 'table' in key:
            for section in defaults.params:
                tmp = TableHTMLFile(template_filepaths[key],
                                    defaults.params[section],
                                    section,
                                    replacers,
                                    True)

        else:
            tmp = HTMLFile(template_filepaths[key],
                           replacers,
                           defaults,
                           True)

        fName = tmp.savePath
        replacers['*%s*' % tmp.title] = tmp.savePath
replacers['*index*'] = index_filePath

# Actually parse and replace the variables in the html files
filesToWrite = {}
for key in templateFilePathsOrder:
    if 'table' in key:
        for section in defaults.params:
            tmp = TableHTMLFile(template_filepaths[key],
                                defaults.params[section],
                                section,
                                replacers)
            filesToWrite[tmp.title] = tmp
#
#    elif 'TopNav' in key:
#        replacers['*top_nav*'] = HTMLFile(template_filepaths[key],
#                                          replacers,
#                                          defaults).fileTxt
#
#    elif 'HeaderTxt' in key:
#        replacers['*header_text*'] = HTMLFile(template_filepaths[key],
#                                              replacers,
#                                              defaults).fileTxt

    else:
        tmp = HTMLFile(template_filepaths[key],
                       replacers,
                       defaults)
        filesToWrite[tmp.title] = tmp


for key in filesToWrite:
    tmp = filesToWrite[key]
    fileName = tmp.savePath
    with open(fileName, 'w') as f:
        f.write(tmp.fileTxt)

os.rename(filesToWrite['index'].savePath, index_filePath)
