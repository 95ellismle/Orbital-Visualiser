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

from src import IO as io
from src import EXCEPT as EXC


def replace_filepaths_in_template(template_text, replacers):
    """
    """
    for replace in replacers:
        template_text = template_text.replace(replace, replacers[replace])
    return template_text





section_header = 'h3'
table_tag = "<table id=table1>"

# Define folders
docs_folder = io.folder_correct('./Docs')
templates_folder = io.folder_correct('./Templates/HTML')
static_folder = io.folder_correct(docs_folder+"Static")

# Where to save the final HTML files
html_filepaths = {'table': io.folder_correct(docs_folder + "tables/",
                                             make_file=True),
                  'index': io.folder_correct("./Documentation.html"),
                  'how_it_works': io.folder_correct(static_folder +
                                                    "How_it_works.html"),
                  'FAQs': io.folder_correct(static_folder +
                                            "FAQs.html"),
                  'How_to_edit': io.folder_correct(static_folder +
                                                   "How_to_edit.html"),
                  'examples': io.folder_correct(static_folder +
                                                "Examples.html")}

# The filepaths to the files acting as the templates to the webpages
template_filepaths = {}
for f in os.listdir(templates_folder):
    fName = f.replace('.html', '')
    template_filepaths[fName] = templates_folder + f

# A dictionary of things to find in the HTML files and replace
replacers = {"*index_file*": html_filepaths['index'],
             "*doc_img_folder*": io.folder_correct(docs_folder+"img"),
             "*how_it_works*": html_filepaths['how_it_works'],
             "*How_to_edit*": html_filepaths["How_to_edit"],
             "*examples*": html_filepaths['examples'],
             "*FAQs*": html_filepaths['FAQs'],
             "*docs_folder*": docs_folder,
             "*vendor_folder_path*": io.folder_correct(docs_folder+"vendor"),
             "*css_folder_path*": io.folder_correct(docs_folder+"css"),
             "*quick_start*": io.open_read(static_folder+"Quick_Start.html"),
             "*intro_text*": io.open_read(static_folder+"Intro.html"),
             "*header_text*": io.open_read(static_folder+"Header_text.html"),
             "*top_nav*": io.open_read(static_folder+"TopNav.html"),
             "*sidebar_text*": "",
             "*title*": "Movie Maker Documentation",
             }

# The default settings filepath
defaults_filepath = io.folder_correct('./Templates/') + 'defaults.py'


class ParseDefaults(object):
    """
    Will parse the default settings file. Stores all the parsed setting in a
    dictionary called self.params. This is divided up by the section header in
    the defaults dict file

    Inputs:
        * default settings filepath (filepath to the defaults.py file)

    (No Public Functions)
    """
    def __init__(self, filePath):
        self.fPath = filePath
        self.params = OrderedDict()
        self.fTxt = io.open_read(filePath)

        # Ignore the docstring at the top and get only the defaults dictionary
        defaultsTxt = self._getDefaultsDict()
        self._separateSections(defaultsTxt)  # Get sections in defaults dict

        for section in self.params:
            for line in self.params[section]['ltxt']:
                setting, params = self._parseLine(line)
                if setting:
                    self.params[section][setting] = params
            self.params[section].pop('ltxt')

    def _getDefaultsDict(self):
        """
        Will find the defaults dictionary in the fTxt
        """
        ltxt = self.fTxt.split('\n')
        for linei, line in enumerate(ltxt):

            words = [i.strip() for i in line.split(' ')]
            if all(word in words for word in ('defaults', '=', '{')):
                return ltxt[linei:]
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
        setting, rest = self._split_by(line,
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
        value, rest = self._split_by(line2,
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
        tmp = self._split_by(line3,
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

    def _split_by(self, line, by, length=2, congeal_last=False, min_lines=2):
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


defaults = ParseDefaults(defaults_filepath)

#for section in params:
#    for param in params[section]:
#        setting, rest_of_stuff = split_by(param, ' : ', 2)
#        setting = setting.strip().replace("'", '')
#        tmp = split_by(rest_of_stuff, ', #', 100, True)
#        default_value, rest_of_stuff = tmp
#        try:
#            if type(eval(default_value)) == dict:
#                default_value = eval(default_value)
#                s = "%s\n" % "<table class=subtable>"
#                s += """
#                <tr>
#                    <th> <font size="2.5"> Key </font> </th>
#                    <th> <font size="2.5"> Value </font> </th>
#                </tr>
#                """
#                for key in default_value:
#                    s += """
#                <tr>
#                    <td> <font size="2"> %s </font> </td>
#                    <td> <font size="2"> %s </font> </td>
#                </tr>""" % (str(key), str(default_value[key]))
#                default_value = s + "</table>"
#        finally:
#            pass
#        if "DEPRECATED" not in rest_of_stuff:
#            tmp = rest_of_stuff.replace('#', '').strip().split('|')
#            description, types, tested = tmp
#            types = eval(types)
#            tested = ('not' not in tested)
#            params_parsed[section][setting]['default'] = default_value
#            params_parsed[section][setting]['tested'] = tested
#            params_parsed[section][setting]['types'] = types
#            params_parsed[section][setting]['desc'] = description
#        else:
#            params_parsed[section].pop(setting, None)
#
#all_variables_which_section = {}
#for section in params_parsed:
#    for param in params_parsed[section]:
#        all_variables_which_section[param] = section
#
#table_filepaths = {i: io.folder_correct(html_filepaths['table'] +
#                                        str(i) +
#                                        ".html")
#                   for i in params_parsed}
#for section in params_parsed:
#    replacers['*sidebar_text*'] += """
#                <li>
#                    <a href="%s"> %s </a>
#                </li>""" % (table_filepaths[section], section)
#
#replacers['*header_text*'] = replace_filepaths_in_template(replacers['*header_text*'],
#                                                           replacers)
#replacers['*top_nav*'] = replace_filepaths_in_template(replacers['*top_nav*'],
#                                                       replacers)
#for key in template_data:
#    template_data[key] = replace_filepaths_in_template(template_data[key],
#                                                       replacers)
#
#for variable in all_variables_which_section:
#    for section in template_data:
#        link_txt = '''<a href="%s" class="param_link">
#        <code  class="CODE"> %s </code>
#        </a>''' % (table_filepaths[all_variables_which_section[variable]],
#                   variable)
#        template_data[section] = template_data[section].replace("*%s*" % variable,
#                                                                link_txt)
#
## Write html files which aren't a table file
#for i in html_filepaths:
#    if 'tab' not in i:
#        io.open_write(html_filepaths[i], template_data[i])
#
## Change bool defaults to yes or no
#for section in params_parsed:
#    for param in params_parsed[section]:
#        if params_parsed[section][param]['default'].strip() == 'True':
#            params_parsed[section][param]['default'] = "'yes'"
#        elif params_parsed[section][param]['default'].strip() == 'False':
#            params_parsed[section][param]['default'] = "'no'"
#
## Create and write the tables
#for table_title in table_filepaths:
#    table_file_text = template_data['table']
#    table_file_text = table_file_text.replace("*table_name*", table_title)
#    s = """%s""" % table_tag
#    s += """<tr>
#    <th>
#      Setting
#    </th>
#    <th>
#      Default Value
#    </th>
#    <th>
#      Description
#    </th>
#    <th>
#      Input Type
#    </th>
#  </tr>\n"""
#    for param in params_parsed[table_title]:
#        type_str = """
#        <ul>"""
#        for T in params_parsed[table_title][param]['types']:
#            type_str += """
#            <li>  %s </li>""" % str(T)
#        type_str += """
#        </ul>"""
#        s += """<tr>
#    <td>
#      %s
#    </td>
#    <td>
#      %s
#    </td>
#    <td>
#      %s
#    </td>
#    <td>
#      %s
#    </td>
#  </tr>\n""" % (param,
#                params_parsed[table_title][param]['default'],
#                params_parsed[table_title][param]['desc'],
#                type_str)
#
#    s += "</table>\n\n<br>\n<br>\n\n"
#    table_file_text = table_file_text.replace("*table_data*", s)
#
#    io.open_write(table_filepaths[table_title], table_file_text)
#
