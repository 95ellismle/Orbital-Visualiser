#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This will Parse the defaults.py file into a set of HTML tables for the
documentation.

N.B. At the momement this file is quite a mess it probably needs a lot of
tidying.
"""
from collections import OrderedDict
# import os

from src import IO as io
from src import EXCEPT as EXC
# from src import type as typ


section_header = 'h3'
table_tag = "<table id=table1>"



docs_folder = io.folder_correct('./Docs')
templates_folder = io.folder_correct('./Templates/HTML')
static_folder = io.folder_correct(docs_folder+"Static")

html_filepaths = { 'table': io.folder_correct(docs_folder+"tables/", make_file=True),
'index': io.folder_correct("./Documentation.html"),
'how_it_works': io.folder_correct(static_folder+"How_it_works.html"),
'FAQs': io.folder_correct(static_folder+"FAQs.html"),
'examples': io.folder_correct(static_folder+"Examples.html"),}

template_filepaths = {'table' : io.folder_correct(templates_folder+"table.html"),
'index' : io.folder_correct(templates_folder+"index.html"),
'how_it_works' : io.folder_correct(templates_folder+"How_it_works.html"),
'FAQs' : io.folder_correct(templates_folder+"FAQs.html"),
'examples' : io.folder_correct(templates_folder+"Examples.html"),}

replacers = {"*index_file*":html_filepaths['index'],
             "*doc_img_folder*":io.folder_correct(docs_folder+"img"),
             "*how_it_works*": html_filepaths['how_it_works'],
             "*examples*": html_filepaths['examples'],
             "*FAQs*": html_filepaths['FAQs'],
             "*docs_folder*" : docs_folder,
             "*vendor_folder_path*":io.folder_correct(docs_folder+"vendor"),
             "*css_folder_path*":io.folder_correct(docs_folder+"css"),
             "*quick_start*":io.open_read(static_folder+"Quick_Start.html"),
             "*intro_text*":io.open_read(static_folder+"Intro.html"),
             "*header_text*":io.open_read(static_folder+"Header_text.html"),
             "*sidebar_text*":"",
             "*title*":"Movie Maker Documentation",
            }

template_data = {i:io.open_read(template_filepaths[i], False) for i in template_filepaths}
for i in template_filepaths:
    if not io.path_leads_somewhere(template_filepaths[i]):
        EXC.ERROR("%s doesn't lead anywhere! This is a necessary template file!"%str(template_filepaths[i]))

defaults_filepath = io.folder_correct('./Templates/defaults.py')
defaults_data = io.open_read(defaults_filepath).split('\n')
defaults_data = [i for i in defaults_data if i][1:-2]
params = OrderedDict()
section_indices = [i for i in range(len(defaults_data)) if '##' in defaults_data[i]]
for prv, nxt in zip(section_indices[:-1], section_indices[1:]):
    params[defaults_data[prv].strip('##').strip()] = defaults_data[prv+1:nxt]
params[defaults_data[nxt].strip('##').strip()] = defaults_data[nxt+1:]


def split_by(line, by, length=2, congeal_last=False, min_lines=2):
    split = line.split(by)
    if len(split) < min_lines:
        EXC.ERROR("""

        ERROR: The length of line (%s) split by '%s' is %i, it should be 2.

        This is probably due to something being entered wrong in the Templates/defaults file.
        Each line needs to have the format:
        \t 'setting' : 'default' , # Explanation | ['list of accepted settings'] | 'not-tested' or 'tested'

        In the Templates/defaults.py file look for the line:
        \t'%s'

        and check the subtring:
        \t'%s'
        if there.
        """%(line,by,len(split),line,by))
    if len(split) > length:
        EXC.ERROR("\n\nWarning docs entry entered incorrectly.\nDetails:\n\t* Line = %s\n\t*Length after split by %s %i"%(line,by,len(split)))
    if congeal_last:
        split = split[0],by.join(split[1:])
    return split

second_layer_keys = {i:[] for i in params}
for section in params:
    for param in params[section]:
        setting, rest_of_stuff = split_by(param,' : ',2)
        setting = setting.strip().replace("'",'')
        second_layer_keys[section].append(setting)

params_parsed = OrderedDict()
for sect in params:
    params_parsed[sect] = {}
    for setting in second_layer_keys[sect]:
        params_parsed[sect][setting] = {'default':''}


for section in params:
    for param in params[section]:
        setting, rest_of_stuff = split_by(param,' : ',2)
        setting = setting.strip().replace("'",'')
        default_value, rest_of_stuff = split_by(rest_of_stuff, ', #',100,True)
        try:
            if type(eval(default_value)) == dict:
                default_value = eval(default_value)
                s = "%s\n"%"<table class=subtable>"
                s += """
                <tr>
                    <th> <font size="2.5"> Key </font> </th>
                    <th> <font size="2.5"> Value </font> </th>
                </tr>
                """
                for key in default_value:
                    s += """
                <tr>
                    <td> <font size="2"> %s </font> </td>
                    <td> <font size="2"> %s </font> </td>
                </tr>"""%(str(key), str(default_value[key]))
                default_value = s + "</table>"


        finally:
            pass
        if "DEPRECATED" not in rest_of_stuff:
            description, types, tested = rest_of_stuff.replace('#','').strip().split('|')
            types = eval(types)
            tested = ('not' not in tested)
            params_parsed[section][setting]['default'] = default_value
            params_parsed[section][setting]['tested']  = tested
            params_parsed[section][setting]['types']   = types
            params_parsed[section][setting]['desc']    = description
        else:
            params_parsed[section].pop(setting,None)
all_variables_which_section = {}
for section in params_parsed:
    for param in params_parsed[section]:
        all_variables_which_section[param] = section





table_filepaths = {i:io.folder_correct(html_filepaths['table']+str(i)+".html") for i in params_parsed}
for section in params_parsed:
    replacers['*sidebar_text*'] += """
                <li>
                    <a href="%s"> %s </a>
                </li>"""%(table_filepaths[section], section)

def replace_filepaths_in_template(template_text, replacers):
    for replace in replacers:
        template_text = template_text.replace(replace, replacers[replace])
    return template_text

replacers['*header_text*'] = replace_filepaths_in_template(replacers['*header_text*'],replacers)
for key in template_data:
    template_data[key] = replace_filepaths_in_template(template_data[key],replacers)

for variable in all_variables_which_section:
    for section in template_data:
        link_txt = '''<a href="%s" class="param_link">
        <code  class="CODE"> %s </code>
        </a>'''%(table_filepaths[all_variables_which_section[variable]],variable)
        template_data[section] = template_data[section].replace("*%s*"%variable, link_txt)

for i in html_filepaths:
    if 'tab' not in i:
        io.open_write(html_filepaths[i], template_data[i])

# Change bool defaults to yes or no
for section in params_parsed:
    for param in params_parsed[section]:
        if params_parsed[section][param]['default'].strip() == 'True':
            params_parsed[section][param]['default'] = "'yes'"
        elif params_parsed[section][param]['default'].strip() == 'False':
            params_parsed[section][param]['default'] = "'no'"


for table_title in table_filepaths:
    table_file_text = template_data['table']
    table_file_text = table_file_text.replace("*table_name*",table_title)
    s = """%s"""%table_tag
    s += """<tr>
    <th>
      Setting
    </th>
    <th>
      Default Value
    </th>
    <th>
      Description
    </th>
    <th>
      Input Type
    </th>
  </tr>\n"""
    for param in params_parsed[table_title]:
        type_str = """
        <ul>"""
        for T in params_parsed[table_title][param]['types']:
            type_str += """
            <li>  %s </li>"""%( str(T) )
        type_str += """
        </ul>"""
        s += """<tr>
    <td>
      %s
    </td>
    <td>
      %s
    </td>
    <td>
      %s
    </td>
    <td>
      %s
    </td>
  </tr>\n"""%(param, params_parsed[table_title][param]['default'], params_parsed[table_title][param]['desc'], type_str)
    s += "</table>\n\n<br>\n<br>\n\n"
    table_file_text = table_file_text.replace("*table_data*", s)

    io.open_write(table_filepaths[table_title], table_file_text)
