#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:00:20 2020

@author: karandeepbhardwaj
"""

######################################### Import necessary libraries ####################################

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import reuters
from nltk import pos_tag
import nltk.parse.earleychart as es
from nltk.tree import Tree
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import reuters
from nltk import RegexpParser
from nltk.parse import RecursiveDescentParser
import pprint
import sys


measurements = ['g_0', 'g_0', 'g_0', 'g_0', 'g_0', 'g_0', 'g_0', 'g_0', 'g_0', 'g_0', 'dimensionless', 'rad', 'rad', 'rad', 'mrad', 
         'mrad', 'urad', 'urad', 'turn', 'turn', 'turn', 'turn', 'turn', 'turn', 'deg', 'deg', 'deg', 'deg', 'deg', 'deg', 'arcmin',
         'arcmin', 'arcmin', 'arcmin', 'arcsec', 'arcsec', 'arcsec', 'arcsec', 'grad', 'grad', 'degN', 'degN', 'degE', 'degE', 'degW',
         'degW', 'degT', 'degT', 'sr', 'sr', 'are', 'are', 'b', 'b', 'cmil', 'cmil', 'D', 'D', 'mD', 'mD', 'ha', 'ha', 'acre', 'acre', 
         'US_survey_acre', '(pc/cm**3)', 'mol', 'L', 'M', 'M', 'mM', 'mM', 'uM', 'uM', '%', 'ct', 'ct', 'cm', 'erg', 'N', 'A', 'A', 'A',
         'A', 'A', 'mA', 'mA', 'mA', 'uA', 'uA', 'nA', 'nA', 'nA', 'pA', 'pA', 'pA', 'aA', 'aA', 'aA', 'esu', 'esu', 'esu', 'esu', 'esu', 
         '(esu/s)', '(esu/s)', 'ampere_turn', 'Gi', 'Gi', 'C', 'C', 'mC', 'mC', 'uC', 'uC', 'V', 'V', 'kV', 'kV', 'mV', 'mV', 'uV', 'uV',
         'F', 'F', 'mF', 'uF', 'nF', 'pF', 'ohm', 'ohm', 'kiloohm', 'megaohm', 'S', 'S', 'mS', 'mS', 'uS', 'uS', 'nS', 'nS', 'pS', 'pS', 
         'Wb', 'Wb', 'T', 'T', 'H', 'H', 'abfarad', 'abhenry',
         'abmho', 'abohm', 'abvolt', 'e', 'e', 'chemical_faraday', 'physical_faraday', 'faraday', 'faraday', 'gamma', 'G', 'Mx', 'Oe', 'Oe',
         'stF', 'stF', 'stF', 'stH', 'stH', 'stH', 'stS', 'stS', 'stS', 'statohm',
         'stV', 'stV', 'stV', 'unit_pole', 'mu_0', 'mu_0', 'mu_0', 'epsilon_0', 'epsilon_0', 'epsilon_0', 'Z_0', 'Z_0', 'Z_0', 
         'cd', 'cd', 'cd', 'J', 'J', 'BTU', 'BTU', 'BTU', 'BTU', 'eV', 'eV', 'meV', 'keV', 'MeV', 'GeV', 'GeV', 'thm', 'thm', 'thm', 'cal', 
         'cal', 'cal', 'cal_IT', 'tTNT', 'US_therm', 'Wh', 'Wh', 'Wh', 'kWh', 'kWh', 
         'kWh', 'MWh', 'MWh', 'MWh', 'GWh', 'GWh', 'GWh', 'E_h', 'E_h', 'E_h', 'oz', 'lb', 'ft', 'N', 'dyn', 'p', 'kgf', 'kgf', 'kgf', 'ozf', 'ozf', 
         'ozf', 'lbf', 'lbf', 'lbf', 'pdl', 'gf', 'gf', 'gf', 'ton_force', 'ton_force', 'kip', 'Hz', 'Hz', 'Hz', 'kHz', 'kHz', 'MHz', 'MHz', 'GHz', 'GHz',
         'rpm', 'rpm', 'counts_per_second', 'RSI', 'clo', 'clo', 'R_value', 'bit', 'B', 'B', 'B', 'B', 'kB', 'kB', 'kB', 'MB', 'MB', 'MB', 'GB', 'GB', 'GB',
         'TB', 'TB', 'TB', 'PB', 'PB', 'PB', 'EB', 'EB', 'EB', 'ZB', 'ZB', 'ZB', 'YB', 'YB', 'YB', 'Bd', 'Bd', 'Bd', 'KiB', 'KiB', 'KiB', 'MiB', 'MiB', 'MiB',
         'GiB', 'GiB', 'GiB', 'TiB', 'TiB', 'TiB', 'PiB', 'PiB', 'PiB', 'EiB', 'EiB', 'EiB', 'ZiB', 'ZiB', 'ZiB', 'YiB', 'YiB', 'YiB', 'm', 'm', 'm', 'km', 'km',
         'km', 'cm', 'cm', 'mm', 'mm', 'mm', 'um', 'um', 'um', 'um', 'nm', 'nm', 'nm', 'pm', 'pm', 'pm', 'angstrom', 'fm', 'fm', 'fm', 'fm', 'in', 'in', 'ft', 'ft',
         'mi', 'mi', 'mi', 'yd', 'yd', 'yd', 'mil', 'mil', 'pc', 'pc', 'ly', 'ly', 'au', 'au', 'nmi', 'nmi', 'pt', 'point', 'point', 'pica', 'US_survey_foot', 'US_survey_yard',
         'US_survey_mile', 'US_survey_mile', 'rod', 'rod', 'rod', 'furlong', 'fathom', 'chain', 'barleycorn', 'arpentlin', 'kayser', 'kayser', 'kg', 'kg', 'g', 'g', 'mg', 'mg', 
         'oz', 'oz', 'lb', 'lb', 'st', 'st', 'carat', 'gr', 'gr', 'long_hundredweight', 'short_hundredweight', 't', 't', 't', 'dwt', 'dwt', 'slug', 'slug', 'toz', 'toz', 'toz', 
         'toz', 'tlb', 'tlb', 'tlb', 'u', 'u', 'u', 'u', 'u', 'scruple', 'dr', 'dr', 'drachm', 'drachm', 'bag', 'short_ton', 'short_ton', 'long_ton', 'denier', 'tex', 'dtex', 'min',
         'W', 'W', 'W', 'mW', 'mW', 'kW', 'kW', 'MW', 'MW', 'hp', 'hp', 'hp', 'hp', 'boiler_horsepower', 'metric_horsepower', 'electric_horsepower', 'water_horsepower', 'refrigeration_ton', 
         'refrigeration_ton', 'conventional_mercury', 'conventional_mercury', 'conventional_mercury', 'mercury_60F', 'H2O', 'H2O', 'H2O', 'H2O', 'water_4C', 'water_4C', 'water_60F', 'Pa', 'Pa',
         'hPa', 'hPa', 'kPa', 'kPa', 'MPa', 'MPa', 'GPa', 'GPa', 'bar', 'mb', 'mb', 'mb', 'kbar', 'kbar', 'Mbar', 'Mbar', 'Gbar', 'Gbar', 'atm', 'atm', 'atm', 'at', 'at', 'torr', 'psi', 'psi', 'ksi',
         'ksi', 'Ba', 'Ba', 'Ba', 'Ba', 'Ba', 'mmHg', 'mmHg', 'mmHg', 'mmHg', 'cmHg', 'cmHg', 'cmHg', 'inHg', 'inHg', 'inHg', 'inHg', 'inch_Hg_60F', 'inch_H2O_39F', 'inch_H2O_60F', 'footH2O', 'cmH2O', 
         'foot_H2O', 'foot_H2O', 'Bq', 'Bq', 'Ci', 'Ci', 'Rd', 'Rd', 'Gy', 'Gy', 'Gy', 'Gy', 'rem', 'rads', 'R', 'R', 'mol', 'mmol', 'umol', 'K', 'K', 'K', 'K', 'degR', 'degR', 'degR', 'degC', 'degC',
         'degC', 'degF', 'degF', 'degF', 's', 's', 's', 'ks', 'ks', 'Ms', 'Ms', 'ms', 'ms', 'us', 'us', 'ns', 'ns', 'ps', 'ps', 'fs', 'fs', 'as', 'min', 'h', 'h', 'h', 'd', 'd', 'week', 'fortnight', 'yr',
         'yr', 'yr', 'yr', 'month', 'shake', 'sidereal_day', 'sidereal_hour', 'sidereal_minute', 'sidereal_second', 'sidereal_year', 'sidereal_month', 'tropical_month', 'synodic_month', 'synodic_month', 'common_year',
         'leap_year', 'Julian_year', 'Gregorian_year', 'millenium', 'eon', 'work_year', 'work_month', 'c', 'c', 'kt', 'kt', 'kt', 'kt', 'P', 'P', 'cP', 'cP', 'St', 'St', 'rhe', 'L', 'L', 'L', 'mL', 'cc', 'mL', 'kL', 'kL',
         'kL', 'ML', 'ML', 'ML', 'GL', 'GL', 'GL', 'cc', 'cc', 'stere', 'GRT', 'GRT', 'acre_foot', 'FBM', 'bu', 'bu', 'bu', 'US_dry_gallon', 'US_liquid_gallon', 'US_liquid_gallon', 'US_liquid_gallon', 'US_dry_quart', 'US_dry_quart',
         'US_dry_pint', 'US_dry_pint', 'quart', 'quart', 'quart', 'pt', 'pt', 'pt', 'cup', 'cup', 'gill', 'gill', 'fl_oz', 'fl_oz', 'fl_oz', 'fl_oz', 'Imperial_bushel', 'UK_liquid_gallon', 'UK_liquid_gallon', 'UK_liquid_quart', 'UK_liquid_pint',
         'UK_liquid_cup', 'UK_liquid_gill', 'UK_fluid_ounce', 'UK_fluid_ounce', 'bbl', 'bbl', 'tbsp', 'tbsp', 'tbsp', 'tbsp', 'tbsp', 'tbsp', 'tbsp', 'tsp', 'tsp', 'pk', 'pk', 'fldr', 'fldr', 'fldr', 'firkin', 'kilogram'
         ,'gram', 'dlrs','million', 'mln', 'bln', 'billion',"metre","metres","m","second","seconds","sec","s","kilogram","kilograms","kg","ampere","amperes","A","kelvin","kelvins","K","candela","candelas","cd","mole","moles","mol","kilometre","kilometres","km","centimetre","centimetres","cm","millimetre","millimetres","mm","micrometre","micrometres","nanometre","nanometres","nm","hectare","hectares","ha","kilolitre","kilolitres","kl","litre","litres","l","centilitre","centilitres","cl","millilitre","millilitres","ml","microlitre","microlitres","tonne","tonnes","ton","gram","grams","g","milligram","milligrams","mg","hertz","Hz","newton","newtons","dozen","N","joule","joules","J","pascal","pascal","Pa","watt","watts","W","coulomb","coulombs","C","volt","volts","V","farad","farads","F","ohm","ohms","metre","metres","hundred","thousand","million","millions","billion","billions","trillion","trillions","mln","mlns","bln","blns","tln","tlns","m","gram","grams","g","gm","inch","inches","foot","feet","yard","yards","mile","miles","acre","acres","gallon","gallons","ounce","ounces","pound","pounds","horsepower","horsepowers","kilowatt","kilowatts"]


########################################## Preparing Data for test ################################################

'''tokenizer() - Splits the paragraphs into sentences and then splits the sentences in words. '''

def tokenizer(data, save_to_file):
    
########################################## Sentence Tokenizer ##########################################

    # Catching the strings like - E.R. to remove the dot with " "spaceLine
    # content = data
    # content = re.sub(r"(?<= [.(a-zA-z]{3})\.(?!=(\n))",' ', content)
    # content = re.sub(r"(?<= [a-zA-z]{2})\.(?!=(\n))",' ', content)

    token_sentences = sent_tokenize(data) #Converting paragraphs to sentences.
    
    #Splitting the titles of the paragraph in seperate string 
    title_token = token_sentences[0].split("\n", 1)  
    
    token_sentences.pop(0) #removing the first element which might be combination of heading and next sentence
    token_sentences = title_token + token_sentences
    
    print("\n\n----> Tokened Sentences\n\n", file = save_to_file)
    for sentence in token_sentences:
        print("SENTENCE -> \n" + sentence, file = save_to_file)
    
########################################## Word Tokenizer ##############################################
    
    token_words = []
    
    #Loop through list of sentences and applying word tokenize on every sentence.
    for word in token_sentences:
        token_words += word_tokenize(word)
    
    print("\n\n----> Tokened Words and Characters\n\n", file = save_to_file)
    print(token_words, file = save_to_file)
    print("\n\n----> End of sentence token module\n\n", file = save_to_file)
    
    pos_tagger(token_words, save_to_file)

########################################## POS Tagger ##############################################

'''pos_tagger() - Tags the words with word type, returns the list of tuples. where each tuple has word on 1st position
and tag on second position.'''

def pos_tagger(data, save_to_file):
    
    #Tagging words from words list by the types
    pos_tagged = pos_tag(data)
    
    print("\n\n----> Tagged Words\n\n", file = save_to_file)
    print(pos_tagged, file = save_to_file)
    print("\n\n----> End of Pos Tagger module\n\n", file = save_to_file)
    
    measuredEntityDetection(pos_tagged, save_to_file)
    named_entity(pos_tagged, save_to_file)

########################################## Measured Entity Detection ##############################################

'''Number Normalization'''

'''mesauredEntityDetection() - Identifies the pattern of entities from the list of words and makes a tree at that place by combining the matched words'''

def measuredEntityDetection(data, save_to_file):
    
    # Pattern for error in the paragraphs - lt&:
    less_than_pattern = 'NN: {<CC><NN><:>}'

    #Identifying pattern fromwords and parsing the tag into them by replacing the words by tagged word.    
    cp = RegexpParser(less_than_pattern)
    cs = cp.parse(data)
    
    #Converting tree to list of tuples.
    myList = []
    for s in cs:
        myList.append(s)
    
    #identifying where error is and replacing with "<" opening tag.    
    for x in myList:
        if type(x) == nltk.tree.Tree:
            myList[myList.index(x)] = ('<','NN')
    
    # Pattern for entity in the paragraphs - 1.3 MLN, 30,000 tonnes.
    pattern = 'ME: {<CD><NNS> | <CD><NN> | <CD><CD><NNS> | <CD><JJ> | <CD><CD>}'
    chunkParser = RegexpParser(pattern)
    chunked = chunkParser.parse(myList)
    
    #Converting tree to list of tuples.
    result = []
    for s in chunked:
        result.append(s)

    recList=list(chunked)
    resultList=[]
    
    for i in recList:
        if type(i)==nltk.tree.Tree and i.label()=="ME":
            for j in i.leaves():
                if j[1] in ["NN","NNS", "JJ", "CD"] and j[0] in measurements:
                    resultList.append(i.leaves())
    
    print("\n\n----> Mesured Enitity detected Word list\n\n", file = save_to_file)
    
    # for x in result
    # for s in result:
    #     if type(s)==nltk.tree.Tree:
    #         if s.label() == "ME":
    #             print(s)
                
    for word in resultList:
        print(word[0][0], word[1][0], file = save_to_file)
    print("\n\n----> End of Measured Entity module\n\n", file = save_to_file)
    
    dateRecognizer(result, save_to_file)    

'''dateRecognizerCFG() - Recognizes the date patterns from the words list and returns the dates one by one to date parser.'''

########################################## Date Recognizer ##############################################


def dateRecognizer(words, save_to_file):
    
    
    MONTHS = ["january","february","march","april","may",
           "june","july","august","september","october",
           "november","december","jan","feb","mar","apr",
           "jun","jul","aug","sept","sep","oct","nov","dec"]
    
    NUMBERS = ['first', 'eleventh', 'second', 'twelfth', 'third',
               'thirteenth', 'fourth', 'fourteenth', 'fifth', 'fifteenth', 'sixth',
               'sixteenth', 'seventh', 'seventeenth', 'eighth', 'eighteenth',
               'ninth', 'ninteenth','ten', 'tenth', 'twentieth', 'twenty', 'thirtieth', 'thirty']
    
    dateList = []
    
    #Check for word, checking pattern of date in every word in word list.
    for word in words:
        index = words.index(word)

        if ("NNP" or "NN") in word and "IN" in words[index+1] and "NNP" in words[index+2] and words[index+2][0].lower() in MONTHS and "CD" in words[index+3]: # Second of October 2020
            
            DAY = word[0]
            CHAR = words[index+1][0]
            MONTH = words[index+2][0]
            YEAR = words[index+3][0]
            DATE = DAY + " " + CHAR + " " + MONTH + " " + YEAR
            dateList.append(DATE)
        
        elif ("NNP" or "NN") in word and word[0] in NUMBERS and "IN" in words[index+1] and "NNP" in words[index+2] and words[index+2][0].lower() in MONTHS and "CD" not in words[index+3]: # Second of October
            
            DAY = word[0]
            CHAR = words[index+1][0]
            MONTH = words[index+2][0]
            DATE = DAY + " " + CHAR + " " + MONTH
            dateList.append(DATE) 
       
        elif "CD" in word and "IN" in words[index-1] and "IN" in words[index+1] and "NNP" in words[index+2] and words[index+2][0].lower() in MONTHS and "IN" in words[index+3] and "NN" in words[index+4] and "CD" in words[index+5]:
                
            DAY = word[0]
            MONTH = words[index+2][0]
            CHAR = words[index+1][0]
            YEAR = words[index+5][0]
            DATE = DAY + " " + CHAR + " " + MONTH + " " + YEAR
            dateList.append(DATE)
            
        elif "CD" in word and "IN" in words[index-1] and "IN" in words[index+1] and "NNP" in words[index+2] and words[index+2][0].lower() in MONTHS and "IN" not in words[index+3]:
                
            DAY = word[0]
            CHAR = words[index+1][0]
            MONTH = words[index+2][0]    
            DATE = DAY + " " + CHAR + " " + MONTH
            dateList.append(DATE) 
        
        elif "CD" in word and "IN" in words[index-2] and "NNP" in words[index-1] and words[index-1][0].lower() in MONTHS and "CD" in words[index+1]: 
                
            MONTH = words[index-1][0]
            DAY = word[0]
            YEAR = words[index+1][0]          
            DATE = MONTH + " " + DAY + " " +  YEAR
            dateList.append(DATE)
        
        elif "CD" in word and "NNP" in words[index-1] and words[index-1][0].lower() in MONTHS:
                
            DAY = word[0]
            MONTH = words[index-1][0]
            DATE =  MONTH +" "+ DAY
            dateList.append(DATE)
        
        
        elif "CD" in word and "IN" in words[index-1] and "NNP" in words[index+1]and words[index+1][0].lower() in MONTHS and "CD" in words[index+2]: 
                
            DAY = word[0]
            MONTH = words[index+1][0]
            YEAR = words[index+2][0]
            DATE = DAY + " " + MONTH + " " + YEAR
            dateList.append(DATE)
            
        elif "CD" in word and "IN" in words[index-1] and "NNP" in words[index+1] and words[index+1][0].lower() in MONTHS and "CD" not in words[index+2]: 
                
            DAY = word[0]
            MONTH = words[index+1][0]
            DATE = DAY + " " + MONTH
            dateList.append(DATE)
            
        elif "CD" in word and "IN" in words[index-1]  and "IN" not in words[index+1] and "NNP" not in words[index+1]:
            dateList.append(word[0])
    
    print("\n\n----> Date Recogniser module list\n\n", file = save_to_file)
    
    #Checking the numbers which could have been recognised as dates , such as 13.2 which is not date in most contexts.
    betterList = []
    for date in dateList:
        if "." not in date and len(date.strip()) > 2:
            betterList.append(date)
        else:
            if len(date.split(".")) == 3:
                betterList.append(date)  
    
    for date in betterList:
        print(date, file = save_to_file)
    
    print("\n\n----> Parsed Dates\n\n", file = save_to_file)
    
    
    
    pattern = 'DATE: {<NNP><CD><,><CD>}'
    chunkParser = RegexpParser(pattern)
    chunked = chunkParser.parse(words)
    dateList = chunked
        
    finalDates = []
    flag = False
    
    for x in dateList:
        if type(x) == nltk.tree.Tree and str(x)[1:5] == "DATE":
            flag = True
            date = ""
            for node in x:
                date = date + " " + node[0]
            dateList[dateList.index(x)] = (date, "DATE")
            finalDates.append(date.strip())
    if not flag:
       finalDates = betterList
       
    for date in finalDates:
        print(date, file = save_to_file)
        dateParseCFG(date, save_to_file)
    print("\n\n----> End of Date Recogniser Module\n\n", file = save_to_file)

'''dateParseCFG() - Parses the string of date by keeping DATE as root and DAY, MONTH and YEAR as children.'''
 

def dateParseCFG(date, save_to_file):
    
    try:
        dateString=[]
        #Splitting the strings of dates which are not the format xx/xx/xxx.
        if " " in date:
            dateString = date.split(" ")
            tempList=[]
            #Splitting and appending the characters as split in the list to recognise.
            for char in dateString:
                if not char.isdigit():
                    tempList.append(char)
                else:
                    tempList += list(char)
            dateString = tempList
        else:
            dateString = date

        # Defining the grammar of the CFG.
        date_grammar = nltk.CFG.fromstring("""
                                       
                                       DATE -> DAY MONTH YEAR | MONTH DAY YEAR | YEAR MONTH DAY | MONTH DAY | DAY MONTH | MONTH YEAR | INP DAY INP MONTH YEAR | DAY INP MONTH | DAY INP MONTH YEAR | NUMDAY CHR MONTH CHR YEAR | YEAR  | MONTH DAY CHR YEAR                                      
                                       NUM -> '1'|'2'|'3'|'4'|'5'|'6'|'7'|'8'|'9'|'0'|'1st'|'2nd'|'3rd'|'4th'|'5th'|'6th'|'7th'|'8th'|'9th'|'10th'|'11th'|'12th'|'13th'|'14th'|'15th'|'16th'|'17th'|'18th'|'19th'|'20th'|'21st'|'22nd'|'23rd'|'24th'|'25th'|'26th'|'27th'|'28th'|'29th'|'30th'|'31st'                                       
                                       DAY -> NUMDAY | TXTDAY | TXTDAY TXTDAY
                                       NUMDAY -> NUM | NUM NUM
                                       TXTDAY -> 'first' | 'First' | 'eleventh' | 'Eleventh' | 'second' | 'Second' | 'twelfth' | 'Twelfth' | 'third' | 'Third' | 'thirteenth' | 'Thirteenth' | 'fourth'| 'Fourth'| 'fourteenth'| 'Fourteenth' | 'fifth'| 'Fifth' | 'fifteenth'| 'Fifteenth' | 'sixth'| 'Sixth' | 'sixteenth'| 'Sixteenth' | 'seventh'| 'Seventh' | 'seventeenth'| 'Seventeenth' | 'eighth'| 'Eighth' | 'eighteenth'| 'Eighteenth' | 'ninth'| 'Ninth' | 'ninteenth'| 'Ninteenth' | 'tenth'| 'Tenth' | 'twentieth'| 'Twentieth' | 'twenty'| 'Twenty' | 'thirtieth'| 'Thirtieth' | 'thirty'| 'Thirty'                                       
                                       MONTH -> NUM | NUM NUM | "Jan"|"January"|"Feb"|"February"|"Mar"|"March"|"Apr"|"April"|"May"|"Jun"|"June"|"Jul"|"July"|"Aug"|"August"|"Sep"|"Sept"|"September"|"Oct"|"October"|"Nov"|"November"|"Dec"|"December"
                                       YEAR -> NUM NUM NUM NUM                                       
                                       CHR -> "," | "/" | "." | "-" | " " 
                                       INP -> 'on'|'in'|'the'|'of'
                                       
                                       """)
                                       
        rd = RecursiveDescentParser(date_grammar) # Identifying dates from the strings of date.
        for node in rd.parse(dateString):
            print("", file = save_to_file)
            print(node, file = save_to_file)# Drawing an image of tree paresed one by one.
    except:
            print("I am Error")

#################################################################################################

'''namedEntity() - Recognizes the names of persons and organisations from the words list and returns the dates one by one to date parser.'''

########################################## Named Entity ##############################################    

def named_entity(tagged, save_to_file):
    
    chunked = nltk.ne_chunk(tagged)
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        if current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue
    print("\n\n----> Named entity Word list\n\n", file = save_to_file)
    print(continuous_chunk, file = save_to_file)
    print("\n\n----> End of Named entity module\n\n", file = save_to_file)   
    
########################################## End of Named Entity ############################################## 

sentence = ""

def sentence_parser(sentences, save_to_file):
    
    try:  
        grammar = nltk.CFG.fromstring("""
                                 S -> NP VP | NP VP POS | VP
                                 
                                 NP -> NNP | NNS | NP PP | NNP NNP | NNP NNP POS NNP | DT | DT NN | DT NN NN | PRP NN | NN NN | JJ NN | JJ NNP | PRP | DATE | PRP NN NNP | NP POS NP POS | PP POS NP | NP POS PP POS NP ADVP | NP POS NP | CC NP | PP NP | ADVP POS NP | NNP CC NNP
                                 VP -> VBD S | VBD NP | VBD PP | VBD NP PP | VBD NP PP PP | VBD NP PP NP | VBD SBAR | MD VP | VB NP PP | VB NP PP PP | VB NP PP PP PP | VB ADJP | VBD NP SBAR | IN VP | VBD ADJP NP | VP CC VP | VB NP NP | VBP NP NP | VB PP
                                 PP -> IN NP | IN NP POS NP
                                 ADVP -> RB
                                 SBAR -> IN S | SBAR CC SBAR
                                 ADJP -> JJ
                                 DATE -> MONTH DAY POS YEAR | MONTH YEAR
                                 
                                 CC -> "and" | "But"
                                 MONTH -> "September"
                                 DAY -> "17"
                                 YEAR -> "2018" | "2021"
                                 VB -> "put" | "eat" | "be" | "share" | "delight" | "be"
                                 MD -> "will" | "would"
                                 IN -> "at" | "On" | "in" | "from" | "on" | "to" | "that" | "with" | "for"
                                 NNP -> "John" | "Monday" | "Tuesday" | "O" | "Malley" | "Mary" | "Sue"
                                 NNS -> "apples"
                                 VBD -> "ate" | "took" | "promised" | "said" | "intended" | "was" | "anticipated" | "put"
                                 VBP -> "promise"
                                 PRP -> "his" | "he" | "He" | "it" | "It" | "her" | "she" | "them" | "I" | "you"
                                 DT -> "an" | "the" | "a" | "that" | "both"
                                 NN -> "apple" | "table" | "fridge" | "office" | "refrigerator" | "week" | "desk" | "colleague" | "replacement" | "day" | "crunchy" | "treat"
                                 JJ -> "Last" | "last" | "crunchy" | "sick"
                                 RB -> "finally" | "Finally"
                                 POS -> "." | "," | "’"
                                     """)
        
        for sentence in sent_tokenize(sentences):
            tokens = word_tokenize(sentence)
            earley = es.EarleyChartParser(grammar)
            chart = earley.chart_parse(tokens)
            parses = list(chart.parses(grammar.start()))
            print(parses[0], file = save_to_file)
    except:
        print("Invalid sentences has been passed.", file = save_to_file)

def get_data():
    sent1 = "John ate an apple."
    sent2 = "John ate the apple at the table."
    sent3 = "On Monday, John ate the apple in the fridge."
    sent4 = "On Monday, John ate the apple in his office."
    sent5 = "On Monday, John ate refrigerator apple in his office."
    sent6 = "Last week, on Monday, John finally took the apple from the fridge to his office."
    sent7 = "Last Monday, John promised that he will put an apple in the fridge. He will eat it on Tuesday at his desk. It will be crunchy."
    sent8 = "On Monday, September 17, 2018, John O’Malley promised his colleague Mary that he would put a replacement apple in the office fridge. O’Malley intended to share it with her on Tuesday at his desk and anticipated that the crunchy treat would delight them both. But she was sick that day."
    sent9 = "Sue said that on Monday, September 17, 2018, John O’Malley promised his colleague Mary that he would put a replacement apple in the office fridge and that O’Malley intended to share it with her on Tuesday at his desk."
    
    data = [sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent8, sent9] 
    return data

def get_challenge_data():
    sent1 = "John ate at his desk."
    sent2 = "Finally, O’Malley ate the crunchy apple on the table."
    sent3 = "Sue intended to share the fridge with John and Mary."
    sent4 = "Mary put apples in the fridge last Monday."
    sent5 = "I promise you an apple for September 2021."
    sent6 = "It will be in the office fridge."
    
    data = [sent1, sent2, sent3, sent4, sent5, sent6] 
    return data

def test_run(data, save_to_file):   
             
    for sentence in data:
        print("\n\n", file = save_to_file)
        print("\t\t\t\t\t--------> START <----------", file = save_to_file)
        print("\n\n", file = save_to_file)
        print("FOR SENTENCE---->          " + sentence, file = save_to_file)
        print("\n\n", file = save_to_file)
        sentence_parser(sentence, save_to_file)
        print("\n\n", file = save_to_file)
        print("\t\t\t\t\t--------> END <----------", file = save_to_file)

def main():
        save_file_option = input("Do you want to save to file ? (y or n) \n\n")
        if save_file_option == "y":
            with open("pipeline.txt", "w") as myFile:
                after_save(myFile)
        elif save_file_option == "n":
            after_save(sys.stdout)
        else:
            print("Please give a valid input.")
        
def after_save(save_to_file):
    one_option = input("\nPress 1 to choose from NLTK Corpus. (Project 1) \nPress 2 to choose Validation Data. (Project 2) \nPress 3 to choose Challenge Run Data. (Project 2)\nPress 4 to input a sentence. (Project 2)\n\n")
            
    if one_option == "1":
        print("\nPlease choose from below provided corpus:\n")
        corpus_list = ["training/12866","training/1412", "training/14313","training/14771","training/1853", "training/1890","training/198","training/220","training/2531","training/9920","training/100"]
        count = 1
        for corpus in corpus_list:
            print(str(count)+ ")", " ", corpus)
            print()
            count+=1
        data = reuters.raw(corpus_list[int(input())-1])
        tokenizer(data, save_to_file)
            
    elif one_option == "2":
        
        count1 = 1
        print("\nPress 1 to run on all validation data.")
        print("Press 2 to choose sentences from validation data.\n")
        second_option = input()
        
        if second_option == "1":
            for sentence in get_data():
                tokenizer(sentence, save_to_file)
            test_run(get_data(), save_to_file)
        elif second_option == "2":
            for sentence in get_data():
                print("\n",str(count1)+ ")", " ", sentence)
                count1+=1
            third_option = input()
            tokenizer(get_data()[int(third_option)-1], save_to_file)
            sentence_parser(get_data()[int(third_option)-1], save_to_file)
        else:
            print("Invalid input please run preprocess again.") 
            
    elif one_option == "3":
        count2 = 1
        print("\nPress 1 to run on all Challenge run data.")
        print("Press 2 to choose sentences from Challenge run data.\n")
        second_option = input()
        
        if second_option == "1":
            for sentence in get_challenge_data():
                tokenizer(sentence, save_to_file)
            test_run(get_challenge_data(), save_to_file)
        elif second_option == "2":
            for sentence in get_challenge_data():
                print("\n",str(count2)+ ")", " ", sentence)
                count2+=1
            fourth_option = input()
            tokenizer(get_challenge_data()[int(fourth_option)-1], save_to_file)
            sentence_parser(get_challenge_data()[int(fourth_option)-1], save_to_file)         
            
    elif one_option == "4":
        sentences = input("Please write a full sentence.\n")
        tokenizer(sentences, save_to_file)
        sentence_parser(sentences, save_to_file)
    else:
        print("Invalid input please run preprocess again.")

main()