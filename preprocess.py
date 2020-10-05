#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:00:20 2020

@author: karandeepbhardwaj
"""

######################################### Import necessary libraries ####################################

import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import reuters
from nltk import pos_tag
from nltk import RegexpParser
from nltk.parse import RecursiveDescentParser

from sys import argv

########################################## Preparing Data for test ################################################

'''tokenizer() - Splits the paragraphs into sentences and then splits the sentences in words. '''

def tokenizer(data):
    
########################################## Sentence Tokenizer ##########################################

    # Catching the strings like - E.R. to remove the dot with " "spaceLine
    content = data
    content = re.sub(r"(?<= [.(a-zA-z]{3})\.(?!=(\n))",' ', content)
    content = re.sub(r"(?<= [a-zA-z]{2})\.(?!=(\n))",' ', content)

    token_sentences = sent_tokenize(data) #Converting paragraphs to sentences.
    
    #Splitting the titles of the paragraph in seperate string 
    title_token = token_sentences[0].split("\n", 1)  
    
    token_sentences.pop(0) #removing the first element which might be combination of heading and next sentence
    token_sentences = title_token + token_sentences
    
    print("\n\n----> Tokened Sentences <----\n\n")
    for sentence in token_sentences:
        print("SENTENCE -> \n" + sentence)
    
########################################## Word Tokenizer ##############################################
    
    token_words = []
    
    #Loop through list of sentences and applying word tokenize on every sentence.
    for word in token_sentences:
        token_words += word_tokenize(word)
    
    print("\n\n----> Tokened Words and Characters<----\n\n")
    print(token_words)
    print("\n\n----> End of sentence token module <----\n\n")
    
    pos_tagger(token_words)

########################################## POS Tagger ##############################################

'''pos_tagger() - Tags the words with word type, returns the list of tuples. where each tuple has word on 1st position
and tag on second position.'''

def pos_tagger(data):
    
    #Tagging words from words list by the types
    pos_tagged = pos_tag(data)
    
    print("\n\n---->Tagged Words<----\n\n")
    print(pos_tagged)
    print("\n\n----> End of Pos Tagger module <----\n\n")
    
    measuredEntityDetection(pos_tagged)

########################################## Measured Entity Detection ##############################################

'''Number Normalization'''

'''mesauredEntityDetection() - Identifies the pattern of entities from the list of words and makes a tree at that place by combining the matched words'''

def measuredEntityDetection(data):
    
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
    pattern = 'ME: {<CD><NNS> | <CD><NNP> | <CD><NN> | <CD><NN><NNS>|<CD><CD><NNP><.><NNP>|<CD><CD><NNP>|<CD><CD><NNP>| <CD><JJ><NN><NN> }'
    chunkParser = RegexpParser(pattern)
    chunked = chunkParser.parse(myList)
    
    #Converting tree to list of tuples.
    result = []
    for s in chunked:
        result.append(s)
    
    print("\n\n---->Mesured Enitity detected Word list<----\n\n")
    for word in result:
        print(word)
    print("\n\n----> End of Measured Entity module <----\n\n")
    dateRecognizer(result)

'''dateRecognizerCFG() - Recognizes the date patterns from the words list and returns the dates one by one to date parser.'''

########################################## Date Recognizer ##############################################


def dateRecognizer(words):
    
    
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
    
    print("\n\n----> Date Recogniser module list<----\n\n")
    
    #Checking the numbers which could have been recognised as dates , such as 13.2 which is not date in most contexts.
    betterList = []
    for date in dateList:
        if "." not in date:
            betterList.append(date)
        else:
            if len(date.split(".")) == 3:
                betterList.append(date)  
    
    for date in betterList:
        print(date)
    print("\n\n----> End of Date Recogniser Module <----\n\n")
    
    print("\n\n---->Parsed Dates<----\n\n")
    for date in dateList:
        dateParseCFG(date)
    
    print("\n\n----> End of Date Parser module <----")

'''dateParseCFG() - Parses the string of date by keeping DATE as root and DAY, MONTH and YEAR as children.'''
 

def dateParseCFG(date):
    
    
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
                                       
                                       DATE -> DAY MONTH YEAR | MONTH DAY YEAR | YEAR MONTH DAY | MONTH DAY | DAY MONTH | MONTH YEAR | INP DAY INP MONTH YEAR | DAY INP MONTH | DAY INP MONTH YEAR | NUMDAY CHR MONTH CHR YEAR | YEAR                                       
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
            print(node)
            node.draw() # Drawing an image of tree paresed one by one.
    except:
           print()
    
test = ["training/12866","training/1412",
            "training/14313","training/14771","training/1853",
            "training/1890","training/198","training/220","training/2531"]
all_data = ["training/9920"]

#Taking name of file from console.
filename = argv[1]

#getting raw data from reuters library as string in data.
data = reuters.raw(filename)
print(data)
print("----------------------------------------START----------------------------------------")
tokenizer(data)
print("----------------------------------------END----------------------------------------")
