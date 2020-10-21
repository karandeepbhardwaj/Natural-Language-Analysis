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
import nltk.parse.earleychart as es
from nltk.parse.chart import demo_grammar
from time import perf_counter
from nltk.tree import Tree
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
    
    # print("\n\n----> Tokened Sentences <----\n\n")
    # for sentence in token_sentences:
    #     print("SENTENCE -> \n" + sentence)
    
########################################## Word Tokenizer ##############################################
    
    token_words = []
    
    #Loop through list of sentences and applying word tokenize on every sentence.
    for word in token_sentences:
        token_words += word_tokenize(word)
    
    # print("\n\n----> Tokened Words and Characters<----\n\n")
    # print(token_words)
    # print("\n\n----> End of sentence token module <----\n\n")
    
    pos_tagger(token_words)

########################################## POS Tagger ##############################################

'''pos_tagger() - Tags the words with word type, returns the list of tuples. where each tuple has word on 1st position
and tag on second position.'''

def pos_tagger(data):
    
    #Tagging words from words list by the types
    pos_tagged = pos_tag(data)
    
    # print("\n\n---->Tagged Words<----\n\n")
    # print(pos_tagged)
    # print("\n\n----> End of Pos Tagger module <----\n\n")
    
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
    
    
    pattern = 'DATE: {<NNP><CD><,><CD>}'
    chunkParser1 = RegexpParser(pattern)
    chunked1 = chunkParser1.parse(chunked)
    
    
    #Converting tree to list of tuples.
    result = []
    for s in chunked1:
        result.append(s)
    
    # print("\n\n---->Mesured Enitity detected Word list<----\n\n")
    # for word in result:
    #     print(word)
    # print("\n\n----> End of Measured Entity module <----\n\n")
    
    named_entity(result)
    # dateRecognizer(result)

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
           
def named_entity(tagged):        
    result = list(tagged)
    # print(result)
    
    for x in result:
        if type(x) == nltk.tree.Tree:
            date = ""
            for node in x:
                date = date + " " + node[0]
            result[result.index(x)] = (date, "DATE")
            
    mylist = nltk.ne_chunk(result)
    print(mylist)

#Taking name of file from console.
# filename = argv[1]

#getting raw data from reuters library as string in data.
# data = reuters.raw(filename)

sent1 = "John ate an apple."
sent2 = "John ate the apple at the table."
sent3 = "On Monday, John ate the apple in the fridge."
sent4 = "On Monday, John ate the apple in his office."
sent5 = "On Monday, John ate refrigerator apple in his office."
sent6 = "Last week, on Monday, John finally took the apple from the fridge to his office."
sent7 = "Last Monday, John promised that he will put an apple in the fridge."
sent7a = "He will eat it on Tuesday at his desk."
sent7b = "It will be crunchy."
sent8 = "On Monday, September 17, 2018, John O’Malley promised his colleague Mary that he would put a replacement apple in the office fridge."
sent8a = "O’Malley intended to share it with her on Tuesday at his desk and anticipated that the crunchy treat would delight them both."
sent8b = "But she was sick that day."
sent9 = "Sue said that on Monday, September 17, 2018, John O’Malley promised his colleague Mary that he would put a replacement apple in the office fridge and that O’Malley intended to share it with her on Tuesday at his desk."

testData = [sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent7a, sent7b, sent8, sent8a, sent8b, sent9]
finalData = [sent1, sent2, sent3, sent4, sent5, sent6, sent7, sent7a, sent7b, sent8, sent8a, sent8b, sent9]

grammar = nltk.CFG.fromstring("""
                                 
                                 S -> NP VP | NP VP POS | VP
                                 
                                 NP -> NNP | NNP NNP | NNP NNP POS NNP | DT | DT NN | DT NN NN | PRP NN | NN NN | JJ NN | JJ NNP | PRP | DATE | PRP NN NNP | NP POS NP POS | PP POS NP | NP POS PP POS NP ADVP | NP POS NP | CC NP | PP NP
                                 VP -> VBD S | VBD NP | VBD NP PP | VBD NP PP PP | VBD SBAR | MD VP | VB NP PP | VB NP PP PP | VB NP PP PP PP | VB ADJP | VBD NP SBAR | IN VP | VBD ADJP NP | VP CC VP | VB NP NP
                                 PP -> IN NP | IN NP POS NP
                                 ADVP -> RB
                                 SBAR -> IN S | SBAR CC SBAR
                                 ADJP -> JJ
                                 DATE -> MONTH DAY POS YEAR
                                 
                                 CC -> "and" | "But"
                                 MONTH -> "September"
                                 DAY -> "17"
                                 YEAR -> "2018"
                                 VB -> "put" | "eat" | "be" | "share" | "delight"
                                 MD -> "will" | "would"
                                 IN -> "at" | "On" | "in" | "from" | "on" | "to" | "that" | "with"
                                 NNP -> "John" | "Monday" | "Tuesday" | "O" | "Malley" | "Mary" | "Sue"
                                 VBD -> "ate" | "took" | "promised" | "said" | "intended" | "was" | "anticipated"
                                 PRP -> "his" | "he" | "He" | "it" | "It" | "her" | "she" | "them"
                                 DT -> "an" | "the" | "a" | "that" | "both"
                                 NN -> "apple" | "table" | "fridge" | "office" | "refrigerator" | "week" | "desk" | "colleague" | "replacement" | "day" | "crunchy" | "treat"
                                 JJ -> "Last" | "crunchy" | "sick"
                                 RB -> "finally"
                                 POS -> "." | "," | "’"
                             """)

test = ""

def demo(
    print_times=True,
    print_grammar=True,
    print_trees=True,
    trace=2,
    sent = test,
    numparses=0,
):
    """
    A demonstration of the Earley parsers.
    """
    tokens = word_tokenize(sent)
    earley = es.EarleyChartParser(grammar)
    chart = earley.chart_parse(tokens)
    parses = list(chart.parses(grammar.start()))
    if print_trees:
       print(parses[0])


# demo(sent=sent9)

for testString in testData:
    print("\n\n")
    print("--------> START <----------")
    print("For String:  " + testString)
    print("\n\n")
    demo(sent = testString)
    print("\n\n")
    print("--------> END <----------")
























# print(pos_tag(word_tokenize(test)))
####################################

# def get_continuous_chunks(text):
    
#     chunked = nltk.ne_chunk(pos_tag(word_tokenize(text)))
#     continuous_chunk = []
#     current_chunk = []
#     for i in chunked:
#         if type(i) == Tree:
#             current_chunk.append(" ".join([token for token, pos in i.leaves()]))
#         if current_chunk:
#             named_entity = " ".join(current_chunk)
#             if named_entity not in continuous_chunk:
#                 continuous_chunk.append(named_entity)
#                 current_chunk = []
#         else:
#             continue
#     return continuous_chunk


####################################



grammar3 = nltk.CFG.fromstring("""
                                 
                                 S -> NP VP | NP VP POS | PP NP VP | PP POS NP VP POS | NP POS PP POS NP ADVP VP POS | NP POS NP VP POS | VP | CC NP VP POS
                                 
                                 NP -> NNP | NNP NNP | NNP NNP POS NNP | DT | DT NN | DT NN NN | PRP NN | NN NN | JJ NN | JJ NNP | PRP | DATE | PRP NN NNP | NP POS NP POS
                                 VP -> VBD S | VBD NP | VBD NP PP | VBD NP PP PP | VBD SBAR | MD VP | VB NP PP | VB NP PP PP | VB NP PP PP PP | VB ADJP | VBD NP SBAR | IN VP | VBD ADJP NP | VP CC VP | VB NP NP
                                 PP -> IN NP | IN NP POS NP
                                 ADVP -> RB
                                 SBAR -> IN S | SBAR CC SBAR
                                 ADJP -> JJ
                                 DATE -> MONTH DAY POS YEAR
                                 
                                 CC -> "and" | "But"
                                 MONTH -> "September"
                                 DAY -> "17"
                                 YEAR -> "2018"
                                 VB -> "put" | "eat" | "be" | "share" | "delight"
                                 MD -> "will" | "would"
                                 IN -> "at" | "On" | "in" | "from" | "on" | "to" | "that" | "with"
                                 NNP -> "John" | "Monday" | "Tuesday" | "O" | "Malley" | "Mary" | "Sue"
                                 VBD -> "ate" | "took" | "promised" | "said" | "intended" | "was" | "anticipated"
                                 PRP -> "his" | "he" | "He" | "it" | "It" | "her" | "she" | "them"
                                 DT -> "an" | "the" | "a" | "that" | "both"
                                 NN -> "apple" | "table" | "fridge" | "office" | "refrigerator" | "week" | "desk" | "colleague" | "replacement" | "day" | "crunchy" | "treat"
                                 JJ -> "Last" | "crunchy" | "sick"
                                 RB -> "finally"
                                 POS -> "." | "," | "’"
                             """)



grammar2 = nltk.CFG.fromstring("""
                                 
                                 S -> NP VP POS
                                 
                                 NP -> NNP | DT NN | PRP NN | NN NN | JJ NN | PP
                                 VP -> VBD NP | VBD NP PP | VBD NP PP PP | POS NP VP | POS PP POS NP ADVP VP 
                                 PP -> IN NP
                                 ADVP -> RB
                                 
                                 IN -> "at" | "On" | "in" | "from" | "on" | "to"
                                 NNP -> "John" | "Monday"
                                 VBD -> "ate" | "took"
                                 PRP -> "his"
                                 DT -> "an" | "the"
                                 NN -> "apple" | "table" | "fridge" | "office" | "refrigerator" | "week"
                                 JJ -> "Last"
                                 RB -> "finally"
                                 POS -> "." | ","
                             """)

grammar1 = nltk.CFG.fromstring("""
                                 
                                 S -> NP VP POS | PP POS NP VP POS | NP POS PP POS NP ADVP VP POS
                                 
                                 NP -> NNP | DT NN | PRP NN | NN NN | JJ NN 
                                 VP -> VBD NP | VBD NP PP | VBD NP PP PP
                                 PP -> IN NP
                                 ADVP -> RB
                                 
                                 IN -> "at" | "On" | "in" | "from" | "on" | "to"
                                 NNP -> "John" | "Monday"
                                 VBD -> "ate" | "took"
                                 PRP -> "his"
                                 DT -> "an" | "the"
                                 NN -> "apple" | "table" | "fridge" | "office" | "refrigerator" | "week"
                                 JJ -> "Last"
                                 RB -> "finally"
                                 POS -> "." | ","
                             """)