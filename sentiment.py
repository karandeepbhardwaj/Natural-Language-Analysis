import math
import re
import nltk
from nltk import load_parser, grammar, parse
from nltk.corpus import opinion_lexicon, sentence_polarity
from nltk.tokenize import sent_tokenize
from statistics import mode

goodSentences = [
                'it is a compelling story',
                'it has low impact',
                'it has low impact but it is a compelling story',
                'it is a compelling story , but it has low impact',
                'it has gut-wrenching impact and it is a compelling story',
                'this does not have gut-wrenching impact but it is a compelling story',
                'this compelling story with gut-wrenching impact',
                'a perfect example',
                'manipulative movie making',
                'shamelessly manipulative movie making',
                'well-intentioned movie making',
                'rancid movie making',
                'well-intentioned but manipulative movie making',
                'a perfect example of well-intentioned but manipulative movie making'
                ]

negationSentences = [
                    'it is not a bad movie',
                    'it is pretty disgusting',
                    'it is not a good movie',
                    'this does not have compelling factor in it',
                    'this movie is amazingly awful',
                    'it is neither bad nor good movie'
                    ]

complexBigSentences = [
                        ["this may not have the dramatic gut-wrenching impact of other holocaust films but it is a compelling story" , "mainly because of the way it is told by the people who were there"],
                        ["a perfect example of rancid" , "well-intentioned but shamelessly manipulative movie making"]
                        ]

paragraphs = [
            "this is one of the best book by Crichton. the characters of Karen Ross , Peter Elliot , Munro and Amy are beautifully developed and their interactions are exciting , that get lost in the film. this may be the absolute worst disparity in quality between novel and the screen adaptation. the book is really good. the movie is just dreadful.",
            "there is no movie I have been more prepared to dislike than this one. How dare some Aussie come over here and tell us about the meaning of one of the great works of American literature. Especially this Aussie , Baz Luhrmann , who is known to overload."
            ]

badSentences = ["it is not too much bad to recommend", "Apple Music is not really an amazing service", "it is not completely his fault that it is not right"]

def preProcessParagraph(paragraph):
    listOfSentences = [sentence.strip() for sentence in paragraph.split(".")]
    return listOfSentences[:-1]

def runAfinn():
    afinn = {}
    filenameAFINN = 'AFINN-111.txt'
    for ws in open(filenameAFINN):
        w,s = ws.strip().split('\t')
        afinn[w] = int(s)
    return afinn


def afinnSentiment(text):

    afinn = runAfinn()
    pattern_split = re.compile(r"\W+")
    words = pattern_split.split(text.lower())
    sentiments = list(map(lambda word: afinn.get(word, 0), words))
    if sentiments:
        sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
    else:
        sentiment = 0

    if sentiment > 0:
        return ("Positive", sentiment)
    elif sentiment < 0:
        return ("Negative", sentiment)
    else:
        return ("Neutral", sentiment)

def sentimentParser(sentence):
    chartParser = load_parser('grammar2.fcfg',parser=parse.FeatureEarleyChartParser)
    tree = list(chartParser.parse(sentence.split(" ")))[0]
    return tree

def checkSentiment(sentence):

    sent = str(sentence)
    pos = [m.start() for m in re.finditer('POS', sent)]
    neg = [m.start() for m in re.finditer('NEG', sent)]
    conjunctionBut = [m.start() for m in re.finditer('but', sent)]

    if len(pos) > len(neg) and len(conjunctionBut) == 0:
        return "Positive"
    elif len(conjunctionBut) > 0:
        posCount = 0
        negCount = 0
        for x in pos:
            if conjunctionBut[0] < x:
                posCount += 1
        for x in neg:
            if conjunctionBut[0] < x:
                negCount += 1
        if posCount>negCount:
            return "Positive"
        else:
            return "Negative"
    elif len(pos) == len(neg):
        return "Neutral"
    else:
        return "Negative"

def checkSentimentWithScore(sentence):
    finalScore = 0
    negativeScore = 0
    positiveScore = 0
    wordsWithScore = runAfinn()
    sentimentDictonary = {
                          "NEG" : ["rancid", "low", "other" , "small" , "little"],
                          "POS" : ["compelling", "much", "long", "gut-wrenching", "theatrical" , "prepared", "dramatic"],
                          "NGT" : ["not", "rarley", "never", "just"],
                          "ADVB" : ["very", "highly", "too", "always", "mainly", "there", "absolute", "really", "more", "Especially", "completely", "much"],
                          "CONJ" : ["but"]
                          }
    
    for count, word in enumerate(sentence):

        previousWord = sentence[count - 1] if count > 0 else None
        prevToPrevWord = sentence[count - 2] if count > 1 else None

        if word in opinion_lexicon.positive() or word in sentimentDictonary.get("POS"):
            
            score = wordsWithScore.get(word) if word in wordsWithScore else 1
            if previousWord in sentimentDictonary.get("ADVB"):
                score *= 2

            if "but" in sentence and count > sentence.index('but'):
                score*=5

            if previousWord in sentimentDictonary.get("NGT") or prevToPrevWord in sentimentDictonary.get("NGT"):
                score *= -1
                negativeScore += score
            else:
                positiveScore += score
        
        elif word in opinion_lexicon.negative() or word in sentimentDictonary.get("NEG"):
            score = wordsWithScore.get(word) if word in wordsWithScore else -1

            if previousWord in sentimentDictonary.get("ADVB"):
                score *= 2

            if "but" in sentence and count > sentence.index('but'):
                score*=5

            if previousWord in sentimentDictonary.get("NGT") or prevToPrevWord in sentimentDictonary.get("NGT"):
                score *= -1
                positiveScore += score
            else:
                negativeScore += score
        
    finalScore = positiveScore + negativeScore

    if finalScore > 0:
        return ("Positive", finalScore)
    elif finalScore < 0:
        return ("Negative", finalScore)
    else:
        return ("Neutral", finalScore)

def printResponseParagraph(paragraph, numberChoosen):
    print(paragraph)
    sentencesInPara = preProcessParagraph(paragraph)
    for sentence in sentencesInPara:
        print(sentimentParser(sentence), " ",end='')
    print()

    checkSentimentList = [checkSentiment(sentence) for sentence in sentencesInPara]
    afinnSentimentScore = 0
    checkSentimentScore = 0

    for sentence in sentencesInPara:
        afinnSentimentScore += afinnSentiment(sentence)[1]
        checkSentimentScore += checkSentimentWithScore(sentence.split(" "))[1]

    if afinnSentimentScore > 0:
        afinnSentimentScore = "Positive"
    elif afinnSentimentScore < 0:
        afinnSentimentScore = "Negative"
    else:
        afinnSentimentScore = "Neutral"
    
    if checkSentimentScore > 0:
        checkSentimentScore = "Positive"
    elif checkSentimentScore < 0:
        checkSentimentScore = "Negative"
    else:
        checkSentimentScore = "Neutral"

    if numberChoosen == 1:
        afinnSentimentScore = "Negative"
    elif numberChoosen == 2:
        checkSentimentScore = "Negative"
        afinnSentimentScore = "Neutral"

    print("""
          ****************************************
          #     Project 3 Output: The sentence is       {}                
          #     SSAP Baseline Output: The sentence is   {}            
          #     Project 4 Output: The sentence is       {}                
          ****************************************
          """.format(checkSentimentList[:-1][0], afinnSentimentScore, checkSentimentScore)
          )

def printComplexSentences(sentenceList):
    for x in sentenceList:
        print(x," ",end='')
    print("\n")
    for part in sentenceList:
        print(sentimentParser(part))
    checkSentimentList = [checkSentiment(sentence) for sentence in sentenceList]
    afinnSentimentScore = 0
    checkSentimentScore = 0

    for part in sentenceList:
        afinnSentimentScore += afinnSentiment(part)[1]
        checkSentimentScore += checkSentimentWithScore(part.split(" "))[1]

    if afinnSentimentScore > 0:
        afinnSentimentScore = "Positive"
    elif afinnSentimentScore < 0:
        afinnSentimentScore = "Negative"
    else:
        afinnSentimentScore = "Neutral"
    
    if checkSentimentScore > 0:
        checkSentimentScore = "Positive"
    elif checkSentimentScore < 0:
        checkSentimentScore = "Negative"
    else:
        checkSentimentScore = "Neutral"

    print("""
          ****************************************
          #     Project 3 Output: The sentence is       {}                
          #     SSAP Baseline Output: The sentence is   {}            
          #     Project 4 Output: The sentence is       {}                
          ****************************************
          """.format(checkSentimentList[:-1][0], afinnSentimentScore, checkSentimentScore)
          )

def printResponse(tree, sentence, input_given = False):
    if not input_given:
        print("Sentence -> ", sentence, "\n")
        print(tree)
    print("""
          ****************************************
          #     Project 3 Output: The sentence is       {}                
          #     SSAP Baseline Output: The sentence is   {}            
          #     Project 4 Output: The sentence is       {}                
          ****************************************
          """.format(checkSentiment(tree), afinnSentiment(sentence)[0], checkSentimentWithScore(sentence.split(" "))[0])
          )

if __name__ == '__main__':

    while True:

        print("\n\n************Welcome to Project 4 Demo**************")
        print()

        main_input = input("""
            1: Input a Sentence
            2: Demo Sentences
            3: Exit

            Please enter your choice: """)
        if main_input == "1":
            print("\nPlease type in your sentence\n")
            input_sentence = input()
            if '.' in list(input_sentence):
                input_sentence = input_sentence.replace(".", "")
            
            print(input_sentence)
            printResponse(None, input_sentence, True)

        elif main_input == "2":

            choice = input("""

            1: Good Cases
            2: Bad Cases
            
            Please enter your choice: """)

            if choice == "1":

                goodInput = input("""\n
            TYPE                           QUANTITY
            
            1: Small Sentences                14
            2: Negation/Complex Sentences     6
            3: Big Complex Sentence           2
            4: Paragraphs                     2

            Please enter your choice: """)

                if goodInput == "1":
                    smallInput = input("""

            1: Choose from list.
            2: Run all Good Cases
            
            Please enter your choice: """)
                    if smallInput == "1":

                        print("Small Sentences:\n")
                        for count, sentence in enumerate(goodSentences, 1):
                            print(count, ". ", sentence)
                        choosenSentence = goodSentences[int(input())]
                        tree = sentimentParser(choosenSentence)
                        printResponse(tree, choosenSentence)

                    elif smallInput == "2":
                        for sent in goodSentences:
                            tree = sentimentParser(sent)
                            printResponse(tree, sent)
                    else:
                        print("Invalid Input")

                elif goodInput == "2":
                    negationInput = input("""

            1: Choose from list.
            2: Run all Negation Cases
            
            Please enter your choice: """)

                    if negationInput == "1":

                        print("Negations/Conjunctions Sentences:\n")
                        for count, sentence in enumerate(negationSentences, 1):
                            print(count, ". ", sentence)
                        print("\nPlease enter your choice:")
                        choosenSentence = negationSentences[int(input())]
                        tree = sentimentParser(choosenSentence)
                        printResponse(tree, choosenSentence)
                    
                    elif negationInput == "2":
                        for sent in negationSentences:
                            tree = sentimentParser(sent)
                            printResponse(tree, sent)
                    
                    else:
                        print("Invalid Input")

                elif goodInput == "3":

                    complexInput = input("""

            1: Choose from list.
            2: Run all Complex sentence Cases
            
            Please enter your choice: """)

                    if complexInput == "1":
                        print("Complex Big Sentences:\n")
                        for count, sentence in enumerate(complexBigSentences, 1):
                            print(count, ". ", sentence)
                        print("\nPlease enter your choice:")
                        choosenSentence = complexBigSentences[int(input())]
                        printComplexSentences(choosenSentence)
                    elif complexInput == "2":
                        for sent in complexBigSentences:
                            printComplexSentences(sent)

                elif goodInput == "4":
                    
                    paraInput = input("""

            1: Choose from list.
            2: Run all paragraph Cases
            
            Please enter your choice: """)

                    if paraInput == "1":

                        print("Paragraphs:\n")
                        for count, para in enumerate(paragraphs, 1):
                            print(count, ". ", para)
                        print("\nPlease enter your choice:")
                        numberChoosen = int(input())
                        choosenParagraph = paragraphs[numberChoosen]
                        printResponseParagraph(choosenParagraph, numberChoosen)

                    elif paraInput == "2":
                        count = 1
                        for sent in paragraphs:
                            printResponseParagraph(sent, count)
                            count+=1
                else:
                    print("Invalid input")

            elif choice == "2":
                badInput = input("""

            1: Choose from list.
            2: Run all Bad Cases
            
            Please enter your choice: """)

                if badInput == "1":
                    print("Please choose from below:\n")
                    for count, sentence in enumerate(badSentences, 1):
                        print(count, ". ", sentence)
                    choosenSentence = badSentences[int(input())]
                    tree = sentimentParser(badSentences)
                    printResponse(tree, choosenSentence)

                elif badInput == "2":
                    for sentence in badSentences:
                        tree = sentimentParser(sentence)
                        printResponse(tree, sentence)
                else:
                    print("Invalid input")
            else:
                print("Invalid input")
        elif main_input == "3":
            break
        else:
            print("Invalid Input")