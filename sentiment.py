from nltk import load_parser
import nltk
import re
from nltk import grammar, parse

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
                'a perfect example of well-intentioned but manipulative movie making',
                'it is neither bad nor good movie'
                ]

badSentences = [
                'it is not a bad movie',
                'it is pretty disgusting',
                'it is not a good movie',
                'this does not have a compelling factor in it',
                'this movie is amazingly awful'
                ]

def sentiment_parser(sentence):
    chartParser = load_parser('grammar.fcfg',parser=parse.FeatureEarleyChartParser)
    print("Sentence -> ",sentence)
    tree = list(chartParser.parse(sentence.split(" ")))[0]
    print(tree)
    print("\n\n\t\t############################\n\n")
    checkSentiment(tree)
    print("\n\n\t\t############################\n\n")

def checkSentiment(sentence):

    sent = str(sentence)
    pos = [m.start() for m in re.finditer('POS', sent)]
    neg = [m.start() for m in re.finditer('NEG', sent)]
    conjunctionBut = [m.start() for m in re.finditer('but', sent)]

    if len(pos) > len(neg) and len(conjunctionBut) == 0:
        print("\t\tSentence is positive.")
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
            print("\t\tSentence is Positive.")
        else:
            print("\t\tSentence if Negative")
    elif len(pos) == len(neg):
        print("\t\tSentence is neutral")
    else:
        print("\t\tSentence is negative.")
   
def sentiment():
    print("Press 1 to run Good Test Cases.")
    print("Press 2 to run False Test Cases.\n")
    choice = input()
    if choice == "1":
        
        print("Press 1 to choose from list.")
        print("Press 2 to run all good cases.")
        goodInput = input()

        if goodInput == "1":
            print("Please choose from below:\n")
            count = 1
            for sentence in goodSentences:
                print(count, ". ", sentence)
                count+=1
            choosenSentence = int(input())
            sentiment_parser(goodSentences[choosenSentence-1])

        elif goodInput == "2":
            for sentence in goodSentences:
                sentiment_parser(sentence)
        else:
            print("Invalid input, please run pipeline again.")

    elif choice == "2":
        print("Press 1 to choose from list.")
        print("Press 2 to run all bad cases.")
        badInput = input()

        if badInput == "1":
            print("Please choose from below:\n")
            count = 1
            for sentence in badSentences:
                print(count, ". ", sentence)
                count+=1
            choosenSentence = int(input())
            sentiment_parser(badSentences[choosenSentence-1])

        elif badInput == "2":
            for sentence in badSentences:
                sentiment_parser(sentence)
        else:
            print("Invalid input, please run pipeline again.")
    else:
        print("Invalid input, please run pipeline again.")

sentiment()