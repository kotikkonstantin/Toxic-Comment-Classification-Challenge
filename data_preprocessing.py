
#Thanks to laol,ODS and Pavel Pleskov for some useful ideas and materials

import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from copy import deepcopy
from nltk.corpus import stopwords
from stop_words import get_stop_words
import time
from tqdm import tqdm

start_time = time.time()

np.random.seed(1)

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

#length thresold for deleting words
WORD_LENGTH_THRESOLD=3

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")

data_train = data_train.fillna(NAN_WORD)
data_test = data_test.fillna(NAN_WORD)

print("Train shape: {}".format(data_train.shape))
print("Test shape: {}".format(data_test.shape))


train_C = data_train.comment_text
test_C =  data_test.comment_text

#это труды индуса -- замена символов и сокращений
repl = {
    "&lt;3": " good ",
    ":d": " good ",
    ":dd": " good ",
    ":p": " good ",
    "8)": " good ",
    ":-)": " good ",
    ":)": " good ",
    ";)": " good ",
    "(-:": " good ",
    "(:": " good ",
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " bad ",
    ":(": " bad ",
    ":s": " bad ",
    ":-s": " bad ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

curse = {"fuck":"fuck",
         "suck":"suck",
         "cunt":"cunt",
         "fuk":"fuck",
         "crap":"crap",
         "cock":"cock",
         "dick":"dick",
         "dumb":"dumb",
         "shit":"shit",
         "bitch":"bitch",
         "damn":"damn",
         "piss":"piss",
         "gay":"gay",
         "fag":"faggot",
         "assh":"asshole",
         "basta":"bastard",
         "douch":"douche",
         "haha":"haha",
         "nigger":"nigger",
         "penis":"penis",
         "vagina":"vagina",
         "niggors":"niggers",
         "nigors":"nigers",
         "fvckers":"fuckers",
         "phck":"fuck",
         "fack":"fuck",
         "sex":"sex",
         "wiki":"wikipedia",
         "viki":"wikipedia",
        }

fuck = {"f**c":"fuck",
        "f**k":"fuck",
        "f**i":"fuck",
        "f*ck":"fuck",
        "fu*k":"fuck",
        "shi*":"shit",
        "s**t":"shit",
        "sh*t":"shit",
        "f***":"fuck",
        "****i":"fuck",
        "c**t":"cunt",
        "b**ch":"bitch",
        "d**n":"damn",
        "*uck":"fuck",
        "fc*k":"fuck",
        "fu**":"fuck",
        "f*k":"fuck",
        "fuc*":"fuck",
        "f**":"fuck"
        #"fck":"fuck" --- ???
        }

keys = [i for i in repl.keys()]
keys_curse = [i for i in curse.keys()]
fuck_keys = [i for i in fuck.keys()]

# для хранения предобраюотанных комментов
prep_c_train = []
prep_c_test = []

for c_train in tqdm(train_C.values):

    #Делаем преобразование на подобие этого: "ＷＨＡＴＡ  ＦＵＣＫ  ＭＡＮ" --> "WHATA FUCK MAN"
    c_train = unidecode(c_train)
    # to lowercase
    c_train = c_train.lower()

    # drop urls
    c_train = re.sub(r'http(s)?:\/\/\S*? ', " ", c_train)
    # drop ips
    c_train = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', c_train, flags=re.MULTILINE)
    # drop images
    c_train = re.sub("image:.*\.jpg|png", " ", c_train, flags=re.IGNORECASE)
    # replace in words with 3 and more same symbol to only one symbol
    c_train = re.sub(r'(.)\1{2,}', r'\1', c_train)

    # preprocessing with according to repl,curse,fuck - замена символов и сокращений
    temp = []
    for word in c_train.split():
        if word in keys:
            temp += [repl[word]]
        elif word in keys_curse:
            temp.append(curse[word])
        elif word in fuck_keys:
            temp.append(fuck[word])
        else:
            temp += [word]
    c_train = deepcopy(" ".join(temp))

    # deleting stop words in different languages by nltk
    temp = []
    flag = True
    for word in c_train.split():
        for lang in ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian',\
                     'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish']:
            if word in stopwords.words(lang):
                flag=False
                break
            flag=True

        if flag:
            temp += [word]
    c_train = deepcopy(" ".join(temp))

    # deleting stop words in different languages by stop-words 2015.2.23.1
    temp = []
    flag = True
    for word in c_train.split():
        for lang in ['arabic', 'catalan', 'romanian', 'ukrainian']:
            if word in get_stop_words(lang):
                flag=False
                break
            flag=True

        if flag:
            temp += [word]
    c_train = deepcopy(" ".join(temp))

    # drop words with length <= WORD_LENGTH_THRESOLD
    temp = []
    for word in c_train.split():
        if len(word) <= WORD_LENGTH_THRESOLD:
            continue
        temp += [word]
    c_train = deepcopy(" ".join(temp))

    # drop digits - try don't drop digits
    c_train = ''.join([i for i in c_train if not i.isdigit()])
    # drop punctuations except apostrophes
    p = re.compile(r"(\b[-']\b)|[\W_]")

    prep_c_train += [p.sub(lambda m: (m.group(1) if m.group(1) else " "), c_train)]


for c_test in tqdm(test_C.values):
    # "ＷＨＡＴＡ  ＦＵＣＫ  ＭＡＮ" --> "WHATA FUCK MAN"
    c_test = unidecode(c_test)
    # to lowercase
    c_test = c_test.lower()

    # drop urls
    c_test = re.sub(r'http(s)?:\/\/\S*? ', " ", c_test)
    # drop ips
    c_test = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', c_test, flags=re.MULTILINE)
    # drop images
    c_test = re.sub("image:.*\.jpg|png", " ", c_test, flags=re.IGNORECASE)
    # replace in words with 3 and more same symbol to only one symbol
    c_test = re.sub(r'(.)\1{2,}', r'\1', c_test)

    # preprocessing with according to repl,curse,fuck - замена символов и сокращений
    temp = []
    for word in c_test.split():
        if word in keys:
            temp += [repl[word]]
        elif word in keys_curse:
            temp.append(curse[word])
        elif word in fuck_keys:
            temp.append(fuck[word])
        else:
            temp += [word]
    c_test = deepcopy(" ".join(temp))

    # deleting stop words in different languages by nltk
    temp = []
    flag = True
    for word in c_test.split():
        for lang in ['danish', 'dutch', 'english', 'finnish', 'french', 'german', 'hungarian', 'italian', \
                     'norwegian', 'portuguese', 'russian', 'spanish', 'swedish', 'turkish']:
            if word in stopwords.words(lang):
                flag = False
                break
            flag = True

        if flag:
            temp += [word]
    c_test = deepcopy(" ".join(temp))

    # deleting stop words in different languages by stop-words 2015.2.23.1
    temp = []
    flag = True
    for word in c_test.split():
        for lang in ['arabic', 'catalan', 'romanian', 'ukrainian']:
            if word in get_stop_words(lang):
                flag = False
                break
            flag = True

        if flag:
            temp += [word]
    c_test = deepcopy(" ".join(temp))

    # drop words with length <= WORD_LENGTH_THRESOLD
    temp = []
    for word in c_test.split():
        if len(word) <= WORD_LENGTH_THRESOLD:
            continue
        temp += [word]
    c_test = deepcopy(" ".join(temp))

    # drop digits
    c_test = ''.join([i for i in c_test if not i.isdigit()])
    # drop punctuations except apostrophes
    p = re.compile(r"(\b[-']\b)|[\W_]")

    prep_c_test += [p.sub(lambda m: (m.group(1) if m.group(1) else " "), c_test)]

print("Data preprocessing has done.")
prep_c_train = np.array(prep_c_train) #np.array
prep_c_test = np.array(prep_c_test)     #np.array

#save files
pd.DataFrame(prep_c_train).to_csv("preprocessed_train_comments.csv",index=False,header=False)
pd.DataFrame(prep_c_test).to_csv("preprocessed_test_comments.csv",index=False,header=False)
print("Preprocessed comments saved")

#После выполнения этого скрипта мы получим предобработанные комменты
#Например
print("Some train comment, before: ", train_C.values[777])
print("Some train comment, after: ", prep_c_train[777])

print("Data_preprocesing has been done for {} seconds.".format(time.time() - start_time))

