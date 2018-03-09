import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from  copy import deepcopy

np.random.seed(1)

#как у Остякова
UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

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

keys = [i for i in repl.keys()]


# для хранения предобраюотанных комментов
prep_c_train = []
prep_c_test = []
#
for c_train in train_C.values:

    #Делаем преобразование на подобие этого: "ＷＨＡＴＡ  ＦＵＣＫ  ＭＡＮ" --> "WHATA FUCK MAN"
    c_train = unidecode(c_train)
    # to lowercase
    c_train = c_train.lower()

    # drop urls
    c_train = re.sub(r'http(s)?:\/\/\S*? ', " ", c_train)
    # preprocessing with according to repl - замена символов и сокращений индуса
    temp = []
    for word in c_train.split():
        if word in keys:
            temp += [repl[word]]
        else:
            temp += [word]

    c_train = deepcopy(" ".join(temp))
    # drop digits - try don't drop digits
    c_train = ''.join([i for i in c_train if not i.isdigit()])
    # drop punctuations except apostrophes
    p = re.compile(r"(\b[-']\b)|[\W_]")

    prep_c_train += [p.sub(lambda m: (m.group(1) if m.group(1) else " "), c_train)]

for c_test in test_C.values:
    # "ＷＨＡＴＡ  ＦＵＣＫ  ＭＡＮ" --> "WHATA FUCK MAN"
    c_test = unidecode(c_test)
    # to lowercase
    c_test = c_test.lower()

    # drop urls
    c_test = re.sub(r'http(s)?:\/\/\S*? ', " ", c_test)
    # preprocessing with according to repl
    temp = []
    for word in c_test.split():
        if word in keys:
            temp += [repl[word]]
        else:
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
#После выполнения этого скрипта мы получим предобработанные комменты
#Например
print("Some train comment, before: ", train_C.values[777])
print("Some train comment, after: ", prep_c_train[777])