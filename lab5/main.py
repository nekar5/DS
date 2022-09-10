import re                                                           # підтримка парсінгу сайту
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt                                     # підтримка парсінгу сайту
from bs4 import BeautifulSoup                                       # підтримка парсінгу сайту
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

#метод парсингу сайту та знаходження й запису назв виступів у файл
def url_parser(url, el, el_class, file):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    quotes_2 = soup.find_all(el, class_=el_class)
    output_file_2 = open(file, 'a')

    for quote in quotes_2:
        quote.encoding = 'cp1251'
        temp=re.sub("’'`", "", quote.text)
        temp=temp.replace('і', 'i')
        output_file_2.write(re.sub(r"\([^()]*\)", "", temp)+"\n")
    return

#метод частотного аналізу даних текст-майнінгу
def make_wordcloud(f):
    text = str(f.readlines())
    words = re.findall('[а-яА-Я]{2,}', text) #- слова з <2 буквами
    stats = {}

    stop_words = stopwords.words("russian")
    sw_arr=['театр', 'оперета', '«','»', '’', 'є', 'яг', 'прем', 'єра', 'и', ]
    stop_words+= sw_arr
    stop_words.remove('без')
    stop_words.remove('им')

    #чистка
    symbols = ['.', '-', ',', '/', '' '', '!', '@', '" "',
              '#', '№', '$', ':', ';', '%', '^', '&', '?',
              '*', '(', ')', '_', '+', '=', '[', ']', '{', '}',
              '"', ' ', '<', '>', '|', '’' '`', '~', ',', '«','»',]
    for w in words:
        for s in symbols:
            w.replace(s, '')

    stop_words_title = [None] * len(stop_words)
    for i in range(len(stop_words)):
        stop_words_title[i] = stop_words[i].title()

    for word in words:
        if word in stop_words:
            for i in range(words.count(word)):
                words.remove(word)
        if word in stop_words_title:
           for i in range(words.count(word)):
                words.remove(word)

    for w in words:
        stats[w] = stats.get(w, 0) + 1
    
    w_ranks = sorted(stats.items(), key=lambda x: x[1], reverse=True)[0:10]
    _wrex = re.findall('[а-яА-Я]+', str(w_ranks))
    _drex = re.findall('[0-9]+', str(w_ranks))

    pl = [p for p in range(10)]
    for j in range(len(_wrex)):
        places = '{} - {} - {} times'.format(pl[j]+1, _wrex[j], _drex[j])
        print('№', places)
    text_raw = " ".join(_wrex)

    wordcloud = WordCloud().generate(text_raw)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    return

#метод докладного аналізу даних текст-майнінгу
def miner(path):
    with open(path) as file:
        text = file.read()

    tokens = word_tokenize(text)

    stop_words = stopwords.words("russian")
    sw_arr=['театр', 'оперета', 'є', 'яг', 'прем', 'єра', 'э', 'рка', 'менна']
    stop_words+=sw_arr

    #токенізація
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)

    #чистка
    regex_numbers = re.compile('^[0-9]{1,4}([,:-][0-9]{1,4})*(\.[0-9]+)?$')
    symbols = ['.', '-', ',', '/', '' '', '!', '@', '" "',
              '#', '№', '$', ':', ';', '%', '^', '&', '?',
              '*', '(', ')', '_', '+', '=', '[', ']', '{', '}',
              '"', ' ', '<', '>', '|', '’' '`', '~', ',', '«','»',]

    for token in filtered_tokens:
        if token in symbols:
            filtered_tokens.remove(token)
    for token in filtered_tokens:
        if (regex_numbers.search(token) != None):
            filtered_tokens.remove(token)

    words = []
    snowball = SnowballStemmer(language="russian")
    for i in filtered_tokens:
        word = snowball.stem(i)
        words.append(word)

    words.sort()
    words_dict = dict()

    for word in words:
        if word in words_dict:
            words_dict[word] = words_dict[word] + 1
        else:
            words_dict[word] = 1

    print("Кількість слів: %d" % len(words))
    print("Кількість унікальних слів: %d" % len(words_dict))
    print("Усі використані слова:")
    for word in words_dict:
        print(word.ljust(20), words_dict[word])

    return


def main():
    print('\nсайт::')
    print('#1 https://kiev.karabas.com/')
    print('#2 https://ticket.kiev.ua/')
    print('#3 https://bilethouse.com/')

    path='C:/Users/Nestor/Desktop/lab5/data.txt'

    mode = int(input('choice:'))
    if (mode == 1):
        print('джерело: https://kiev.karabas.com/')
        el='a'
        el_class='el-name'
        url = 'https://kiev.karabas.com/theatres/'
    if (mode == 2):
        print('джерело: https://kontramarka.ua/')
        url = 'https://ticket.kiev.ua/theater/'
        el='h5'
        el_class='card-title'
    if (mode == 3):
        print('джерело: https://bilethouse.com/')
        el='span'
        el_class='schedule-event'
        url = 'https://bilethouse.com/events/actiongroup/theatres'

    if(not 0<mode<4):
        print('incorrect input')
        exit()

    with open(path, 'w'):
            pass
    url_parser(url, el, el_class, path)

    print('докладний аналіз:', mode, ':', url)
    words_dict = miner(path)

    f = open(path, 'r')
    print('Домінуючий контент сайту:', mode, ':', url)
    make_wordcloud(f)

main()