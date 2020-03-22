import requests
import re
import time

# base url to fetch text corpus from
base = 'http://www.paulgraham.com/'

# get the html link for the corresponding article whose text is to be fetched
res = requests.get('{}articles.html'.format(base)).text

# using regular expression to find article .html file name from the HTML web page content
links = re.findall(
    r'<font size=2 face="verdana"><a href="([a-zA-Z0-9\-\.html]*)">',
    res
)

# format the fetched .html file names to generate absolute links
links = ['{}{}'.format(base, link) for link in links]

# parse every link for text
for i, link in enumerate(links):

    # get the article html, without abusing the site
    time.sleep(0.1)
    res = requests.get(link).text  # getting raw response text from the the link

    # find the essay content by searching for the below pattern in our web response content
    article = re.findall(
        r'<font size=2 face="verdana">(.*)<br><br></font>',
        res, re.DOTALL
    )

    # if the text article fetched has length of 0 then search over another pattern in the html text.
    if len(article) == 0:
        article = re.findall(
            r'<font size=2 face="verdana">(.*)<br><br><br clear=all></font>',
            res, re.DOTALL
        )

    # having fetched the essay content, replace the <br> tags with newline character tags in Python so as to
    # maintain the line break characteristics in our essay
    filter_words = ['<center>_____</center>', '<a href="(.*)">', '<u>', '</u>', '</a>', '<b>', '</b>', '<i>', '</i>']
    try:
        article = article[0].replace('<br>', '\n')  # replacing <br> tags with new line tags

        for ft in filter_words:
            article = re.sub(ft, '', article)

        article += '\n'  # finally append new line character into our article

        # write retreived text in a file
        with open('paul_graham_essay.txt', 'a') as f:
            f.write(article)
        print(i + 1, '. Text taken from: ', link)

    except IndexError:
        print(i + 1, '. Could not get text from: ', link)
