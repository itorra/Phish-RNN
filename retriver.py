import json
from urllib.request import urlopen
import re

yemkey = 'CC881D79509D124E351D'


def getAllShows(year_from,year_to):
    l = list()
    for year in range(year_from, year_to+1):
        url = getJsonUrlAllShowsOfYear(year)
        file = getJsonbyUrl(url)
        if type(file) == list:
            for show in file:
                l.append(show['showid'])
    return l


def getJsonUrlAllShowsOfYear(year):
    return "https://api.phish.net/api.js?api=2.0&format=json&method=pnet.shows.query&year=" + str(
        year) + "&apikey=" + yemkey


def getJsonbyUrl(url):
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)

def getJsonUrlSetlist(showid):
    return "https://api.phish.net/api.js?showid=" + showid + "&api=2.0&format=json&method=pnet.shows.setlists.get&apikey=" + yemkey

def analyzeShow(show):
    show = getJsonbyUrl(getJsonUrlSetlist(show))


def fixSetlist(sl):
    cleanr = re.compile('<.*?>')
    sl = sl.replace("Come On Baby, Let's Go Downtown","Come On Baby Let's Go Downtown")
    sl = sl.replace("My Friend, My Friend","My Friend My Friend")
    sl = sl.replace("Ob-La-Di, Ob-La-Da","Ob-La-Di Ob-La-Da")
    sl = sl.replace("Love Reign O'er Me","Love Reign O'er Me")
    sl = sl.replace("Swing Low, Sweet Chariot","Swing Low Sweet Chariot")
    sl = sl.replace("I'm Blue, I'm Lonesome","I'm Blue I'm Lonesome")
    sl = sl.replace("Set 1:", "S1, ")
    sl = sl.replace("Set 2:", ", S2, ")
    sl = sl.replace("Set 3:", ", S3, ")
    sl = sl.replace("Set 4:", ", S4, ")
    sl = sl.replace("Encore:", ", ENCORE,")
    sl = sl.replace("Encore 2:", ", ENCORE,")
    sl = re.sub(cleanr,'',sl)
    sl = sl.replace(" >",",")
    sl = sl.replace(" ->", ",")
    sl = sl.replace(",  ", ",")
    sl = sl.replace(", ", ",")
    notes = notesNum(sl)
    if notes > 0:
        sl = sl[0:sl.rfind('[1]')]
        for i in range(1,notes+1):
            s = '[' + str(i) + ']'
            sl = sl.replace(s,"")
    sl += ",END"
    return sl


def notesNum(sl):
    end = len(sl)
    i = sl.rfind(']')
    while i > 0:
        if sl[i-1] > '0' and sl[i-1] <= '9':
            return int(sl[i-1])
        else:
            end = i-1
            i = sl.rfind(']',0,end)
    return 0;

def createDataSet(filename,year_from,year_to):
    shows = getAllShows(year_from,year_to)
    filename += '.yem'
    target = open(filename,mode='w')
    for show in shows:
        dataurl = getJsonUrlSetlist(show)
        data = getJsonbyUrl(dataurl)
        setlist = data[0]['setlistdata']
        print(data[0]['showdate'] + data[0]['showid'])
        setlist = fixSetlist(setlist)
        target.write(setlist)
        target.write("\n")


#createDataSet('data_new',1983,2015)

