import os
import emoji
emoji.EMOJI_ALIAS_UNICODE

# directory names
path = '/Users/sharmi/personal-easy/ilady/project/electionsentiment/Emojis/'

dirEmojiUni = path + "Unicode"
dirEmojiTxt = path + "Text_Based"

positiveMap = {}
negativeMap = {}
neutralMap = {}

positiveMap['arousal-person with folded hands'] = '\\U0001F64F'
positiveMap['joy-person raising both hands in celebration'] = '\\U0001F64C'
positiveMap['joy-happy person raising one hand'] = '\\U0001F64B'
positiveMap['joy-happy person raising one hand'] = '\\U0001F64B'
positiveMap['joy-victory hand']= '\\U000270C'
positiveMap['joy-grinning face']= '\\U0001F600'
positiveMap['joy-smiling face with halo']= '\\U0001F607'
positiveMap['arousal-astonished']= '\\U0001F632'
positiveMap['dominance-thumbs_up']= '\\U0001F44D'
positiveMap['arousal-blush']= '\\U0001F60A'
positiveMap['dominance-point_right']= '\\U0001F449'


positiveMap['joy-smiling face with halo']= '\\U0001F607'
positiveMap['joy-clapping hands']= '\\U0001F44F'
positiveMap['joy-smiling face with horns']= '\\U0001F608'
positiveMap['joy-smiling face with sunglasses']= '\\U0001F60E'
positiveMap['joy-smiling face with halo']= '\\U0001F607'
positiveMap['joy-smiling face with halo']= '\\U0001F607'
positiveMap['arousal-face with open mouth']= '\\U0001F62E'
positiveMap['joy-rolling_on_the_floor_laughing'] = '\\U0001F923'
positiveMap['joy'] = '\\U0001F602'
positiveMap['joy-100'] = '\\U0001F4AF'
positiveMap['arousal-face1'] = '\\U0001F914'
positiveMap['joy-wave'] = '\\U0001F44B'
positiveMap['joy-smiling_face_with_smiling_eyes'] = '\\U0001F60A'
positiveMap['arousal-winking_face'] = '\\U0001F609'
positiveMap['arousal-winking_face_with_tongue'] = '\\U0001F61C'


neutralMap['neutral-face with ok gesture'] = '\\U0001F646'
neutralMap['neutral-speak-no-evil monkey'] = '\\U0001F64A'
neutralMap['neutral-hear-no-evil monkey'] = '\\U0001F649'
neutralMap['neutral-see-no-evil monkey'] = '\\U0001F648'
neutralMap['neutral-innocent'] = '\\U0001F607'
positiveMap['neutral-point_up_2'] = '\\U0001F446'


negativeMap['sad-face with stuck-out tongue'] = '\\U0001F61B'
negativeMap['angry4-grimacing face'] = '\\U0001F62C'
negativeMap['angry3'] ='\\U0001F4A2'
negativeMap['angry2'] ='\\U0001F620'
negativeMap['angry-anguished'] ='\\U0001F627'
negativeMap['angry1'] ='\\U0001F620'
negativeMap['sad-disappointed_relieved'] ='\\U0001F625'
negativeMap['sad-disappointed'] ='\\U0001F61E'
negativeMap['sad-lion face'] ='\\U0001F981'
negativeMap['sad-point_down'] = '\\U0001F447'
negativeMap['sad-face_with_tears_of_joy'] = '\\U0001F602'


def readUnicode(directory):
    unicodeList = {}
    for file in os.listdir(directory):
        infile = open(directory + "/" + file)
        for line in infile:
            line = line.rstrip("\n")
            line = line.lstrip(" ")
            lineList = line.split(' ', 1)
            ucoded = lineList[0].encode('unicode-escape').decode('utf8')[1:]
            if ('negative' in file):
                unicodeList[ucoded] =  [lineList[1], -1]
            elif ('positive' in file):
                unicodeList[ucoded] = ['joy-' + lineList[1], -1]
            elif ('neutral' in file):
                unicodeList[ucoded] = ['neutral-' + lineList[1], -1]

    for i in range(len(positiveMap.keys())):
        key = list(positiveMap.values())[i][1:]
        value = list(positiveMap.keys())[i]
        unicodeList[key] = [ value, 1]

    for i in range(len(neutralMap.keys())):
        key = list(neutralMap.values())[i][1:]
        value = list(neutralMap.keys())[i]
        unicodeList[key] = [value, 1]


    for i in range(len(negativeMap.keys())):
        key = list(negativeMap.values())[i][1:]
        value = list(negativeMap.keys())[i]
        unicodeList[key] = [value, 1]


    return unicodeList



def readEmoticon(directory):
    emoticonList = {}
    for file in os.listdir(directory):
        infile = open(directory + "/" + file)
        for line in infile:
            line = line.rstrip("\n")
            if ('negative' in file):
                emoticonList[line] = -1
            elif ('positive' in file):
                emoticonList[line] = 1
            elif ('neutral' in file):
                emoticonList[line] = 0

    return emoticonList

## Create the function to extract the emojis
def extract_emojis(a_list):
    emojis_list = map(lambda x: ''.join(x.split()), emoji.UNICODE_EMOJI.keys())
    r = re.compile('|'.join(re.escape(p) for p in emojis_list))
    aux = [' '.join(r.findall(s)) for s in a_list]
    return (aux)



if __name__ == '__main__':
    import emoji
    import re

    unicodeEmojiList = readUnicode(dirEmojiUni)
    print(unicodeEmojiList)
    emoticonList = readEmoticon(dirEmojiTxt)
    print(emoticonList)