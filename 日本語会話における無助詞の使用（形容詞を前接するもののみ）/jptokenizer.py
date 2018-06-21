#! /usr/bin/python
# -*- coding: utf-8 -*-

import nltk
import re
from nltk.tokenize.api import *

class JPMeCabTokenizer(TokenizerI):
    def __init__(self):
        import MeCab
        self.mecab = MeCab.Tagger('-Ochasen')

    def tokenize(self, text):
        #preprocessing
        if '：' in text:
            text = text[text.find('：')+1:]
        #parsing
        node = self.mecab.parseToNode(text)
        node = node.next
        result = []
        while node:
            result.append((node.surface, node.feature))
            node = node.next
        return result

    
