import xml.etree.ElementTree as ET
import os
import codecs
import sys
import numpy as np
import random
import os
import math
from reader import get_gold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#u = urlopen('E:/SCIENCE/nlp/final_project/Universal_Word_Segmentation/ud-treebanks-conll2017/UD_Ancient_Greek/stats.xml')

filePath = 'ud-treebanks-conll2017'
filelist = os.listdir(filePath)
languages_list = []
for language in filelist:
	f = open( filePath+'/'+language+'/stats.xml' , mode = 'r' , encoding= ' utf-8 ' )
	tree = ET.parse(f)
	root = tree.getroot()
	languages_dict={}
	languages_dict['name'] = language
	for item in tree.iterfind('size/total/sentences'):
		languages_dict['SentenceNum'] = item.text

	for item in tree.iterfind('size/total/tokens'):
		languages_dict['TokensNum'] = item.text

	for item in tree.iterfind('size/total/words'):
		languages_dict['WordsNum'] = item.text

	for item in tree.iterfind('size/total/fused'):
		languages_dict['FusedNum'] = item.text
	for item in tree.iterfind('forms'):
		languages_dict['LS'] = float(item.attrib['unique'])
	infilelist = os.listdir(filePath+'/'+language)
	char_set = {}
	spaces = 0
	for file in infilelist:
		if 'train.txt' in file:
			for line in codecs.open('ud-treebanks-conll2017/'+language+ '/'+file, 'rb', encoding='utf-8'):
				spaces+=line.count(' ')
				line = line.strip()
			# if sea == 'sea':
			# 	line = pre_token(line)
				for ch in line:
					if ch in char_set:
						char_set[ch] += 1
					else:
						char_set[ch] = 1
			CS = len(char_set)
		if 'dev.txt' in file:
			for line in codecs.open('ud-treebanks-conll2017/'+language+ '/'+file, 'rb', encoding='utf-8'):
				spaces += line.count(' ')
				line = line.strip()
			# if sea == 'sea':
			# 	line = pre_token(line)
				for ch in line:
					if ch in char_set:
						char_set[ch] += 1
					else:
						char_set[ch] = 1
		CS = len(char_set)
	languages_dict['CS'] = float(CS)
	languages_dict['AL'] = float(CS)/float(languages_dict['WordsNum'])
	languages_dict['MP'] = (float(languages_dict['WordsNum'])-float(languages_dict['TokensNum']))/float(languages_dict['TokensNum'])
	languages_dict['SF'] = float(languages_dict['WordsNum'])/float(spaces)
	languages_list.append(languages_dict)

print(languages_list)
X = []
y = []
for item in languages_list:
	X.append([item['LS'], item['CS'], item['AL'], item['MP'], item['SF']])
	y.append(item['name'])


ss = StandardScaler()
#ss =  MinMaxScaler()
pca = PCA(n_components=2)
X = ss.fit_transform(X)
pca.fit(X)
print(len(X))
XT = pca.transform(X)
XT = np.array(XT)
print(len(XT))
# plt.scatter(XT[:,0],XT[:,1])
# plt.text(XT[:,0]+0.3, XT[:,1]+0.3, y, fontsize=9)
# plt.show()
kmeans = KMeans(n_clusters=8).fit(XT)
label_pred = kmeans.labels_
centroids = kmeans.cluster_centers_
color = ['b','g','r','c','m','y','k','w']
j = 0
for i in label_pred:
	if y[j] == 'UD_Arabic':
		plt.plot([XT[j:j+1,0]], [XT[j:j+1,1]], color = color[i], marker='o')
		plt.text(XT[j:j+1, 0] + 0.1, XT[j:j+1, 1] + 0.1, y[j], fontsize=9)
	j +=1
plt.show()


