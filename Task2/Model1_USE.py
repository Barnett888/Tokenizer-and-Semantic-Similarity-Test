import pandas as pd 
import os 
import spacy_universal_sentence_encoder

import numpy as np
from numpy import dot
from numpy.linalg import norm

from scipy.stats.stats import pearsonr 
import math

def cosine_similarity(a,b):
    #a = a.flatten()
    #b = b.flatten()
    return ((a.dot(b)/(norm(a)*norm(b)))+1)*5/2

# Read txt file with correct local directory
path = os.getcwd()
print(path)
input_path = os.chdir("c:/Users/94248/Desktop/Natural_Language_Processing/A1/Task2")

###################################### MODEL 1: USE ###############################################################
###################################################################################################################
# Group 1: Answer-Answer 
# Read txt and remove unnecessary url, author column
answer_answer = pd.read_table('STS2016.input.answer-answer.txt', header=None, delimiter=None, on_bad_lines='skip')
answer_answer = answer_answer.drop(answer_answer.columns[3], axis=1)
answer_answer = answer_answer.drop(answer_answer.columns[2], axis=1)

# load one of the models: ['en_use_md', 'en_use_lg', 'xx_use_md', 'xx_use_lg']
nlp = spacy_universal_sentence_encoder.load_model('en_use_lg')
col1_group1 = answer_answer[0]
col2_group1 = answer_answer[1]

# nlp.pipe gives a generator that yields Doc objects
# not a list. So if you want to use it like a list, you’ll have to call list() on it first
doc1_group1 = list(nlp.pipe(col1_group1))
doc2_group1 = list(nlp.pipe(col2_group1))

# obtain embeddings, as a list
vector1_group1 = [term.vector for term in doc1_group1]
vector2_group1 = [term.vector for term in doc2_group1]

vector1_group1 = np.asarray(vector1_group1)
vector2_group1 = np.asarray(vector2_group1)

g1_score = []
for index in range(len(vector1_group1)):
    g1_score.append(cosine_similarity(vector1_group1[index,:],vector2_group1[index,:]))
    index += 1

g1_score = np.asarray(g1_score)
#########################################################################################################################
#########################################################################################################################
# Group 2: Question-Question 
# Read txt and remove unnecessary url, author column
question_question = pd.read_table('STS2016.input.question-question.txt', header=None, delimiter=None, on_bad_lines='skip')
question_question = question_question.drop(question_question.columns[3], axis=1)
question_question = question_question.drop(question_question.columns[2], axis=1)

col1_group2 = question_question[0]
col2_group2 = question_question[1]

doc1_group2 = list(nlp.pipe(col1_group2))
doc2_group2 = list(nlp.pipe(col2_group2))

# obtain embeddings, as a list
vector1_group2 = [term.vector for term in doc1_group2]
vector2_group2 = [term.vector for term in doc2_group2]

vector1_group2 = np.asarray(vector1_group2)
vector2_group2 = np.asarray(vector2_group2)

g2_score = []
for index in range(len(vector1_group2)):
    g2_score.append(cosine_similarity(vector1_group2[index,:],vector2_group2[index,:]))
    index += 1
g2_score = np.asarray(g2_score)
###################################################################################################################

#########################################################################################################################
#########################################################################################################################
# Group 3: Headlines 
# Read txt and remove unnecessary url, author column
headlines = pd.read_table('STS2016.input.headlines.txt', header=None, delimiter=None, on_bad_lines='skip')
headlines = headlines.drop(headlines.columns[3], axis=1)
headlines = headlines.drop(headlines.columns[2], axis=1)

col1_group3 = headlines[0]
col2_group3 = headlines[1]

# nlp.pipe gives a generator that yields Doc objects
doc1_group3 = list(nlp.pipe(col1_group3))
doc2_group3 = list(nlp.pipe(col2_group3))

# obtain embeddings, as a list
vector1_group3 = [term.vector for term in doc1_group3]
vector2_group3 = [term.vector for term in doc2_group3]

vector1_group3 = np.asarray(vector1_group3)
vector2_group3 = np.asarray(vector2_group3)

g3_score = []
for index in range(len(vector1_group3)):
    g3_score.append(cosine_similarity(vector1_group3[index,:],vector2_group3[index,:]))
    index += 1
g3_score = np.asarray(g3_score)
###################################################################################################################

#########################################################################################################################
#########################################################################################################################
# Group 4: Plagiarism
# Read txt and remove unnecessary url, author column
plagiarism = pd.read_table('STS2016.input.plagiarism.txt', header=None, delimiter=None, on_bad_lines='skip')
plagiarism = plagiarism.drop(plagiarism.columns[3], axis=1)
plagiarism = plagiarism.drop(plagiarism.columns[2], axis=1)

col1_group4 = headlines[0]
col2_group4 = headlines[1]

doc1_group4 = list(nlp.pipe(col1_group4))
doc2_group4 = list(nlp.pipe(col2_group4))

vector1_group4 = [term.vector for term in doc1_group4]
vector2_group4 = [term.vector for term in doc2_group4]

vector1_group4 = np.asarray(vector1_group4)
vector2_group4 = np.asarray(vector2_group4)

g4_score = []
for index in range(len(vector1_group4)):
    g4_score.append(cosine_similarity(vector1_group4[index,:],vector2_group4[index,:]))
    index += 1
g4_score = np.asarray(g4_score)
###################################################################################################################

#########################################################################################################################
#########################################################################################################################
# Group 5: Postediting
# Read txt and remove unnecessary url, author column
postediting = pd.read_table('STS2016.input.postediting.txt', header=None, delimiter=None, on_bad_lines='skip')
postediting = postediting.drop(postediting.columns[3], axis=1)
postediting = postediting.drop(postediting.columns[2], axis=1)

col1_group5 = headlines[0]
col2_group5 = headlines[1]

doc1_group5 = list(nlp.pipe(col1_group5))
doc2_group5 = list(nlp.pipe(col2_group5))

vector1_group5 = [term.vector for term in doc1_group5]
vector2_group5 = [term.vector for term in doc2_group5]

vector1_group5 = np.asarray(vector1_group5)
vector2_group5 = np.asarray(vector2_group5)

g5_score = []
for index in range(len(vector1_group5)):
    g5_score.append(cosine_similarity(vector1_group5[index,:],vector2_group5[index,:]))
    index += 1
g5_score = np.asarray(g5_score)

###################################################################################################################
############################### Score for MODEL 1 ################################################################
############################## Pearson Correlation with human similarity score ###############################################

"""
human_score_group1 = np.array((np.asarray(pd.read_table('STS2016.gs.answer-answer.txt', header=None, delimiter=None, on_bad_lines='skip')).flatten()).tolist())
human_score_group2 = np.array((np.asarray(pd.read_table('STS2016.gs.question-question.txt', header=None, delimiter=None, on_bad_lines='skip')).flatten()).tolist())
human_score_group3 = np.array((np.asarray(pd.read_table('STS2016.gs.headlines.txt', header=None, delimiter=None, on_bad_lines='skip')).flatten()).tolist())
human_score_group4 = np.array((np.asarray(pd.read_table('STS2016.gs.plagiarism.txt', header=None, delimiter=None, on_bad_lines='skip')).flatten()).tolist())
human_score_group5 = np.array((np.asarray(pd.read_table('STS2016.gs.postediting.txt', header=None, delimiter=None, on_bad_lines='skip')).flatten()).tolist())

"""

"""
print("length vector1group1", len(vector1_group1))
print("length vector1group2", len(vector1_group2))
print("length vector1group3", len(vector1_group3))
print("length vector1group4", len(vector1_group4))
print("length vector1group5", len(vector1_group5))

print("g1score", g1_score)
print("length g1score is ",len(g1_score))
print("length g2score is ",len(g2_score))
print("length g3score is ",len(g3_score))
print("length g4score is ",len(g4_score))
print("length g5score is ",len(g5_score))

print("humangroup1",human_score_group1)
print("length hugroup1 is", len(human_score_group1))
print("length hugroup2 is", len(human_score_group2))
print("length hugroup3 is", len(human_score_group3))
print("length hugroup4 is", len(human_score_group4))
print("length hugroup5 is", len(human_score_group5))

#r1 = pearsonr(human_score_group1, g1_score[0:len(human_score_group1)])
#print("r1", r1)
#r2 = pearsonr(human_score_group2, g2_score[0:len(human_score_group2)])
#print("r2", r2)
#r3 = pearsonr(human_score_group3, g3_score[0:len(human_score_group3)])
#print("r3", r3)
#r4 = pearsonr(human_score_group4, g4_score[0:len(human_score_group4)])
#print("r4", r4)
#r5 = pearsonr(human_score_group5, g5_score[0:len(human_score_group5)])
#print("r5", r5)
"""

np.savetxt("SYSTEM_OUT.answer-answer-Model1.txt", np.array(g1_score))
np.savetxt("SYSTEM_OUT.question-question-Model1.txt", np.array(g2_score))
np.savetxt("SYSTEM_OUT.headlines-Model1.txt", np.array(g3_score))
np.savetxt("SYSTEM_OUT.plagiarism-Model1.txt", np.array(g4_score))
np.savetxt("SYSTEM_OUT.postediting-Model1.txt", np.array(g5_score))