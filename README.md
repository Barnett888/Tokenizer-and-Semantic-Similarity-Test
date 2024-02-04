# Tokenizer-and-Semantic-Similarity-Test
A group project to implement tokenizer and run LLMs to perform STS

## Task1: Corpus processing (legal text) tokenization and word counting

### Task Description
Implement a word tokenizer that splits texts into tokens and separates punctuation marks and other symbols from the words. Implement a program that counts the number of occurrences of each token in the corpus.
Use the Atticus dataset of legal contacts: https://zenodo.org/record/4595826#.YyXT6HbMI2w
Download the file CUAD_v1.zip, unzip, and see the folder full_contact_txt/
It contains 510 files with full text contracts (a collection of TXT files of the underlying contracts). Each file is named as “[document name].txt”. These contracts are in a plaintext format and are not labeled. You will need to concatenate all the text files to form a corpus.

Provide the following information about the corpus:
a) Generate a file output.txt with the tokenizer’s output for the whole corpus. Include the first 20 lines from output.txt. <br />
b) How many tokens did you find in the corpus? How many types (unique tokens) did you have? What is the type/token ratio for the corpus? The type/token ratio is defined as the number of types divided by the number of tokens. <br />
c) For each token, print the token and its frequency in a file called tokens.txt (from the most frequent to the least frequent) and include the first 20 lines. <br />
d) How many tokens appeared only once in the corpus? <br />
e) From the list of tokens, extract only words, by excluding punctuation and other symbols, if any. Please pay attention to end of sentence dot (full stops). How many words did you find? List the top 20 most frequent words, with their frequencies. What is the type/token ratio when you use only words (called lexical diversity)? <br />
f) From the list of words, exclude stopwords. List the top 20 most frequent words and their frequencies. You can use this list of stopwords (or any other that you consider adequate). Also compute the type/token ratio when you use only word tokens without stopwords (called lexical density)? <br />
g) Compute all the pairs of two consecutive words (bigrams) (excluding stopwords and punctuation). List the most frequent 20 pairs and their frequencies in your report. <br />

### How to run code


### Output file description




## Task2: Semantic Similarity Test (STS), Evaluation of pre-trained sentence embedding models 

### Task Description
Word embeddings are dense representations of the meaning of words, build via neural language models. Sentence embeddings are dense representations of sentences, that can be composed by averaging the word vectors or sentence representations can be learned directly. Chose at least 5 pre-trained sentence embeddings. Include a version of SBERT (Reimers and Gurevych, 2019, https://aclanthology.org/D19-1410/) (https://www.sbert.net/) and one model based on recent generative LLM models, in addition to other pre-trained language models that can represent sentences.

Use the dataset from the Semeval 2016-Task1 Semantic Textual Similarity (STS).
Use the test data STS Core (English Monolingual subtask) - test data with gold labels.  Do not use the training data. Read more about the task at https://alt.qcri.org/semeval2016/task1/#

The evaluation score to report is the Pearson correlation between the score obtained by your model and the expected solution. The expected solution scores are numeric values between and 5 (from low similarity to high similarity).

### How to run code

### Output file description





## Authors

Xiao Yang: 
coded all STS model scipts; wrote report for USE, BERT, RoBERTa section; editted task 1, 2 report; created github repo, readme.

Mingze Li:
coded all task 1 scripts; wrote task 1 section of report.

Sara Haroon: 
tested task 2 model scripts; ran pearl to examine pearson coefficents; wrote report for SBERT, DistilBERT, task2 result section. 
