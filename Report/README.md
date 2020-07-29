# COVID-19 Report

## Team Members
Kalyani Goyal - 170010009  
Pulaksh Garg - 170010015  
Rahul Baviskar - 170010004  
Raj Hansini Khoiwal - 170010014  
Raj Jagtap - 170030004  
Rajat Kumar - 1913105    

## Table Of Contents
- [Introduction](#Introduction)
- [Preprocessing](#Pre-processing)
    - [Reading the Dataset](#Reading-the-Dataset)
    - [Cleaning up Dataset](#Cleaning-up-Dataset)
    - [Filtering Text in Datset](#Filtering-Text-in-Datset)
    - [Filtering English Documents](#Filtering-English-Documents)
    - [Removing small papers from Dataset](#Removing-small-papers-from-Dataset)
    - [Generating Stopwords](#Generating-Stopwords)
- [Sentence Transformers](#Sentence-Transformers)
- [TF-IDF](#TF-IDF)
- [Topic Modelling : LDA](#Latent-Dirichlet-Allocation)
- [Word as Vector](#Word-as-Vector)
  - [GloVe Vectors](#GloVe-Vectors)
  - [Document to Vector](#Document-to-Vector)
  - [Question to Vector](#Question-to-Vector)
  - [Cosine Similarity](#Cosine-Similarity)
  - [KNN Approach](#KNN-Approach)
- [1D CNN](#1D-CNN)
- [Text Summarisation](#Text-Summarisation)
- [Analysis](#Analysis)
    - [TF-IDF](#TF-IDF1)
    - [Word Vectors](#Word-to-Vectors-Cosine-Similarity-amp-KNN-Approach)
    - [1D CNN](#1D-CNN1)
    - [Sentence Transformers](#Sentence-Transformers-BERT)
    - [LDA](#LDA)
- [Experimental Results Comparison](#Experimental-Results-Comparison)
- [Related Work](#Related-Work)
- [Conclusion](#Conclusion)
- [References](#References)

## Introduction
> The task chosen is "What do we know about virus genetics, origin, and evolution? What do we know about Real-time tracking of whole genomes?"
The aim is to take the former mentioned query and apply a machine learning technique to retrieve the answer from the corpus (group of documents). We have applied five different methods for information retrieval. Each method has been explained and lastly analysis and comparison has been done for each method to give a qualitative approach in regards to the merits and demerits of each method. 


## Pre-processing
We used multiple functions to process the json files with raw data to convert it into standard usable format.
### Reading the Dataset
> all json paths ---> read_dataset ---> dataframe
> 
We use ```read_dataset``` to read all the json files sequentially and store thier *title*, *authors*, *abstract*, *body_text*, etc. in *pandas* dataframe.

### Cleaning up Dataset
> dataframe ---> data_cleanup ---> dataframe 

We use ```data_cleanup``` function to remove all duplicates and empty entries from dataframe.

Duplicates are found based on pairs (*abstract*, *body_text*) or (*abstract*, *filtered_body_text*).

### Filtering Text in Datset
> dataframe ---> text_filtering ---> dataframe

We use ```text_filtering``` function to 
1. Remove text inside [] and () as it is most likely to be references and comments.
2. Filter only alphanumeric characters in text as we cannot use <img src="https://render.githubusercontent.com/render/math?math=$\alpha , \beta, \sum$" > etc. special characters while applying methods below.
3. Remove \n and \t.
4. Remove extra spaces.
5. Lowercase all the characters.

### Filtering English Documents
> dataframe ---> filter_english ---> dataframe

We use ```filter_english``` function to select only english documents from the dataset.
We use a module *langdetect* to detect the language of the filtered body text.
If the language is detectable and is English then only select the document.

### Removing small papers from Dataset
> dataframe ---> remove_small_papers ---> dataset

We use ```remove_small_papers``` function to remove small papers from the dataset.
We check if the filtered body text is less than 500 chars, if yes then do not take select the document.

### Generating Stopwords
> The stop words in the model comes from three sources namely:
* Customized stop words like 'doi', 'preprint'  etc
* Stopwords imported from spacy.lang.en.stop_words library
* Stopwords imported from library

A final set with all the above was generated and then applied on the corpus. In the function words_freq_atleast_2 the stop words are removed from the text of the document passed in the arguemenr of the function.
    


## Sentence Transformers
> **Idea** : This method explores an encoding method for the data known as *sentence transformer* library. 

* First and foremost data preprocessing included removing square bracket contents, content in parenthesis, punctuation but hyphen, extra spaces and making everthing lower case. Also filtered out non-english papers and papers with less than 500 words. 
* Using *nltk tokenize* we split the data part of the research paper(single data loci).
* Use *sentence tranformers* to generate a matrix with 768 components for each sentence. 
* Then we calculate average over a paper and also find the norm(root mean square) of the matrix. 
* Using these embeddings we calculate the cosine similarity with our questions (sentences) and list the top similar results. 
 
The problem encountered was that this method was computationally quite taxing. 

## TF-IDF
>**Idea** : It is the principal technique used for removing stop words from the corpus apart from the normalization techniques learned in the course. It works on the term-document matrix initially generated. 
* TF-IDF helps you to see how important a word is to your document. The matrix comprises of D document vectors of vocabulary size V. In the TF_IDF matrix each document has it's own V sized vector.
* The query(question) is converted to a similar V sized vector by doing pre-processing of the query. 
* Then using Cosine Similarity top N most relevant document's indexes are stores in a list. 
* Then this list is passed to the function generate_summary() which performs text summarization and will retrieve the K most relevant sentences with concerning the query and display the final answer.
* Note that N and K both are input parameters to be set by the user.

The library imported for TF-IDF is
* sklearn.feature_extraction.text
* TfidfVectorizer
* sklearn.metrics.pairwise
* linear_kernel

Formula for calculation of TF_IDF matrix is given below:
<center>
<img src="https://render.githubusercontent.com/render/math?math=$tf\text{-}idf=tf_{dt} * log(\frac{D}{d_{ft}})$" >
</center>
<br>


* where <img src="https://render.githubusercontent.com/render/math?math=$tf_{dt}$" > is the term frequency 
* *D* is the total number of documents
* <img src="https://render.githubusercontent.com/render/math?math=$d_{ft}$" > is the total number of documents in the corpus   in which the term occurs. 
### Example run
* The query given to the model is "What is the evidence that livestock could be infected (field surveillance, genetic sequencing, receptor binding) Evidence of whether farmers are infected, and whether farmers could have played a role in the origin."
* The results generated are as follows:
> *  Summary for article: A questionnaire-based survey on the uptake and use of cattle vaccines in the UK Eighty-six per cent (n=229/266) of respondents indicated that they had vaccinated their cattle in the past year, with more dairy farmers vaccinating compared to beef farmers (Table 3 ). Timing of vaccination may also be important.
> *  Summary for article: Environmental Economics and Policy Studies Environmental effects of intensification of agriculture: livestock production and regulation
As the distance from the city center increases, rents from livestock production may decline because of higher transportation costs, until rents from livestock and forestry are equal at X.
> * Summary for article: Pandemic shows deep vulnerabilities Third, we need to extinguish for good the ugly inequity of healthy food for the wealthy and junk food for poor people. And these programs can purchase all or nearly all of their food from local farmers using sustainable farming practices, to strengthen city-region links and support struggling farmers.



## Latent Dirichlet Allocation
This method uses topic modelling to find the hidden topics in the documents and then comparing their relative scores with the question.
* **Finding Topics** : The first step is to find the topics in the given set of documents. This is done by first creating a bag of words and then using the gensim library function LdaMulticore and providing the parameters such as number of topics, dictionary etc. The function gives the different topics in the document. For our use we have chosen number of topics as 10
* **Relative Scoring of Topics** : In this step we score the topics based on their similarity with the question keyword and obtain the topic that best describes the question
* **Per Document Topic Distribution** : We find out the per document topic distribution of the dataset which gives us a matrix where each row describes the relative proprtion of each topic in a document.
* **Find the most similar documents** : Now that we have the topic that best describes the question, we select the first ten documents which have highest proportion of that topic in the per document topic distribution. These documents best describe the topic which best describes our question

- LDA consumes a lot of computing time and power but gives good results.
- Scores generated for topics with respect to the question " What do we know about virus genetics, origin and its evolution ?"

```python3
['real-time', 'evolution', 'tracking', 'genome', 'know', 'virus', 'origin', 'genetics']
Score: 0.8712934851646423  Topic: 0.014*"sequence" + 0.013*"sample" + 0.010*"patient" + 0.008*"gene" + 0.008*"result"
Score: 0.014302533119916916  Topic: 0.041*"cell" + 0.015*"gene" + 0.013*"mouse" + 0.010*"protein" + 0.010*"disease"
Score: 0.01430114172399044   Topic: 0.024*"cell" + 0.014*"antibody" + 0.012*"protein" + 0.009*"1" + 0.009*"mouse"
Score: 0.01430098619312048   Topic: 0.017*"patient" + 0.016*"disease" + 0.014*"health" + 0.013*"case" + 0.009*"outbreak"
Score: 0.014300837181508541  Topic: 0.042*"protein" + 0.021*"cell" + 0.011*"rna" + 0.009*"viral" + 0.008*"sequence"
Score: 0.014300500974059105  Topic: 0.020*"cell" + 0.013*"patient" + 0.012*"influenza" + 0.012*"calf" + 0.010*"group"
Score: 0.01430040318518877   Topic: 0.010*"data" + 0.009*"cell" + 0.009*"case" + 0.008*"bat" + 0.008*"model"
Score: 0.014300144277513027  Topic: 0.019*"patient" + 0.017*"disease" + 0.011*"vaccine" + 0.010*"clinical" + 0.009*"respiratory"
Score: 0.014300143346190453  Topic: 0.014*"cell" + 0.011*"protein" + 0.010*"sample" + 0.010*"patient" + 0.009*"viral"
Score: 0.014299794100224972  Topic: 0.051*"cell" + 0.016*"viral" + 0.013*"protein" + 0.010*"activity" + 0.010*"response"
- The per document topic scores show the following distribution for four documents
[[(1, 0.2968149), (4, 0.5133625), (7, 0.102317214), (9, 0.08552837)], 
 [(0, 0.8750857), (2, 0.11611986)],
 [(6, 0.99899405)], [(3, 0.20245415), (9, 0.79017067)],
 [(0, 0.71342075), (7, 0.28082848)], [(3, 0.983916)]]
 ```


## Word as Vector
> **Idea** : If we represent words as vectors then we can find similarity of words using vector distance metric.

In this method we convert words to vectors in *n*-dimensional space. This allows us to find simiarity and difference between words mathematically based on meaning of words.
For this project we will make use of GloVe vectors mapping to convert words to vectors.

### GloVe Vectors
> [GloVe Vectors website](https://nlp.stanford.edu/projects/glove/)

GloVe stands for Global Vectors for Word Representation. This is an algorithm designed by Stanford University's team to find word similarities. It was trained on Wikipedia articles which consist of nearly 6 billion words.

The algorithm uses word-word co-occurrences to find similarity index of 2 words.

Let's take an example of pairs of words : (*dog*, *puppy*), (*dog*, *chair*)
* There can be many articles found which have *dog* and *puppy* coxisting. Hence *dog* and *puppy* will be relatively close to each other in their respective vector representations.
* On the other hand, *dog* and *chair* will not be coexisting in many articles. So, *dog* and *chair* will be far from each other.

If we check the cosine similarity between (*dog*, *puppy*) and (*dog*, *chair*) then,
* cosine_sim (*dog*, *puppy*) = 0.7236376216894685
* cosine_sim (*dog*, *chair*) = 0.26111826072490374

we can observe that *dog* and *puppy* are very close to each other as compared to *dog* and *monkey*.

### Document to Vector
We use a multistep sequential process to convert document to vector.
> #### Steps to make Document Vector
1. Tokenize and Lemmatize filtered body text and get list of keywords.
2. Remove stopwords.
3. Filter keywords with atleast 2 ocurences in document.
4. Convert all selected keywords to vectors.
5. Take weighted mean of these vectors over document vocabulary to get vector for document.

### Question to Vector
We convert question to vectors as well using similar process.
> #### Steps to make Query Vector
1. Tokenize and Lemmatize the question and get list of keywords
2. Remove stopwords
3. Convert all selected keywords to vectors.
4. Take weighted mean of these vectors over question vocabulary to get vector for question.


## Cosine Similarity
> **Idea** : Use Cosine Similarity to find relevance of topic with question.

We use document vectors and query vector to find similarity of document with question.

For this we can use Cosine Similarity as a metric of closeness.
<center>
<img src="https://render.githubusercontent.com/render/math?math=$CosineSimilarity(\vec{u}, \vec{v}) = \frac{\vec{u}.\vec{v}}{||\vec{u}||*||\vec{v}||}$" >
</center>
<br>


If keywords of question and document do not match then we will get vectors which are far apart. So, they will have CosineSimilarity index <img src="https://render.githubusercontent.com/render/math?math=$\approx 0$" >

On the other hand, if keywords of question and document match then we will get vectors which are close. So, they will have CosineSimilarity index <img src="https://render.githubusercontent.com/render/math?math=$\approx 1$" >

Using this approach we can find similarity index *(largest to smallest)* between question and document based on their vectors.

Then, we can rank each document based on the similarity score and show top *k* documents with highest cosine similarity.


## KNN Approach
> **Idea** : Using Euclidean distance to find *k* nearest neighbours of query vector.

We use document vectors and query vector to find closeness of documents with question.

For this we can use Euclidean Distance as a metric of distance.

<center>
<img src="https://render.githubusercontent.com/render/math?math=$EuclideanDistance(\vec{u}, \vec{v}) = ||(\vec{u} - \vec{v})||_2^2$" >
</center>
<br>


If we have document and query vector far apart, their euclidean distance will be big.

If we have document and query vector close to each other their euclidean distance will be small.

Using this approach we can rank the documents based on distance *(smallest to largest)* between question and document vectors.

Then, we can show top *k* documents.


## 1D CNN
*1D Convolutional Neural Network works well for text classification. So, use that idea here to get document embedding using output of it's Average Pooling Layer. Output of Average Pooling layer is low dimensional feature representation of document for classifying it's cluster. Cluster will have similar document, so in that way it can act as document embedding when we are searching for a particular article. 1D CNN should work better than plain averaging of word vectors of a document because it will consider all possible phrases of filter size with given stride parameter.*  
 - Load data into DataFrame.
 - Work on a slice of DataFrame for the experiment.
 - Clean the body text.
 - Filter articles which are in English and remove those which are
   likely noise.
 - Load the pre-trained GloVe vectors.
 - Apply K-Means Clustering to get labels.
 - Learn a simple 1D Convolutional Network to predict cluster.
 - Pick the output AveragePooling Layer of 1D Convolutional Neural
   Network as embedding for the document.
 - Get the norm of each document embedding and store that in a vector.
 - Get the keywords from the question after tokenizing the question.
 - Get the keywords embedding using AveragePooling Layer output of 1D
   Convolutional Neural Network.
 - Get the norm of keywords embedding.
 - Compute the cosine similarity using matrix vector multiplication and
   norms of document embeddings and keyword embeddings.
 - Get the top 5 articles that have the highest cosine similarity with
   the keywords of the question.


## Text Summarisation
> **Idea** : We rank the sentences according to a score and get top 8 sentences as our summary.
- First step is to get a list of top ranked documents.
- Then we need to tokenize sentences to get a list of sentences from original document.
- From formatted documents we generate a list of all words, removing stopwords.
- Now we measure weighted frequency of all the words after removing stopwords.
- We use these weights for each words to calculate score for a sentence.
- If the sentence has that word we add the weighted frequency to the score of that sentence.
- Each sentence will have some score based of the method above.
- Now we just find the top 8 sentences with high scores, these sentences will form our final summary.
> **Example Run**

Article:
"Arctic animals on land include small plant-eaters like ground squirrels, hares, lemmings, and voles; large plant-eaters like moose, caribou/reindeer, and musk ox; and meat-eaters like weasels, wolverine, wolf, fox, bear, and birds of prey. Climate-related changes are likely to cause cascading impacts involving many species of plants and animals. Compared to ecosystems in warmer regions, arctic systems generally have fewer species filling similar roles. Thus when arctic species are displaced, there can be important implications for species that depend upon them. For example, mosses and lichens are particularly vulnerable to warming. Because these plants form the base of important food chains, providing primary winter food sources for reindeer/caribou and other species, their decline will have far-reaching impacts throughout the ecosystem. A decline in reindeer and caribou populations will affect species that hunt them (including wolves, wolverines, and people) as well as species that scavenge on them (such as arctic foxes and various birds)....
> **Weighted Word Frequencies**
 
- species 1.0
- populations 0.7777777777777778
- lemmings 0.6666666666666666
- arctic 0.6666666666666666
- decline 0.6666666666666666
- animals 0.5555555555555556
- snow 0.5555555555555556
- ground 0.4444444444444444
- voles 0.4444444444444444
- plants 0.4444444444444444
- predators 0.4444444444444444
- land 0.3333333333333333
- birds 0.3333333333333333
- prey 0.3333333333333333
- impacts 0.3333333333333333
- particularly 0.3333333333333333
- food 0.3333333333333333
- winter 0.3333333333333333
- ice 0.3333333333333333
- result 0.3333333333333333
- musk 0.2222222222222222
- ox 0.2222222222222222
- weasels 0.2222222222222222
- fox 0.2222222222222222
...

We can see the words which are most relevant to the document appear most frequently, so they have more weight than others which helps us find the most important sentences in the document.

> **Sentence Scores**

- Ice crust formation resulting from freeze-thaw events affects most arctic land animals by encapsulating their food plants in ice, severely limiting forage availability and sometimes killing the plants. [5.333333333333332]
- Thus, a decline in lemmings can also indirectly result in a decline in populations of other prey species such as waders and other birds. [5.222222222222221]
- A decline in lemming populations would be very likely to result in an even stronger decline in populations of these specialist predators. [4.555555555555555]
- Because these plants form the base of important food chains, providing primary winter food sources for reindeer/caribou and other species, their decline will have far-reaching impacts throughout the ecosystem. [4.5555555555555545]
- More generalist predators, such as the arctic fox, switch to other prey species when lemming populations are low. [4.0]
- In winter, lemmings and voles live and forage in the space between the frozen ground of the tundra and the snow, almost never appearing on the surface. [3.666666666666667]
- Thus when arctic species are displaced, there can be important implications for species that depend upon them. [3.3333333333333335]
- Climate-related changes are likely to cause cascading impacts involving many species of plants and animals. [3.333333333333333]
- Compared to ecosystems in warmer regions, arctic systems generally have fewer species filling similar roles. [2.666666666666667]
...

>**Summary Generated**

Ice crust formation resulting from freeze-thaw events affects most arctic land animals by encapsulating their food plants in ice, severely limiting forage availability and sometimes killing the plants. Thus, a decline in lemmings can also indirectly result in a decline in populations of other prey species such as waders and other birds. A decline in lemming populations would be very likely to result in an even stronger decline in populations of these specialist predators. Because these plants form the base of important food chains, providing primary winter food sources for reindeer/caribou and other species, their decline will have far-reaching impacts throughout the ecosystem. More generalist predators, such as the arctic fox, switch to other prey species when lemming populations are low. In winter, lemmings and voles live and forage in the space between the frozen ground of the tundra and the snow, almost never appearing on the surface. Thus when arctic species are displaced, there can be important implications for species that depend upon them. Climate-related changes are likely to cause cascading impacts involving many species of plants and animals.


## Analysis
We analysed our models on a small subset of the dataset that consists of around 10,000 papers due to limited time and computing resources, as most of these methods consume a lot of time and computational power.

### TF-IDF
#### Pros
1. Uses better techniques to remove stop words. It removes the words which are common to all the documents as they are not going to play any role in deciding the document similarities rather than just fixing the stopwords first and rmoving them.
2. Estimates importance of a word to a document.

#### Cons
1. As number of documents increase and vocabulary increase the computation complexity of the TF-IDF matrix becomes very high.
2. The matrix is usually sparse and there is also noise involved.
3. This method does not take into account the statistical view that there is distribution of words in the corpus. 


### Word to Vectors (Cosine Similarity & KNN Approach)
#### Pros
1. Find topic relevance based on meaning of words not number of occurences.

#### Cons
1. Computationally heavy as it needs to calculate norm with each document.
2. Takes time when number of documents gets larger. (currently >100k).
3. Takes time to calculate document vectors as GloVe only has vectors for words. Hence need to take average over all words.

Both TF-IDF and word vectors approach use cosine similarity to rank documents but word vectors is better than TF-IDF.
This is because of TF-IDF only takes only number of occurences of a word into account to give TF-IDF score. On the other hand, word vectors uses meaning of words as vectors.

Word vectors method is less computationally expensive than TF-IDF because matrix size in TF-IDF is <img src="https://render.githubusercontent.com/render/math?math=$D * V$" > but in word vectors it is <img src="https://render.githubusercontent.com/render/math?math=$D * n$" > where <img src="https://render.githubusercontent.com/render/math?math=$D$" > is number of documents, <img src="https://render.githubusercontent.com/render/math?math=$V$" > is size of vocabulary and <img src="https://render.githubusercontent.com/render/math?math=$n$" > is a decidable constant. 
In COVID-19 dataset, <img src="https://render.githubusercontent.com/render/math?math=$D \approx 100,000$" >, <img src="https://render.githubusercontent.com/render/math?math=$V \approx 300,000$" >, <img src="https://render.githubusercontent.com/render/math?math=$n = 100$" >. So, as <img src="https://render.githubusercontent.com/render/math?math=$n \ll V$" >, time taken to run word vectors approach is much less than TF-IDF.

### 1D CNN
#### Pros
1. Considers word vectors for every *h* word phrase, where *h* is filter size. So, it captures more detail than plain word averaging.
2. With multiple *channels* (filter sizes), it can capture very good details of sentences in document.
3. Less computationally expensive as compared to  *LSTM* and *RNN*.

#### Cons
1. Does not consider word order for very long sentences.
2. Not very linguistic.



### Sentence Transformers (BERT)
#### Pros
1. Find topic relevance semantically.
2. Based on state of the art BERT.

#### Cons
1. Computationally heavy as it needs to calculate norm with each document.
2. Takes a lot of time when number of documents gets larger.
3. Only first 768 tokens are used for every sentence.
4. Not multilingual, works only for English sentences.

### LDA
- There are many benefits of using LDA, it is a soft grouping method that is any document can belong to any number of topics and hence the relationships between words and documents is more fairly captured as compared to other methods. It will take into account synonymy, polysemy and contexts.
- By changing parameters to the LDA function, we can observe the changes in the topics that are generated and arrive at the parameters that are best fit for our model. We tested the dataset on number of topics, where we found that as we increase the number of topics, we arrive at a better description for our dataset but the overlaps between the topics increase with the increasing number of topics.
- Decreasing the number of dictionary words in the corpus dramatically decreases the computational time of the model.


#### Pros
1. Finds topic relationships between different documents
2. LDA is a probabilistic model capable of expressing uncertainty about the placement of topics across texts and the assignment of words to topics
3. LDA is good in identifying coherent topics.
#### Cons
1. Computationally expensive
2. Cannot be used for hard clustering documents into topics


## Experimental Results Comparison
Highest ranked Research Papers for a question, "What do we know about virus genetics, origin, and evolution? What do we know about Real-time tracking of whole genomes?" using different methods were found to be:  
- GloVe Averaging:  
  - Quasispecies and the implications for virus persistence and escape  
- KNN:  
  - Functional genomics as a tool in virus research  
- LDA:  
  - Virus survey in populations of two subspecies of bent-winged bats (Miniopterus orianae bassanii and oceanensis) in south-eastern Australia reveals a high prevalence of diverse herpesviruses
- Transformers:  
  - Thermal-induced unfolding-refolding of a nucleocapsid COVN protein  
- 1D-CNN:  
  - Sequencing the whole genome of infected human cells obtained from diseased patients

## Related Work 
- Kim, Convolutional Neural Networks for Sentence Classification, 2014  
> A series of experiments with convolutional neural networks (CNN) trained on top of pre-trained word vectors for sentence-level classification tasks. We show that a simple CNN with little hyperparameter tuning and static vectors achieves excellent results on multiple benchmarks  

- MaksimEkin Notebook on COVID-19 Literature Clustering  
> Used TF-IDF vectorizer and PCA to find clusters of documents using K-Means Clustering  

- Mohit Iyer et al., Deep Unordered Composition Rivals Syntactic Methods for Text Classification, 2015  
> This model, the deep averaging network (DAN), works in three simple steps: 
> - take the vector average of the embeddings associated with an input sequence of tokens  
> - pass that average through one or more feedforward layers  
> - perform (linear) classification on the final layer’s representation  


## Conclusion
Different methods that were taught in the course were applied and on careful analysis, pros and cons of each method were studied upon. Some methods were principle techniques while some were statistical based methods. All of these were helpful in retrieval of information from the raw corpus and give us the relevant answers to the corresponding questions. In terms of comparison we can't state that a specific model is the solution approach to each problem. Every method has it's own merits which are unique to the problem. 

## References
- Mohit Iyer et al., Deep Unordered Composition Rivals Syntactic Methods for Text Classification, 2015  
- Kim, Convolutional Neural Networks for Sentence Classification, 2014  
- Jeffrey Pennington et al., GloVe: Global Vectors for Word Representation, 2014  
- Representations for Language: From Word Embeddings to Sentence Meanings, Christopher Manning, Stanford University, 2017  
- https://www.kaggle.com/maksimeren/covid-19-literature-clustering  
- https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool  
- https://towardsdatascience.com/comparing-text-summarization-techniques-d1e2e465584e
