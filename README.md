# PA4 :
**Note on ethics: In this task, we will simulate a backend component of a service exchange platform. Service exchange 
is a concept closely related to the notion of shared economy, examples of which include Uber, AirBnB, TaskRabbit etc. 
The existing platforms have been extensively criticised (e.g. [Tom Slee's position](https://pdfs.semanticscholar.org/4100/91be5c14ba4b17dcbd74f7e594f0c22782b5.pdf)) for their business models, 
which clearly need to be changed in order to meet contemporary ethical standards of work. In this task, 
we focus on the technical side of a fictional toy-scale platform counting on future developments in business regulations 
that will make such application ethically acceptable from the point of view of data protection, 
fair trade and all other related issues.**

Assuming a service exchange application whose main backend component is a database storing the information 
about competences that can be exchanged in one-to-one service agreements. A user who wishes to obtain a service inputs a query. 
Our task is to match this query with the most relevant entries in the database and 
return the contacts or profiles of all the registered users who can provide the service in the query.

For example:

| Query | Response | Exclude |
| --- | --- | ---|
| cut hair | hair dresser | gardener|
| | barber | |

In order to do this, we will calculate semantic similarity between the query and all the items in the database. 
Your task is to write a Python script that takes as input a given test set  and outputs for each query similarity 
ranking of all the competences.

We will compare two methods for calculating the similarity rankings:

1. The baseline method: cosine between word vector averages
2. Advanced method: Word mover's distance (WMD)

In particular, our test set consists of 19 queries (pa4_Q.txt) and 16 competences and (pa4_C.txt). 
For each query, you will output two rankings of each competence (one baseline and one WMD).

You will assess the success of the method by counting the number of times that the correct competence 
(given in the test set) is within top 3 ranks returned by your script.

For both methods, we will use pretrained word embeddings available via the **gensim** library 
(function **KeyedVectors**). For the baseline method, look up the embeddings of each word found in the test set, 
find the averages and then find the cosine similarity between the average vaules. For the word mover's distance, 
you will use the function implemented in gensim (try to find out how it works yourself).

To simulate an existing database, you will hard-code the list of competences, while the list of queries should 
be given as a command-line argument while running the script. No other command-line arguments should be allowed.

Submission:

1. A single Python script that takes one input file (pa4_Q.txt) as command-line arguments. **Note:** the list that we provide contains the correct rankings. In reality, the user would not know the correct ranking, but we ignore this point for now.

2. A text file called "results.txt" containing:

   2.1 the output of your script, 19 tables (one for each query) in the following form:
  
   | query | | |
   | --- | --- | ---|
   | ranking | baseline | WMD |
   | competence_1 |
   | competence_2 |
   | competence_3 |
   | competence_4 |
   | competence_5 |
   | competence_6 |
   | competence_7 |
   | competence_8 |
   | competence_9 |
   | competence_10 |
   | competence_11 |
   | competence_12 |
   | competence_13 |
   | competence_14 |
   | competence_15 |
   | competence_16 |

   2.2 a short (one paragraph) interpretation of the results of the comparison

**Deadline:**
11.05.2020 at 15h
