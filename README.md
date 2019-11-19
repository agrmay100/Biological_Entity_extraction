#######
    --author details--
    date of creation = 2019-04-24
    author = Mayank
    email = mayankagrawal@virtusa.com
    phone = 
    city = BLR
    country = India
    ###please add you details if you are modifying as follows
    ###--modifier1--
    ###date=
    ###name=
    ###email=
    ###phone=
    ###country=
    #######
    
** Flask API run on port no. 3231    
BER model
   
Biological entity recognition (B.E.R) is a subfield of named entity recognition (N E R), whose goal is to mine the textual data and identify concepts of interest in the text by mapping all relevant words and phrases to a set of predefined categories. 
Stanford Named Entity Recognition model is used for the use case which is based on conditional random field sequence model. Stanford NER is run from the command line (i.e., shell or terminal). Current releases of Stanford NER require Java 1.8 or later.

The data used is a brat annotated downloaded from Bio N L P shared task sources.
The downloaded files are then converted into conll format which is used in Stanford NER model.

Stanford NER Java command line code - 
#!java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop data/prop.txt
which is used to create the trained model from given training data in prop.txt file

where prop.txt is a property file with the following description-

# trainFile = data/train_data.txt

# serializeTo = bio-ner-model.ser.gz

# map = word=0,answer=1


# useClassFeature=true

# useWord=true

# useNGrams=true

# noMidNGrams=true

# maxNGramLeng=6

# usePrev=true
# useNext=true

# useSequences=true

# usePrevSequences=true

# maxLeft=1

# useTypeSeqs=true

# useTypeSeqs2=true

# useTypeySequences=true

# wordShape=chris2useLC

# useDisjunctive=true

Using the built model, predictions can be made on test data using following command line
#!java -cp ../code/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ../code/bio-ner-model2.ser.gz -outputFormat tabbedEntities -testFile ../code/data/dev_data.txt > ../code/data/test_data.tsv


To make predictions on sample text, the following command is used
#!java -cp ../code/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ../code/bio-ner-model2.ser.gz -textFile sample.txt -outputFormat tsv

