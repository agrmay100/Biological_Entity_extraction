 
** Flask API run on port no. 3231    
## BER model
<br><br>
Biological entity recognition (B.E.R) is a subfield of named entity recognition (N E R), whose goal is to mine the textual data and identify concepts of interest in the text by mapping all relevant words and phrases to a set of predefined categories. 
Stanford Named Entity Recognition model is used for the use case which is based on conditional random field sequence model. Stanford NER is run from the command line (i.e., shell or terminal). Current releases of Stanford NER require Java 1.8 or later.
<br><br>
The data used is a brat annotated downloaded from Bio N L P shared task sources.<br>
The downloaded files are then converted into conll format which is used in Stanford NER model.
<br><br>
Stanford NER Java command line code - 
!java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop data/prop.txt
which is used to create the trained model from given training data in prop.txt file
<br><br>
where <b>prop.txt</b> is a property file with the following description-
<br>
trainFile = data/train_data.txt<br>
serializeTo = bio-ner-model.ser.gz<br>
map = word=0,answer=1<br>
useClassFeature=true<br>
useWord=true<br>
useNGrams=true<br>
noMidNGrams=true<br>
maxNGramLeng=6<br>
usePrev=true<br>
useNext=true<br>
useSequences=true<br>
usePrevSequences=true<br>
maxLeft=1<br>
useTypeSeqs=true<br>
useTypeSeqs2=true<br>
useTypeySequences=true<br>
wordShape=chris2useLC<br>
useDisjunctive=true<br>
<br>
Using the built model, predictions can be made on test data using following command line<br>
!java -cp ../code/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ../code/bio-ner-model2.ser.gz -outputFormat tabbedEntities -testFile ../code/data/dev_data.txt > ../code/data/test_data.tsv
<br><br>

To make predictions on sample text, the following command is used<br>
!java -cp ../code/stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier ../code/bio-ner-model2.ser.gz -textFile sample.txt -outputFormat tsv

