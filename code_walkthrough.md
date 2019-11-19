
<h1><center><font size=6><font color='brown'>**Biological Entity Recognition **</font></font></center></h1>


#### Importing Dependencies


```python
import numpy as np
import pandas as pd
import glob
import errno
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import seaborn as sns
```

<br>Imported pandas library for data handling & since it is built on top of numpy scientific computation on pandas dataframe is quite easy. Matplotlib and seaborn libraries were imported for visualizing the dataset. Numpy was imported for performing mathematical functions. Scikit-learn model selection tool train_test_split was imported for generating train and test data. Different scikit-learn metrics such as f1_score, precision_score, recall_score and confusion_matrix were imported for evaluating the model performance.

#### Data

The data was extracted from bionlp shared task 2011 website. The useful data is obtained from Genia event extraction task (GE) which aims at extracting events occurring upon genes or gene products, which are typed as "Protein" without differentiating genes from gene products. Other types of physical entities, e.g. cells, cell components, are not differentiated from each other, and their type is given as "Entity". for more details visit [here](http://2011.bionlp-st.org/home/genia-event-extraction-genia).<br>
The data can be found [here](http://2011.bionlp-st.org/home)


The training data is only used for training and test purpose. The data downloaded is in standoff format. The brat annotation tool was used to extract data for the BioNLP shared task.

You can find the example below for the standoff format.


*Text file*
![Capture.PNG](attachment:Capture.PNG)


*annotation file*
![Capture0.PNG](attachment:Capture0.PNG)


We have defined *convert_data()* function which converts data from standoff format to CoNLL format.

Now, you may wonder why we are converting to CoNLL format and what is this format. Don't you?


    As we are using stanford NER model (which I explain later), the data used can only be in CoNLL format.


what is CoNLL format?

    CoNLL format is tab seperated text format.

#### Conversion function
The annoted file(a*.ann) contain entities and tags for the corresponding text file. We have extracted all the entities from these files and annoted 'o' (Other) for the remaining words in the text file.


Since the conversion function is little longer, so I have removed it from here. But, you can find it in the notebook file (bionlp.ipynb)
<br><br><br>
<font size=3>The function returns a dataframe which contains tokens (entity) in one column and tag(entity type) in other. The dataframe contains data extracted from all the files (.ann and .txt) present in genia folder.</font>


```python
# the dataframe contains training data from GE task
df = convert_data('data')
```


```python
# test_df = convert_data('dev_data')
```


```python
#getting first 100 rows of dataframe
df.head(100)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Token</th>
      <th>Tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Down-regulation</td>
      <td>Negative_regulation</td>
    </tr>
    <tr>
      <th>1</th>
      <td>of</td>
      <td>o</td>
    </tr>
    <tr>
      <th>2</th>
      <td>interferon regulatory factor 4</td>
      <td>Protein</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gene</td>
      <td>o</td>
    </tr>
    <tr>
      <th>4</th>
      <td>expression</td>
      <td>Gene_expression</td>
    </tr>
    <tr>
      <th>5</th>
      <td>in</td>
      <td>o</td>
    </tr>
    <tr>
      <th>6</th>
      <td>leukemic</td>
      <td>o</td>
    </tr>
    <tr>
      <th>7</th>
      <td>cells</td>
      <td>o</td>
    </tr>
    <tr>
      <th>8</th>
      <td>due</td>
      <td>o</td>
    </tr>
    <tr>
      <th>9</th>
      <td>to</td>
      <td>o</td>
    </tr>
    <tr>
      <th>10</th>
      <td>hypermethylation</td>
      <td>o</td>
    </tr>
    <tr>
      <th>11</th>
      <td>of</td>
      <td>o</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CpG</td>
      <td>o</td>
    </tr>
    <tr>
      <th>13</th>
      <td>motifs</td>
      <td>o</td>
    </tr>
    <tr>
      <th>14</th>
      <td>in</td>
      <td>o</td>
    </tr>
    <tr>
      <th>15</th>
      <td>the</td>
      <td>o</td>
    </tr>
    <tr>
      <th>16</th>
      <td>promoter</td>
      <td>o</td>
    </tr>
    <tr>
      <th>17</th>
      <td>region</td>
      <td>o</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Although</td>
      <td>o</td>
    </tr>
    <tr>
      <th>19</th>
      <td>the</td>
      <td>o</td>
    </tr>
    <tr>
      <th>20</th>
      <td>bcr</td>
      <td>Protein</td>
    </tr>
    <tr>
      <th>21</th>
      <td>abl</td>
      <td>Protein</td>
    </tr>
    <tr>
      <th>22</th>
      <td>translocation</td>
      <td>o</td>
    </tr>
    <tr>
      <th>23</th>
      <td>has</td>
      <td>o</td>
    </tr>
    <tr>
      <th>24</th>
      <td>been</td>
      <td>o</td>
    </tr>
    <tr>
      <th>25</th>
      <td>shown</td>
      <td>o</td>
    </tr>
    <tr>
      <th>26</th>
      <td>to</td>
      <td>o</td>
    </tr>
    <tr>
      <th>27</th>
      <td>be</td>
      <td>o</td>
    </tr>
    <tr>
      <th>28</th>
      <td>the</td>
      <td>o</td>
    </tr>
    <tr>
      <th>29</th>
      <td>causative</td>
      <td>o</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>sites</td>
      <td>o</td>
    </tr>
    <tr>
      <th>71</th>
      <td>or</td>
      <td>o</td>
    </tr>
    <tr>
      <th>72</th>
      <td>direct</td>
      <td>o</td>
    </tr>
    <tr>
      <th>73</th>
      <td>deletions/insertions</td>
      <td>o</td>
    </tr>
    <tr>
      <th>74</th>
      <td>of</td>
      <td>o</td>
    </tr>
    <tr>
      <th>75</th>
      <td>genes</td>
      <td>o</td>
    </tr>
    <tr>
      <th>76</th>
      <td>are</td>
      <td>o</td>
    </tr>
    <tr>
      <th>77</th>
      <td>mechanisms</td>
      <td>o</td>
    </tr>
    <tr>
      <th>78</th>
      <td>of</td>
      <td>o</td>
    </tr>
    <tr>
      <th>79</th>
      <td>a</td>
      <td>o</td>
    </tr>
    <tr>
      <th>80</th>
      <td>reversible</td>
      <td>o</td>
    </tr>
    <tr>
      <th>81</th>
      <td>or</td>
      <td>o</td>
    </tr>
    <tr>
      <th>82</th>
      <td>permanent</td>
      <td>o</td>
    </tr>
    <tr>
      <th>83</th>
      <td>silencing</td>
      <td>o</td>
    </tr>
    <tr>
      <th>84</th>
      <td>of</td>
      <td>o</td>
    </tr>
    <tr>
      <th>85</th>
      <td>gene</td>
      <td>o</td>
    </tr>
    <tr>
      <th>86</th>
      <td>expression</td>
      <td>o</td>
    </tr>
    <tr>
      <th>87</th>
      <td>,</td>
      <td>o</td>
    </tr>
    <tr>
      <th>88</th>
      <td>respectively</td>
      <td>o</td>
    </tr>
    <tr>
      <th>89</th>
      <td>.</td>
      <td>o</td>
    </tr>
    <tr>
      <th>90</th>
      <td>Therefore</td>
      <td>o</td>
    </tr>
    <tr>
      <th>91</th>
      <td>IRF-4</td>
      <td>Protein</td>
    </tr>
    <tr>
      <th>92</th>
      <td>promoter</td>
      <td>o</td>
    </tr>
    <tr>
      <th>93</th>
      <td>methylation</td>
      <td>o</td>
    </tr>
    <tr>
      <th>94</th>
      <td>or</td>
      <td>o</td>
    </tr>
    <tr>
      <th>95</th>
      <td>mutation</td>
      <td>o</td>
    </tr>
    <tr>
      <th>96</th>
      <td>may</td>
      <td>o</td>
    </tr>
    <tr>
      <th>97</th>
      <td>be</td>
      <td>o</td>
    </tr>
    <tr>
      <th>98</th>
      <td>involved</td>
      <td>o</td>
    </tr>
    <tr>
      <th>99</th>
      <td>in</td>
      <td>o</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 2 columns</p>
</div>




```python
# print(len(test_df))
print(len(df))
```

    207147
    


```python
#defining new dataframe df, to convert the dataframe into train and test dataframes
df = df.iloc[0:int(len(df)*0.2), :]
# test_df = test_df.iloc[0:int(len(test_df)*0.1), :]
```


```python
#using sklearn, training data is splitted into train and test data.
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)
```

<font size = 3>Since we are getting so many entity types from the data, we have used *tag_select()* function to extract the most important entity types which are given below.</font>

- Protien
- Gene expression
- Positive regulation
- negative regulation


```python
#function to get the important entity types.
# stop_ = ['(', ')', '.', ':', ';', ',', '']
def tag_select(df):
# #     for i in range(len(stop_)):
# #         df = df[df['Token'] != stop_[i]]
    
#     #getting entities for top five tags
    tag = (df.Tag.value_counts())[6:]
    for i in range(len(tag.keys())):
        df = df[~df['Tag'].str.contains(tag.keys()[i])]
    
    return df
```


```python
#calling the function to get the dataframes of important entity types
train_df1 = tag_select(train_df)
test_df1 = tag_select(test_df)
# # dev_df1 = data_clean(dev_df)
```


```python
#printing the tags (entity types) for training data.
tags = train_df1.Tag.value_counts()
print('Value counts of Tags in the training data {}' .format(tags))
```

    Value counts of Tags in the training data o                      23835
    Protein                 1917
                             811
    Positive_regulation      361
    Gene_expression          338
    Negative_regulation      177
    Name: Tag, dtype: int64
    


```python
#printing the tags (entity types) for test data.
tags = test_df1.Tag.value_counts()
print('Value counts of Tags in the test data {}' .format(tags))
```

    Value counts of Tags in the test data o                      11774
    Protein                  923
                             396
    Positive_regulation      203
    Gene_expression          158
    Negative_regulation       85
    Name: Tag, dtype: int64
    

**Saving dataframes in tab seperated text format**


```python
train_df1.to_csv('./train_data.txt',sep='\t',index=False)
# dev_df1.iloc[0:int(len(dev_df1)*0.5), :].to_csv('./data/dev_data.txt',sep='\t',index=False)
test_df1.to_csv('./test_data.txt',sep='\t',index=False)
```

### The stanford NER model

Stanford NER is also known as CRFClassifier(general implementation of linear chain Conditional Random Field sequence models). As Stanford NER is a Java implementation of a Named Entity Recognizer, **Java v1.8+** need to be installed. This package provides a high-performance machine learning based named entity recognition system, including facilities to train models from supervised training data.



for more details visit [here](https://nlp.stanford.edu/software/CRF-NER.html#Download)


<font size = 3>Run [this](#refer) java command to generate the model file for the already generated training file and properties defined in prop.txt which contains the below content. We are using 2 files in below java commands, one is stanford ner jar file which can be downloaded from stanford ner site, and other is model file which is generated using [this](#refer) command</font>

One may encounter issue in running java commands in jupyter notebook, in that case use command prompt or any other terminal.


```python
# trainFile = data/train_data.txt
# serializeTo = bio-ner-crf-model.ser.gz
# map = word=0,answer=1

# saveFeatureIndexToDisk = true
# printFeatures=true
# featureDiffThresh=0.05
# useClassFeature=true
# useSequences=true
# useWord=true
# useNGrams=true
# noMidNGrams=true
# maxNGramLeng=10
# usePrev=true
# useNext=true
# maxLeft=1
# useTypeSeqs=true
# useTypeSeqs2=true
# useTypeySequences=true
# wordShape=chris2useLC
# useDisjunctive=true
```

<a id="refer"></a>


```python
#Run the java command on the 4th line of this cell to generate a training model
#!java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -prop prop.txt
```


```python
# import os
# myCmd = 'java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier bio-ner-crf-model.ser.gz -outputFormat tabbedEntities -testFile test_data.txt > test_data_pred.tsv'
# os.system(myCmd)
```

<font size= 3>Below command results in prediction on test data using the classifier created in above code, generating output file (test_data.tsv) with 3rd column as predicted result.</font>


```python
!java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier bio-ner-crf-model.ser.gz -outputFormat tabbedEntities -testFile test_data.txt > test_data_pred.tsv
```


```python
#generating the confusion matrix for the test data
df = pd.read_csv('./data/test_data.tsv', sep = '\t')
test_actual = df.iloc[:, 1].ravel()
test_predict = df.iloc[:, 2].ravel()
labels = pd.Series(test_predict).value_counts().keys()

cm = confusion_matrix(test_actual, test_predict, labels=labels)

corrmat = cm
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(corrmat, square=True, annot=True, fmt='.2f', cmap = "summer", xticklabels=labels, yticklabels=labels);
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```


![png](output_26_0.png)



```python
#print f1 score with different average scores
print(f1_score(test_actual, test_predict, labels = labels, average  = 'micro'))
```

    0.9684242562580841
    


```python
print(f1_score(test_actual, test_predict, labels = labels, average  = 'macro'))
```

    0.7190834194034782
    


```python
print(f1_score(test_actual, test_predict, labels = labels, average  = 'weighted'))
```

    0.9666164021714315
    

#### Calculating F1 score, precision score, recall for the test data for individual labels


```python
#print f1 score for individual labels
x = labels
y = f1_score(test_actual, test_predict, labels = labels, average  = None)
pd.DataFrame(list(zip(x, y)), columns = ['Entity', 'f1 score'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>f1 score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>o</td>
      <td>0.982813</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Protein</td>
      <td>0.949495</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gene_expression</td>
      <td>0.783626</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Positive_regulation</td>
      <td>0.488506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Negative_regulation</td>
      <td>0.390977</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = precision_score(test_actual, test_predict, labels = labels, average  = None)
pd.DataFrame(list(zip(x, y)), columns = ['Entity', 'precision'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>o</td>
      <td>0.977324</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Protein</td>
      <td>0.984866</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gene_expression</td>
      <td>0.728261</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Positive_regulation</td>
      <td>0.586207</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Negative_regulation</td>
      <td>0.541667</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = recall_score(test_actual, test_predict, labels = labels, average  = None)
pd.DataFrame(list(zip(x, y)), columns = ['Entity', 'recall'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Entity</th>
      <th>recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>o</td>
      <td>0.988364</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Protein</td>
      <td>0.916576</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gene_expression</td>
      <td>0.848101</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Positive_regulation</td>
      <td>0.418719</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Negative_regulation</td>
      <td>0.305882</td>
    </tr>
  </tbody>
</table>
</div>




```python
#tagging a given text using the built model.
text = '''BMP-6 inhibits growth of mature human B cells; induction of Smad phosphorylation and upregulation of Id1 
Background
Bone morphogenetic proteins (BMPs) belong to the TGF-beta superfamily and are secreted proteins with pleiotropic roles in many different cell types. A potential role of BMP-6 in the immune system has been implied by various studies of malignant and rheumatoid diseases. In the present study, we explored the role of BMP-6 in normal human peripheral blood B cells. 
Results
The B cells were found to express BMP type I and type II receptors and BMP-6 rapidly induced phosphorylation of Smad1/5/8. Furthermore, Smad-phosphorylation was followed by upregulation of Id1 mRNA and Id1 protein, whereas Id2 and Id3 expression was not affected. Furthermore, we found that BMP-6 had an antiproliferative effect both in naive (CD19+CD27-) and memory B cells (CD19+CD27+) stimulated with anti-IgM alone or the combined action of anti-IgM and CD40L. Additionally, BMP-6 induced cell death in activated memory B cells. Importantly, the antiproliferative effect of BMP-6 in B-cells was completely neutralized by the natural antagonist, noggin. Furthermore, B cells were demonstrated to upregulate BMP-6 mRNA upon stimulation with anti-IgM. 
Conclusion
In mature human B cells, BMP-6 inhibited cell growth, and rapidly induced phosphorylation of Smad1/5/8 followed by an upregulation of Id1.
'''
```


```python
# saving text in a sample.txt file as this will be used later to generate the tags.
file = open('sample.txt', 'w') 
file.write(text)  
file.close()
```


```python
#run this java command to generate the prediction of entity types on the given saved sample.txt
!java -cp stanford-ner.jar edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier bio-ner-crf-model.ser.gz -textFile sample.txt -outputFormat tsv
```
