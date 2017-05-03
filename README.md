GRAM-CNN
===================
GRAM-CNN is a novel end-to-end approach for biomedical NER tasks. 
To automatically label a word, this method uses the local information around the word. Therefore, the GRAM-CNN method doesn't require any specific knowledge or feature engineering and can be theoretically applied to all existing NER problems. \\
The GRAM-CNN approach was evaluated on three well-known biomedical datasets containing different BioNER entities. It obtained an F1-score of 87.38\% on the Biocreative II dataset, 86.65\% on the NCBI dataset, and 72.57\% on the JNLPBA dataset. Those results put GRAM-CNN in the lead of the biological NER methods.

Pre-trained embedding are from: <br>
https://github.com/cambridgeltl/BioNLP-2016 <br>
Some code (loader.py and utils.py) are adopted from: <br>
https://github.com/glample/tagger <br>
https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/TDNN.py <br>
The examples have to be run from the src repository. <br>
Source code for the paper: 


----------
Requirements:
==================

 - Tensorflow  1.0.0 : pip install tensorflow
 - gensim 0.13.2 : pip install gensim==0.13.2
 - numpy : pip install numpy
 - python2.7
 - pre-trained embedding: download from https://drive.google.com/open?id=0BzMCqpcgEJgiUWs0ZnU0NlFTam8 and put it into embeddings folder
--------

Datasets (in dataset folder):
==================
 - Biocreative II (http://biocreative.sourceforge.net/biocreative_2_dataset.html)
 - NCBI (https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)
 - JNLPBA (http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html) 



Train GRAMCNN example:
=================
~~~~
> python train.py --train ../dataset/NLPBA/train/train.eng --dev ../dataset/NLPBA/train/dev.eng --test ../dataset/NLPBA/test/Genia4EReval1.iob2 --pre_emb ../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin -W 100 -H 1 -D 0.5 --lower 1 -A 0 --tag_scheme iob -P 0 -S 0 -w 200 -K 2,3,4 -k 40,40,40
~~~~
> - This will train a one layer Bi-directional LSTM network with hidden size 100 and drop out ratio 0.5, -P set to 0 means that use LSTM 

~~~~
> python train.py --train dataset/NLPBA/train/train.eng --dev dataset/NLPBA/train/dev.eng --test dataset/NLPBA/test/Genia4EReval1.iob2 --pre_emb embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin -D 0.5 -A 0 -W 675 -w 200 -H 7 --lower 1 -K 2,3,4 -k 40,40,40 -P 1 -S 0 --tag_scheme iob
~~~~
> - Set -p 1, this will train GRAMCNN network, -W and -H has no meaning here, drop out ratio is 0.5

> - Detailed parameters setting are in src/train.py
~~~~
> python train.py --help
~~~~

Infer GRAMCNN example:
======================
> - To test the pre-trained model, just replace train.py with infer.py
> - The result output file is in the evaluation repository. 
~~~~
> python infer.py --train ../dataset/NLPBA/train/train.eng --dev ../dataset/NLPBA/train/dev.eng --test ../dataset/NLPBA/test/Genia4EReval1.iob2 --pre_emb ../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin -W 675 -H 12 -D 0.5 --lower 1 -A 0 --tag_scheme iob -P 1 -S 1 -w 200 -K 2,3,4 -k 40,40,40
~~~~

Example pre-trained model:
=======================
> JNLPBA:
> - use_wordTrue use_charTrue drop_out0.5 hidden_size675 hidden_layer12 lowerTrue allembFalse kernels2, 3, 4 num_kernels40, 40, 40 paddingTrue ptsTrue w_emb200
> - Result image:
![alt text](https://github.com/valdersoul/GRAM-CNN/blob/master/src/JNLPBA_res.png)

