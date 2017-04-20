GRAM-CNN
===================

Pre-trained embedding are from: <br>
https://github.com/cambridgeltl/BioNLP-2016 <br>
Some code (loader.py and utils.py) are adopted from: <br>
https://github.com/glample/tagger <br>
https://github.com/carpedm20/lstm-char-cnn-tensorflow/blob/master/models/TDNN.py <br>
Source code for the paper: 

----------
Requirements:
==================

 - Tensorflow  1.0.0
 - gensim 0.13.2
 - numpy
 - python2.7

--------

Datasets (in dataset folder):
==================
 - Biocreative II (http://biocreative.sourceforge.net/biocreative_2_dataset.html)
 - NCBI (https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)
 - JNLPBA (http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html) 


Train GRAMCNN example:
=================
> - python train.py --train ../dataset/NLPBA/train/train.eng --dev ../dataset/NLPBA/train/dev.eng --test ../dataset/NLPBA/test/Genia4EReval1.iob2 --pre_emb ../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin -W 100 -H 1 -D 0.5 --lower 1 -A 0 --tag_scheme iob -P 0 -S 0 -w 200 -K 2,3,4 -k 40,40,40 <br>
> - This will train a one layer Bi-directional LSTM network with hidden size 100 and drop out ratio 0.5, -P set to 0 means that use LSTM 

> - python train.py --train dataset/NLPBA/train/train.eng --dev dataset/NLPBA/train/dev.eng --test dataset/NLPBA/test/Genia4EReval1.iob2 --pre_emb embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin -D 0.5 -A 0 -W 675 -w 200 -H 7 --lower 1 -K 2,3,4 -k 40,40,40 -P 1 -S 0 --tag_scheme iob <br>
> - Set -p 1, this will train GRAMCNN network, -W and -H has no meaning here, drop out ratio is 0.5

> Detailed parameters setting are in src/train.py

Infer GRAMCNN example:
======================
> - To test the pre-trained model, just replace train.py with infer.py
> - python infer.py --train ../dataset/NLPBA/train/train.eng --dev ../dataset/NLPBA/train/dev.eng --test ../dataset/NLPBA/test/Genia4EReval1.iob2 --pre_emb ../embeddings/bio_nlp_vec/PubMed-shuffle-win-30.bin -W 100 -H 1 -D 0.5 --lower 1 -A 0 --tag_scheme iob -P 0 -S 0 -w 200 -K 2,3,4 -k 40,40,40 <br>

Example pre-trained model:
=======================
> JNLPBA:
> - use_wordTrue use_charTrue drop_out0.5 hidden_size675 hidden_layer12 lowerTrue allembFalse kernels2, 3, 4 num_kernels40, 40, 40 paddingTrue ptsTrue w_emb200
