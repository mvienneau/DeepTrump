# Deep Trump
Simple recurrent nueral network trained on a dump of Trump's twitter. 

https://twitter.com/deeptrumpai

Currently very unoptimized and an overall mediocre langauge model. Working to improve by adding LSTM and more parameter tuning.


## Pytorch Model
*Under Development*
Under the `pytorch/` folder you will find the pytorch model, `torch_rnn.py`. Currently, this is a first pass at writing the RNN in an actual deep learning library. 


Issues:
- ~Small Training Data~
- Doesn't train on "Tweets", but rather the whole corpus of tweets as one
	- Explore different training splits
- Padding for the tweets? 
