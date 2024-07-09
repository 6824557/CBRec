# CBRec
This is the pytorch implementation for paper:

**A Causal Way Balancing Multidimensional Attraction effect in POI Recommendation**

Please cite our paper if you use the code.
## Environment Requirement

The code has been tested running under Python 3.8.10. The required packages are as follows:

- pytorch == 1.11.0
- torch-geometric ==  2.1.0
- pandas == 1.5.1
- recbole == 1.1.1
- geopy == 2.4.0
- gensim == 4.3.2
- tqdm == 4.66.1

## Running

Here is the process of running the model with NYC dataset.
### 1. preprocess the dataset

Run dataProcessFinal.py to generate data for model from the dataset (operate on NYC by default). 

~~~
python dataProcessFinal.py
~~~

### 2. run the model

Run the model with the following command.

~~~
python main.py (operate on NYC by default). 
~~~

