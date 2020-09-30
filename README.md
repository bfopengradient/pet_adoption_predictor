pet_adoption_predictor:
 
Repo contains several data files(.csv and .npy), two .py files and two jupyter notebooks. 

The first notebook(pet_adoption_accuracy_github) is used for illustrative purposes to call in the classes from the two .py files and then run the algorithm to predict pet adoption rates. It will also produce a copy of each of the data files described in the second notebook. Be warned the embedding matrix output file is almost 850MB.

The second notebook(pet_adoption_datasets) was used to read in the various .csv and .npy data files so the reader can see quickly what the top of each dataset looks like. This notebook explains what each of data files do in the analysis. 

I cited the work of the data scientists in the notebook whose dataset and feature set I decided to work with. Their original work made my life easier in terms of creating my feature set and predictor matrix. The structure of the algorithm was similar to one I had used for a natural language processing task to classify sentences. 

The objective here was to see how this model would behave on this classification task after I had addressed sparsity in the predictor matrix.

BF
  
