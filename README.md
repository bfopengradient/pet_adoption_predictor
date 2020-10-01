pet_adoption_predictor:
 
Repo contains several data files(.csv), two .py files and two .ipynb jupyter notebooks. 

The first notebook(pet_adoption_predictor.ipynb) is used for illustrative purposes to call in the classes from the two .py files and then run the algorithm to predict pet adoption rates. It will also produce a copy of each of the data files described in the second notebook. Be warned the embedding matrix output file is almost 850MB. You may want to comment out the code(line 158) that produces the saved embedding file in the pet_adoption.py file.

The second notebook(pet_adoption_datasets.ipynb) was used to read in the various .csv and .npy data files so the reader can see quickly what the top of each dataset looks like. This notebook explains what each of data files do in the analysis. 

I did work with a dataset created by other researchers. I cited their work in both jupyter notebooks. Their original work made my life easier in terms of creating my feature set and predictor matrix. The structure of the algorithm I decided to work with was similar to one I had used for a natural language processing task to classify sentences. It allows for increased dimensionality beyond the size of the feature set contained in the original dataset. I felt this was useful as there was 
no data on the people visiting the animal shelter who may or may not adopt a pet.

The objective here was to see how this model would behave on this classification task after I had addressed sparsity in the predictor matrix.

The target variable is split 58%/42% in favor of the non-adopoted outcomes. Accuracy levels of my model were arpox 77%. f scores were aprox 73%.

BF
  
