# DBpedia_Img2Ent - Predicting DBpedia Entitity Types From Wikipedia Images

*Information Service Engineering, Prof. Dr. Harald Sack & Dr. Mehwish Alam, FIZ Karlsruhe & AIFB, KIT Karlsruhe, WS 2019/20*

## Tasks
### Dataset Generation:

- Extract the entities from DBpedia which are typed only as owl:Thing and dbo: Agent. Also, use the entities from the SDType dataset [1]
- Extract the images from the respective Wikipedia pages of these entities.
- For missing images in Wikipedia pages, use external links.

### Image Classification:
- Perform Image classification task with the pretrained image vectors from ImageNet. 3. Mapping:
- Establish a mapping between the obtained ImageNet classes for these entities to DBpedia classes using ML algorithm

# Instructions
## Instructions for using Code/get_random_entities.ipynb
### Purpose:
This notebook is the first part of the dataset creation. It is used to retrieve random entities for certain rdf:types. Later, images from Wikipedia will be scraped for these entities to create a training dataset.
In the training process, the images will be the input for the algorithm and the rdf:type is the label. The rdf:types, on which base training examples are selected, are loaded in the first step.

### Instructions:
#### Load rdf:types and entities
In these cells, the rdf:types with top entity counts are collected. From a central list, all types are imported. Via a *SPARQL* query the total number of entities is counted and stored into a table.

#### Get X random resources for top 100 rdf types
From all rdf:types, the top 100 ones with highest entity count are selected. To exclude less useful types, these types are filtered out:
- less than 5% of the entities contain image on wiki page (this was measured in a following step)
- general types like *Agent* or *Image* that do not contribute any classification granularity
The functions *get_random_resources* and *get_more_random_resources* are used to collect a certain amount of random entity resources from DBpedia via *SPARQL*. They are stored in json formatting in a text file.

## Instructions for using Code/get_wiki_images.ipynb
### Purpose:
This notebook creates a dataset that will later be used for training and testing. After the initialization, a resource file, created with *get_random_entities.ipynb* is loaded. Using parallel computing, images of Wikipedia pages are downloaded, processed and uploaded into a google storage bucket. 

### Instructions:
#### Threading Process
To run multiple download processes at once, python threading is used. Each thread runs the function *get_imageurl_threading* , which is handed a rdf:type and its random resources. The function calls *get_one_imageurl*, which does the following:
- Open a Wikipedia url via the resource
- Find the first .jpg image on the page by parsing with *BeautifulSoup* (ignoring images >45 pixels in height)
- Get a standard Wikimedia thumbnail url with *get_thumbnail_url*
The image is downloaded temporarily, converted to RGB and uploaded to a google cloud bucket.
When all threads are finished processing, a .csv table is created to document the image files locations inside the bucket.

#### Ranking approach
This approach is not applied in the paper but implemented in code. It is analogous to the previous approach but instead of getting the first Wikipedia image url, several images are compared. This is done via the tf-idf score.

## Instructions for using EfficientNet.ipynb
### Purpose:
The purpose of this notebook is to train and evaluate EfficientNet models for the task of predicting the rdf:type of DBpedia entities based on images. Furthermore it contains code to denoise the dataset as mentioned in the report. Evaluating a trained CNN with the hierarchy approach is possible too. The last part of the code can be used to predict the rdf:type of individual images with models loaded from checkpoints

### Instructions:
#### Imports and Environment:
Necessary imports to run the code, among others: TensorFlow, EfficientNet. Also contains code cell to set the logging level and connect to a TPU runtime.

#### Training Fully Connected Layer:
Parameters and Functions to train and evaluate the EfficientNet Model.

- Parameters: Define Model and dataset to use. Specify the unique name of the training run (used to specify the location to save checkpoints). Define whether to use pretrained initialization or training from scratch. Define further parameters like number of epoch, augmentation methods, learning rate scheduling. (edits only needed in first two cells)

- Function to get exponential moving average variables. (no edits needed)

- InputFn: Two input functions defining the dataset and parameters (batch size to be used by the estimator. The first is a standard input function used for Training and Evaluation. The second is unshuffled to be able to retrace the filename of a prediction. (no edits needed)

- ModelFn: A Model Funtion defining the model architecture (how the pretrained model is connected to the new classification head) and the training, evaluation, prediction procedure. (no edits needed)

- Run Config and Estimator: Functions to define the Run Config and Estimator. (no edits needed)

- Train: Function for training the defined model. (no edits needed)

- Evaluation: Function to evaluate the trained model. (no edits needed)

#### Clean Datasets:
Code to denoise the dataset by identifying possible mislabeled images. Set the checkpoint path to the weights for prediction and set the difference between the highest confidence and the ground truth confidence as a parameter, when to remove an image. (edits only needed in first cell).

#### Evaluation with Hierarchy Approach
Functions to evaluate trained models with the proposed hierarchy approach.

- Functions to get predictions and ground truth labels for images (no edits needed)

- Functions to store predictions and ground truth labels (no edits needed)

- Create datastructures for metric calculation (no edits needed)

- Functions to calculate metrics: Functions that aggregated confidences based on the class hierarchy and to calculate different metrics: hierarchy approach as mentioned in report, predicting with the class that is one level above in the hierarchy than the actual prediction, predicting with the class that is one level above in the hierarchy than the actual prediction if the actual prediction is below a threshold), always predict on top hierarchy level, use actual prediction (baseline). (no edits needed)

- Functions to store and load metrics (no edits needed)

- Functions to plot metrics (no edits needed)

- Run Evaluation (Create plots for report): Run the previously defined functions to create the graphs used in the report. Exchange checkpoint file paths to generate new plots. 

#### Predict single Image
Functions to predict a single image provided by a url. Provide the url to the wget command in the last cell and run to return the top 5 actual predictions and the top 5 predictions based on the hierarchy approach.
