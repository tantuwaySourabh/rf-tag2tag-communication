# rf-tag2tag-communication

## Overview:
This project is based on the research paper by Dr. Zygmunt J. Haas, which deals with the idea of passive tags communicating with each other in a large-scale network. While the research paper provides an extensive study about how T2T communication results in significantly larger capacity than centralized RFID reader-based system, this project is just an effort to practically implement the basic routing algorithm mentioned as “Algorithm 1: Routing Protocol for Scenario A” in the paper. Kindly go through the paper to better understand the project.


Research paper link: (https://ieeexplore.ieee.org/document/9146888)

## Tools and Technologies Used:
In this project, I used Python language and some Machine Learning libraries for the implementation of the algorithm. Entire project can be run using the python notebook file (i.e. Project.ipynb). In my case I used Jupyter Notebook server locally and ran the .ipynb file on that.

## Assumptions:
* For making things simpler, in this project we assumed that each tag cannot have more than one message to transmit.
* Weights of all messages will be same.

## Running Instructions:
Python notebook file should be executed in sequence, from starting to end.
Some dependencies might be required to run the code successfully. They can be installed as you run the code.
Before executing the python file make sure if the csv files are in place:
* dist_table.csv should be in the same folder as Project.ipynb
* folder named ‘dataset’ should also be in the same folder as Project.ipynb file.

This folder contains dataset.csv used for training and testing the ML model.
dist_table.csv contains specific arrangement of tags for which ML model will be trained. dataset.csv contains the 80k rows of instances and solutions. These rows correspond to the tag
arrangement stored in dist_table.csv.
Now change the path of dataset.csv in the code and execute all the cells. The model will be
trained, and the accuracy of model will come nearly 70%.
New arrangement of tags can be generated and stored in dist_table.csv and new dataset can be generated using generate_data(num_of_rows) function and saved in dataset.csv. If done so, ML model will be generated for this new data and accuracy will be according to the amount of data generated.

## Results:

I used DecisionTreeClassifier from sklearn library. MultiOutputClassifier class in sklearn helped dealing with multi label classification and supported vector type prediction in our case. I divided all 80k data rows in training and testing set 80% and 20% respectively. Trained the classifier over training set and tested over testing set to achieve 70% accuracy.


