#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Following code generetes random points on a 2D plane. These points will be our Tags.
#In this code we can set how many tags we want and the size of the plane.

import matplotlib.pyplot as plt
import random

# Generate random points
num_points = 20  # Change this value to generate a different number of points

# generating random x and y coordinates of the tags with values between-10 to 10
x_coords = [random.randint(-15, 15) for _ in range(num_points)]
y_coords = [random.randint(-15, 15) for _ in range(num_points)]

# assigning labels to points as P1, P2... and so on
point_labels = [f"P{i+1}" for i in range(num_points)]

# for Figure plotting
fig, ax = plt.subplots()
# Set the x and y limits of the axes
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])

# Plot the points on the axis
# ax.scatter(x_coords, y_coords)
for i in range(num_points):
    ax.scatter(x_coords[i], y_coords[i])
    ax.annotate(point_labels[i], (x_coords[i], y_coords[i]))

# Add gridlines
ax.grid(True)

# Set the title and axis labels
ax.set_title(f"Random Points on Cartesian Plane ({num_points} points)")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")


# Show the plot
# plt.show()


# In[165]:


# Code for calculating distance maxtrix. It will store distance between each pair of tags.

import pandas as pd
from scipy.spatial.distance import pdist, squareform
from IPython.display import display

points = [(x_coords[i], y_coords[i]) for i in range(num_points)]
point_labels = [f"P{i+1}" for i in range(num_points)]

# Calculate pairwise distances
distances = pdist(points)

# Convert the distances to a square matrix
dist_matrix = squareform(distances)

# Create a pandas DataFrame with the distances
df = pd.DataFrame(dist_matrix, columns=point_labels, index=point_labels)

# Print the DataFrame
# display(df)


# In[160]:


#just a code to make a dataframe from local file.

# df = pd.read_csv('dist_table.csv')
# default_ind = [i for i in range(num_points)]
# dictionary = dict(zip(default_ind, point_labels))
# #print(dictionary)
# df = df.rename(index=dictionary)
# df = df.drop(df.columns[0], axis=1)
# #display(df2)
# pd.DataFrame(dist_matrix, columns=point_labels, index=point_labels)


# In[ ]:


#just a code to make a dataframe from local file.

df = pd.read_csv('dist_table.csv')
default_ind = [i for i in range(num_points)]
dictionary = dict(zip(default_ind, point_labels))
#print(dictionary)
df = df.rename(index=dictionary)
df = df.drop(df.columns[0], axis=1)
# display(df)
# pd.DataFrame(dist_matrix, columns=point_labels, index=point_labels)


# In[149]:


#code for saving the distance matrix.
#df.to_csv('dist_table.csv');


# In[166]:



# give a value of radius till what a tag can communicate.
radius = 8.00

# defining neighbor table according to the defined radius.
# value 1 in the table means there is an transmission as well as interference link is present between that pair.
# a random large value of 100000 for absence of links between pairs. 0 means self link.

neighbor_table = df
neighbor_table = neighbor_table.applymap(lambda x: 100000 if x > radius else 1)

# filling 0 for self links.
for i in range(num_points):
    neighbor_table.at[f"P{i+1}", f"P{i+1}"] = 0
    
neighbor_table


# In[22]:


#code to save the neighbor table
#neighbor_table.to_csv('n_table.csv');


# In[167]:


import random


# defining vector X will which will store which tag is chosen for sending messages.
# X[0] is by default 0 , for making it 1-based indexing.


def select_nodes():

    # Generate a list of 20 random values between 0 and 1
    X = [random.uniform(0, 1) for _ in range(num_points + 1)]

    # Replace values less than 0.4 with 0 and values greater than or equal to 0.4 with 1
    X = [0 if v < 0.4 else 1 for v in X]
    X[0] = 0
    #print(X)
    
    return X


# In[168]:


import numpy as np

#code for selecting random destination for all transmitting nodes.


def select_destinations(X_vec):
    # dest vector to store destination of selected transmitting nodes. 
    dest = [0 for i in range(num_points + 1)]

    # Setting up destination of all transmitting nodes.
    selected_nodes = [i for i, x in enumerate(X_vec) if x == 1]


    for node in selected_nodes:
        neighbors = np.array(neighbor_table.loc[f"P{node}"])
        neighbor_indices = np.where(neighbors == 1)[0]
        if len(neighbor_indices) > 0:
            random_neighbor = np.random.choice(neighbor_indices) + 1
            dest[node] = random_neighbor
        else:
            pass

    #print(dest)
    return dest


# In[169]:


# defining vector M which tells which message is selected for transmission for a particular tag. 
# for example M[1] = 5 means 5th message of tag 1 is selected.
# M = [0 for i in range(num_points + 1)]
# M[2] = M[3] = M[6] = M[9] = 1
# M


# In[170]:


# defining a dictionary for storing ci values(as mentioned in Paper) of every message, Key of the dictionary is tag's number.

# C = {}

# for i in range(num_points + 1):
#     key = i
#     values = []
#     C[key] = values


# manually setting the values as of now.
# C[2].append(1)
# C[3].append(1)
# C[6].append(1)
# C[9].append(1)
# C[9].append(0)


#C


# In[61]:


# # function for difining C2 constraints
# def define_c2(x, c):
#     constraint = []
#     n = len(x)
    
#     for i in range(n):
#         str = []
#         if(i == 0):
#             continue
#         if(len(c[i]) == 0):
#             constraint.append(f'x\u0305{i}')
#             continue
        
#         str.append(f'x{i}(')
#         for j in range(len(c[i])):
#             str.append(f'c{i}{j + 1}')
#             for k in range(len(c[i])):
#                 if(k != j):
#                     str.append(f'c\u0305{i}{k + 1}')
            
#             if(j != len(c[i]) - 1):
#                 str.append(f' + ')
        
        
#         str.append(f')')
#         str.append(f'+')
#         str.append(f'x\u0305{i}')
        
#         for j in range(len(c[i])):
#             str.append(f'c\u0305{i}{j + 1}')
            
#         constraint.append(''.join(str))
        
#     return constraint
 

    
    
# # running the function.
# print(define_c2(X, C))


# In[171]:


# defining vector dest which stores destination for each tag.
# Assuming destination is set for all tags using the Routing table.

# dest = [0 for i in range(num_points + 1)]

# dest[2] = 6
# dest[3] = 10
# dest[6] = 3
# dest[9] = 2
# dest


# In[172]:



def define_c3(x, m, dst, neighbor):
    
    # vector to store all constraints
    constraint = []
    
    for i in range(len(x)):
        str = [] # for storing constraint of a single tag
        
        # for every active tags we will see interfering tags in neighbor table and append them to the constraints.
        if(x[i] == 1):
            #str.append(f'g(c{i}{m[i]}, ')
            ind = dst[i]
            str.append(ind)
            for j in range(num_points):
                if(neighbor[f'P{ind}'][f'P{j + 1}'] == 1 and j != i - 1):
                    #str.append(f'x{j + 1} ')
                    #str.append('+ ')
                    str.append(j + 1)
             
            #if(str[-1] == '+ ' ): # just to remove extra + at the end
                #str = str[: -1]
                
            #str.append(f')')
            constraint.append(str)
        else:
            constraint.append('')
            
    return constraint


# running the function
# here we are passing X again for vector m because assumption is that each nodes have single msg only.
#c3 = define_c3(X, X, dest, neighbor_table)
#print(c3)


# In[219]:


import sys
import itertools
def solve_instance(x, d, c):
    
    indexes = [i for i, num in enumerate(x) if num == 1]
    #print(indexes)
    combinations = []
    result = None
    unique_combinations = []
    comb_size = -1

    for size in range(1, len(indexes) + 1):
        combinations = itertools.combinations(indexes, size)
        
        
        for combination in combinations:
            sorted_combination = tuple(sorted(combination))
            if sorted_combination not in unique_combinations:
                unique_combinations.append(sorted_combination)
                
                
                valid_comb = True
                for elem in sorted_combination:
                    intersection = set(c[elem]).intersection(set(sorted_combination))
                    if(len(intersection) > 0): 
                        valid_comb = False
                        break
                
                if(valid_comb):
                    if(comb_size < len(sorted_combination)):
                        result = sorted_combination
                
                    
    return list(result)

    
    
#combinations = solve_instance(X,dest,c3)
#print(combinations)


# In[215]:


# utility function for converting two lists into a dictionary
def make_dictionary(x_vec , d_vec):
    inst = {}
    
    for j in range(1, len(x_vec)):
            if x_vec[j] == 0:
                inst[j] = 0
            else:
                inst[j] = d_vec[j]
                   
    return inst
    


# In[237]:



def generate_data(rows):
    
    data = pd.DataFrame(columns=['Instance', 'Solution'], index=[i for i in range(rows)])
#     df.loc[0, 'X'] = X
#     df.loc[0, 'Y'] = Y
    
    for i in range(rows):
        X = select_nodes()
        print(i, end= " ")
        D = select_destinations(X)
        #print(f" D is {D}")
        C3 = define_c3(X, X, D, neighbor_table)
        #print(C3)
        
        sol = solve_instance(X,D,C3)
        #print(f" sol is {sol}")
        
        #data.at[i, 'Instance'] = make_dictionary(X, D) #for representing input as dictionary
        data.at[i, 'Instance'] = D[1:]
        data.at[i, 'Solution'] = sol
        
        #data.to_csv('dataset.csv');
    
    return data

generate_data(1)    


# In[238]:


data = generate_data(20000)
#pd.set_option('display.max_colwidth', None)
#display(data)

#code to save the dataset
data.to_csv('dataset.csv');


# In[229]:





# In[ ]:




