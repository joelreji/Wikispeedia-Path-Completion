#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from urllib.parse import unquote
import networkx as nx
import matplotlib.pyplot as plt
from random import seed
from random import randint
import numpy as np


# ## Read in finished paths data

# In[2]:


colnames =['ipAddress', 'timestamp', 'duration_Secs', 'path', 'rating']
finished_df = pd.read_csv('wikispeedia_paths-and-graph/paths_finished.tsv', sep='\t', skiprows=15, names=colnames, na_values='NULL')
finished_df['path'] = finished_df.apply(lambda row: unquote(row.path), axis=1)
finished_df.head()


# ## Drop unecessary columns & create new column with number of clicks

# In[3]:


finished_df['click_Path'] = finished_df['path'].str.split(';')
finished_df['click_Count'] = finished_df.apply(lambda row: len(row.click_Path), axis=1)
finished_df = finished_df.drop(['ipAddress', 'timestamp', 'rating'], axis=1)
finished_df.head()


# ## Add in binary feature for backclick and number of backclicks

# In[4]:


finished_df['back_Click'] = finished_df.apply(lambda row: '<' in row.click_Path, axis=1)
finished_df.loc[finished_df['back_Click'] == True, 'back_Click'] = 1
finished_df.loc[finished_df['back_Click'] == False, 'back_Click'] = 0
finished_df['back_click_Count'] = finished_df.apply(lambda row: row.click_Path.count('<'), axis=1)
finished_df


# ## Add in a binay feature to indicate that paths were completed

# In[5]:


finished_df['completed_Path'] = 1
finished_df


# ## Read in the unfinished paths data and drop unwanted columns

# In[6]:


colnames_unfinished =['ipAddress', 'timestamp', 'duration_Secs', 'path', 'target', 'type']
unfinished_df = pd.read_csv('wikispeedia_paths-and-graph/paths_unfinished.tsv', sep='\t', skiprows=16, names=colnames_unfinished, na_values='NULL')
unfinished_df = unfinished_df.drop(['ipAddress', 'timestamp', 'type'], axis=1)
unfinished_df.head()


# ## Prepare data to match finished dataframe

# In[7]:


unfinished_df['click_Path'] = unfinished_df['path'].str.split(';')
unfinished_df['click_Count'] = unfinished_df.apply(lambda row: len(row.click_Path), axis=1)
unfinished_df['back_Click'] = unfinished_df.apply(lambda row: '<' in row.click_Path, axis=1)
unfinished_df.loc[unfinished_df['back_Click'] == True, 'back_Click'] = 1
unfinished_df.loc[unfinished_df['back_Click'] == False, 'back_Click'] = 0
unfinished_df['back_click_Count'] = unfinished_df.apply(lambda row: row.click_Path.count('<'), axis=1)
unfinished_df['completed_Path'] = 0
unfinished_df


# ## Balancing class distribution in unfinished

# In[8]:


unfinished_df2 = unfinished_df.sample(n=26443, replace=True, axis=0)
frames = [unfinished_df,unfinished_df2]
unfin_df = pd.concat(frames)
unfin_df


# ## Add in target variable to finished dataframe and merge

# In[9]:


finished_df['target'] = finished_df.apply(lambda row: row.click_Path[-1], axis=1)
frames = [finished_df,unfin_df]
final_df = pd.concat(frames)
final_df['source'] = final_df.apply(lambda row: row.click_Path[0], axis=1)
final_df = final_df[final_df['click_Count'] != 1]
final_df


# ## Add in source variable (potentially can be used for category tagging)

# In[10]:


index = []
for x in range(76193):
    if 'GNU_Free_Documentation_License' in final_df.iloc[x,1]:
        index.append(x)
final_df.drop(final_df.index[index], axis=0, inplace=True)
final_df


# # Read in the links data & add it to a new dictionary

# In[11]:


colnames =['Start_page', 'Linked_page']
links_df = pd.read_csv('wikispeedia_paths-and-graph/links.tsv', sep='\t', skiprows=11, names=colnames)
links_df.head()


# In[12]:


links_dict ={}
for x in links_df.index:
    start_Page = unquote(links_df["Start_page"][x])
    if start_Page in links_dict.keys():
        links_dict[start_Page].append(unquote(links_df["Linked_page"][x]))
    else:
        links_dict[start_Page] = [unquote(links_df["Linked_page"][x])]


# # Read in the categories data & and add it to a dictionary

# In[13]:


colnames=['Page', 'Categories']
categories_df = pd.read_csv('wikispeedia_paths-and-graph/categories.tsv', sep='\t', skiprows=12, names=colnames)
categories_df.head()


# In[14]:


categories_dict = {}
for x in categories_df.index:
    page = unquote(categories_df["Page"][x])
    if page in categories_dict.keys():
        categories_dict[page].append(unquote(categories_df["Categories"][x])[8:].split('.'))
    else:
        categories_dict[page] = [unquote(categories_df["Categories"][x])[8:].split('.')]


# # Create category mapping for the first click

# In[16]:


final_df['first_Click'] = final_df.apply(lambda row: len(row.click_Path) > 1, axis=1)
cats = []

for x in range(91535):
    if (final_df.iloc[x,3] >= 2):
        final_df.iloc[x,9] = final_df.iloc[x,2][1]
        if (final_df.iloc[x,9] in categories_dict.keys()):
            cats.append(categories_dict.get(final_df.iloc[x,9]))
        else:
            cats.append("none")
    else:
        cats.append("none")
final_df['category'] = cats


# ## Save first data frame 

# In[17]:


cat = []
for x in range(91535):
    if (final_df.iloc[x,10] != 'none'):
        cat.append(final_df.iloc[x,10][0][0])
    else:
        cat.append('none')
final_df['first_Category'] = cat

final_df = final_df.drop(['first_Click', 'category'], axis=1)


# ## Create dummy variables

# In[18]:


dummies = pd.get_dummies(final_df['first_Category'])
final_df = pd.concat([final_df, dummies], axis=1)
final_df.index = list(range(len(final_df.index)))
    
final_df.to_csv("df_main.csv", index=False)
print("Done!")
final_df


# ## Save fragmented paths to a list

# In[19]:


# final_df = pd.read_csv('df_main.csv')
df_fragPath = []
for x in range(91535):
    seed(101)
    path_len = len(list(final_df.iat[x, 2]))
    if path_len != 1:
        random_stop = randint(1,path_len-1)
        df_fragPath.append(list(final_df.iat[x, 2])[0:random_stop])
    else:
        df_fragPath.append(list(final_df.iat[x, 2]))


# ***

# # Create a graph object with the category data as metadata

# In[20]:


node_list = []
for key in categories_dict:
    node_list.append((key, {"subtree":categories_dict[key]}))

G = nx.DiGraph()
G.add_nodes_from(node_list)
list(G.nodes(data=True))

edge_list = []
for key in links_dict:
    for link in links_dict[key]:
        edge_list.append((key,link))
edge_list

G.add_edges_from(edge_list)
print(len(list(G.nodes())))
print(len(list(G.edges())))
G.add_edges_from([('Finland', 'Åland'),
 ('Finland', 'Åland'),
 ('Republic_of_Ireland', 'Éire'),
 ('Claude_Monet', 'Édouard_Manet'),
 ('Republic_of_Ireland', 'Éire'),
 ('Claude_Monet', 'Édouard_Manet'),
 ('Ireland', 'Éire'),
 ('Impressionism', 'Édouard_Manet'),
 ('Republic_of_Ireland', 'Éire'),
 ('Republic_of_Ireland', 'Éire'),
 ('Claude_Monet', 'Édouard_Manet')])
print(len(list(G.edges())))


# ### Create Test DF

# In[21]:


test_df = final_df.copy()
test_df.insert(2,'fragmented_Path', df_fragPath)


# In[22]:


for n in range(91535):
    if test_df.iat[n,2] == test_df.iat[n,3] and len(test_df.iat[n,3]) !=1:
        print(n)


# ### Dict. of assigned start/target combos with indices -   (root,target):[indices]

# In[23]:


unique_assignments = {}
for n in range(91535):
    if (unquote(final_df.iat[n,8]), unquote(final_df.iat[n,7])) not in unique_assignments.keys():
        unique_assignments[(unquote(final_df.iat[n,8]), unquote(final_df.iat[n,7]))] = [n]
    else:
        unique_assignments[(unquote(final_df.iat[n,8]), unquote(final_df.iat[n,7]))].append(n)


# ### Find impossible assignments

# In[24]:


possible=[]
impossible=[]
for (root,target) in unique_assignments.keys():
    try:
        nx.dijkstra_path(G, root, target)
        possible.append(unique_assignments[(root,target)])
    except:
        impossible.append(unique_assignments[(root,target)])


# ### Make list of rows to drop and drop them

# In[25]:


impossible_concat = []
for n in impossible:
    for m in n:
        impossible_concat.append(m)
print(len(test_df.index))
print(len(impossible_concat))
test_df.drop(test_df.index[impossible_concat], axis = 0, inplace = True)
print(len(test_df.index))


# # Average out degree of the nodes in the path & add to final dataframe

# In[27]:


avg_deg_out = []
for n in range(91486):
    degs = []
    path = list(test_df.iat[n,2])
    path = [value for value in path if value != '<']
    for m in range(len(path)):
        path[m] = unquote(path[m])
        degs.append(G.out_degree(path[m]))
    
    try:
        avg_deg_out.append(sum(degs) / len(degs))
    except:
        avg_deg_out.append('err')
#         print(n, fragmented_Path)
avg_deg_out
test_df['Avg_outDegree'] = avg_deg_out
assert 'err' not in avg_deg_out
test_df


# # Function to clean up backspaces

# In[28]:


def backspace_cleaner(path):
    totalcount = len([index for index, page in enumerate(path) if page=='<'])
    while totalcount > 0:
        count = 1
        backpress = [index for index, page in enumerate(path) if page=='<']
        to_fix = backpress[-1]
        current = backpress[-1]
        while count > 0:
            check = []
            for n in range(2*count):
                check.append(path[current-(n+1)])
            current -= len(check)
            count = len([index for index, page in enumerate(check) if page=='<'])
        path[to_fix] = path[current]
        totalcount = len([index for index, page in enumerate(path) if page=='<'])
    return(path)


# # Average number of times each link in path is traversed

# In[29]:


avg_link_traversals = []
for n in range(91486):
    path = backspace_cleaner(list(test_df.iat[n,2]))
    
    links = []
    for m in range(len(path)):
        path[m] = unquote(path[m])
    for m in range(len(path)-1):
        links.append([path[m],path[m+1]])
   
    unique_links = []
    for p in links:
        if p not in unique_links:
                unique_links.append(p)
    try:
        avg_link_traversals.append(len(links) / len(unique_links))
    except:
        avg_link_traversals.append(0) #chose 0 for single node paths. Maybe should be changed, or just omit paths
test_df['Avg_linkTraversals'] = avg_link_traversals
test_df.head()


# # Detourness: (number of links in shortest path from start to last node) / (links in path)

# In[30]:


detourness = []
for n in range(91486):
    path = backspace_cleaner(list(test_df.iat[n,2]))
    for m in range(len(path)):
        path[m] = unquote(path[m])
    link_count = len(path) - 1
    try:
        shortest_path = nx.dijkstra_path(G, path[0], path[-1])
        detourness.append((len(shortest_path)-1)/link_count)
    except:
        detourness.append('err')
test_df['detourness'] = detourness
test_df


# # The Relative Mean Distance from Root (RMDFR)
# Mean distance from the root node to every other node in the clickstream graph normalized over the longest distance from starting node.

# In[31]:


RMDFR = []
errors = []
for n in range(91486):
    path = backspace_cleaner(list(test_df.iat[n,2]))
    for m in range(len(path)):
        path[m] = unquote(path[m])
    if len(path)>1:
        DFR = []
        for m in range(1,len(path)):
            try:
                DFR.append(len(nx.dijkstra_path(G, path[0], path[m]))-1) #distance from root
            except:
                DFR.append(0)
                errors.append(n)
        #print(DFR)
    
        MDFR = sum(DFR)/len(DFR) # mean distance from root
        RMDFR.append(MDFR/max(DFR)) #relative mean distance from root
    else:
        RMDFR.append(0)
    
test_df['RMDFR'] = RMDFR
test_df


# In[32]:


error_paths = []
for n in errors:
    path = backspace_cleaner(list(test_df.iat[n,2]))
    for m in range(len(path)):
        path[m] = unquote(path[m])
    error_paths.append(path)
edges_to_add = []
for broken_path in error_paths:
    counter = 0
    for m in range(len(broken_path)):
        if counter != 0:
            try:
                nx.dijkstra_path(G, broken_path[m-1], broken_path[m])
            except:
                edges_to_add.append((broken_path[m-1], broken_path[m]))
        counter += 1
        
edges_to_add


# # Connection ratio:(links used in path/all links connecting nodes in path)  

# In[33]:


con_rat = []
for n in range(91486):
    path = backspace_cleaner(list(test_df.iat[n,2]))
    for m in range(len(path)):
        path[m] = unquote(path[m])
    sub = G.subgraph(path)
    edges = len(sub.edges())
    try:
        con_rat.append((len(path)-1)/edges)
    except:
        con_rat.append(0)
        #print(n,path) #errors for unfixed names and single-node paths?
test_df['connection_ratio'] = con_rat
test_df


# # Length of Shortest Path from Root to Target
# i threw unquote in here, havent tried running it to see if it worked

# In[34]:


shortest_path = []
for n in range(91486):
    start = unquote(str(final_df.iat[n,2][0]))
    target = unquote(str(final_df.iat[n,7]))
    try:
        shortest_path.append(len(nx.dijkstra_path(G,start,target)))
    except:
        shortest_path.append('err') #naming errors
test_df['shortest_path'] = shortest_path
test_df


# ### Backtracking (% of traversals to previously visited nodes)

# In[35]:


relative_backtracking = []
for n in range(91486):
    path = backspace_cleaner(list(test_df.iat[n,2]))
    uniques=[]
    for node in path:
        if node not in uniques:
            uniques.append(node)
    relative_backtracking.append( (len(path)-len(uniques))/len(path) )

test_df['relative_backtracking'] = relative_backtracking
test_df


# ### Clickstream Compactness

# In[ ]:


click_comp = []
errors = []
for n in range(91486):    
    path = backspace_cleaner(list(test_df.iat[n,2]))
    for m in range(len(path)):
        path[m] = unquote(path[m])
    N=[]
    for node in path:
        if node not in N:
            N.append(node)
    n = len(N)
#     M = pd.DataFrame(np.zeros((len(path),len(path))), columns = path, index = path)
#     edges = [(path[n],path[n+1]) for n in range(len(path)-1)]
#     for (a,b) in edges:
#         M.at[a,b] = 1
    if n>1:
        C = pd.DataFrame(columns = path, index = path)
        for a in list(C.index):
            for b in list(C.index):
                try:
                    distance = len(nx.dijkstra_path(G,a,b))-1
                    C.at[a,b] = int(distance)
                except:
                    errors.append(n)
                    C.at[a,b] = n  #paper says to set inf distance to n, but how would there be nodes with inf distance in same clickstream?
        Csum = C.values.sum()
        click_comp.append( ((n**2)*(n-1)-Csum) / (n*((n-1)**2)) )
    else:
        click_comp.append(0)
test_df['clickstream_compactness'] = click_comp
test_df


# In[ ]:


test_df.to_csv("model_Data.csv", index=False)
print("Done!")


# In[ ]:


error_paths = []
for n in errors:
    path = backspace_cleaner(list(test_df.iat[n,2]))
    for m in range(len(path)):
        path[m] = unquote(path[m])
    error_paths.append(path)
edges_to_add = []
for broken_path in error_paths:
    counter = 0
    for m in range(len(broken_path)):
        if counter != 0:
            try:
                nx.dijkstra_path(G, broken_path[m-1], broken_path[m])
            except:
                edges_to_add.append((broken_path[m-1], broken_path[m]))
        counter += 1
        
edges_to_add


# In[ ]:





# In[ ]:


current_distance_to_target = []
for n in range(91486):
    current = unquote(str(test_df.iat[n,2][-1]))
    target = unquote(str(test_df.iat[n,8]))
    try:
        current_distance_to_target.append(len(nx.dijkstra_path(G,current,target)))
    except:
        current_distance_to_target.append('err') #naming errors
test_df['current_distance_to_target'] = current_distance_to_target
test_df


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Project_paths.ipynb')


# In[ ]:




