# crowdwisdom

The Data generated should be kept in the Data directory, only selected data logs 
should be uploaded to github. The gitignore ignores the Data directory, but if you 
wish to push a data file then use a command similiar to this: 

`git add -f Data/gaDataCummulative_20170904_155230.txt`

RECENT CHNAGES:
1. The code logs all relevant data from a run to a file with the date time 
stamp as part of its filename. This file was 1.8mb in size for the 222 node graph.

2. You can specify the number of nodes on the command line, the number of nodes that
you can choose from are: 11, 22, 44, 77, 97, 100, 222. The default if no number is 
specified is 22 nodes. The format for the command to run the code with 44 nodes is:

`./WeightedWoC.py 44`



