#!/usr/bin/env python3 

from multiprocessing import Process, Manager
import sys
import time
import math,time,random
import matplotlib.pyplot as plt

###############################################################################################
#  Program Purpose:   Program uses a hybrid approach to TSP
#  Specifically, several GA algorithm results are combined using a Wisdom of Crowds approach
#  
#   Windows command line to execute script multiple times is:  for %i in (#ofIterations) do python scriptName.py
#   Coded by:  Chris Lowrance
################################################################################################
# sudo apt-get install python3-matplotlib

#************** Start - Read File Subroutine **************************************************
# Reads file and returns 'cities' array sequence that needs to be permutated
# Also returns 'cityCoords' matrix that contains each cities' x,y coordinates
# Finally, it returns 'numOfCities' which provides the total number of nodes
def readFile(fileName):
    #Initialize local function variables
    cityCoords = [[0.0,0.0]] # initialize coordinate matrix; position zero is all zeroes and won't be used; index 1 cooresponds to city 1, etc. 
    cities = [] #initialize city matrix
    firstEntry = False
    f = open(fileName,'r')

    while not firstEntry:
        line = f.readline()
        if line[0] == '1': #then, found first city entry; store these until EOF
            firstEntry = True
    #found first file entry; now iterate over file and store coordinates in array as floats (array index = city #)
    EOF = False
    while not EOF:
        line = line.split()
        city = int(line[0])
        xCoord = float(line[1])
        yCoord = float(line[2])
        cities.append(city)
        cityCoords.append([xCoord,yCoord])
        line = f.readline()
        if line == '':
            EOF = True

    #get size of array (tells us how many cities in file)
    numOfCities = len(cities)

    return cities,cityCoords,numOfCities
#************** End - Read File Subroutine ******************************************************

#************** Start - Generate Distance Matrix Subroutine**************************************
# Generates a matrix with all distances between nodes
# Prevents program from repeating the same calculations; instead uses look-up table
# Look-up any distance using between nodes using 'dist[cityA][cityB]' where city A/B = 1,2,...
def calcAllDistances(numOfCities,cityCoords):
    #initialize matrix for storing distance calculations; will save computational time by not repeating same calculations
    dist = [[0]*(numOfCities+1) for x in range(numOfCities+1)] # generate a n x n matrix of zeros for storing distance calculations; +1 because ignoring 0 index location    
    #calculate the distances between all possible nodes and store in matrix
    for i in range(1,numOfCities): # for i = 1 to one less iteration than # of cities
        for j in range(i+1,numOfCities+1): # for j = i+1 (i.e. next city) to end of city list (shrinking window iteration)
            d = distCalc(cityCoords[i],cityCoords[j])
            dist[i][j] = d # distance from city A to B is the same as B to A
            dist[j][i] = d # so, store both in matrix
    return dist
#************** End - Generate Distance Matrix Subroutine*****************************************

#************** Start - Distance Calculator Subroutine ****************************************
# Performs distance formula and returns answer
# Must send city A and city B coordinates in the form of an array (x_coord,y_coord)
def distCalc(city1Coords,city2Coords):
    dist = math.sqrt( (city1Coords[0] - city2Coords[0])**2 + (city1Coords[1] - city2Coords[1])**2 )
    return dist
#************** End - Distance Calculator Subroutine ******************************************

#************** Start - Generate Population ******************************************************
#Function generates a population of random lists (array) of cities w/o duplicates
#The size of the population that is generated can be changed using variable w/i function. 
def genRandomPop(numOfCities,popSize):
    population = [] #initialize array to hold a series of populations
    for i in range(0,popSize):
        newPop = random.sample(range(1,numOfCities+1), numOfCities) #generate a random population w/ no duplicates based on number of given cities
        population.append(newPop) #append the new population to array
    return population
       
#************** End - Generate Population ******************************************************

#************** Start - Genetic Algorithm **********************************************************
#record lowest cost (track number of iterations where cost doesn't lower)
def gaTSP(population,numOfCities, gaPaths, gaCosts, genCosts):
	fitEnough = False # flag; once evolution determined satisfactory, then this flag goes True
	firstIteration = True # flag for storing initial shortest path as best path overall
	bestOverallCost = 0
	loopsWithoutImprovement = 0
	generationNum = 0
	generationBestCost = []
	while not fitEnough:
        #select parents
		parents, population,lowestCost,lowestTourKey = parentSelection(population) # returns a two element list containing parents & key of shortest tour
		child = reproduce(parents,numOfCities) #produce child; arguments are from parentsSection function
		generationNum += 1
		population.append(child)
        #are we seeing improvement in genetics?
		if lowestCost < bestOverallCost or firstIteration:
			bestOverallCost = lowestCost
			firstIteration = False
			loopsWithoutImprovement = 0 #reset
		else:
			loopsWithoutImprovement += 1
		if loopsWithoutImprovement >=50: # once GA stagnates, try mutating some more; add another mutated child to population
            #mutate child again and add to population
			childCost = findCost(child,numOfCities)
            #print ("random mutating")
            #print ("child 1 costs = ",childCost)
			child2 = randomMutation(child,childCost,numOfCities)
			child3 = randomMutation(child2,childCost,numOfCities)
            #child2Cost = findCost(child2,numOfCities)
            #child3Cost = findCost(child3,numOfCities)
            #print ("child 2 & 3 costs = ",child2Cost,child3Cost)
			population.append(child2)
			population.append(child3)
		if loopsWithoutImprovement >= 300 or generationNum > 3000: # stopping criteria:  if looping w/o improvement or produced lots of generations
            #print (loopsWithoutImprovement,generationNum)
			fitEnough = True
		generationBestCost.append(bestOverallCost)
	gaPaths.append(population[lowestTourKey])
	gaCosts.append(lowestCost)
	genCosts.append(generationBestCost)
	#return population[lowestTourKey],lowestCost,generationBestCost

#************** End - Genetic Algorithm ************************************************************

#************** Start - Find Cost*******************************************************************    
#Function that determines the total cost of a tour (used for fitness function)
def findCost(tour,numOfCities):
    totalDist = 0 # initialize summer to track distance of tour
    for i in range(0,numOfCities): # iterate through all cities
        nodeA = tour[i]
        nodeB = tour[i-1]
        totalDist = totalDist + dist[nodeA][nodeB] #find distance between adjacent cities, then add to total
    return totalDist
#************** End - Find Cost*********************************************************************     

#************** Start - Parent Selection******************************************************************* 
# Function that selects parents using biased random function; function prefers lower costs, but randomly
# selects two parents for breeding out of a pool of "good" candidates using fitness function for evaluation
def parentSelection(population): # argument is a dictionary w/ key = pop index; value = tour cost
    #parentCandidates = {}
    toursDict = {} # initialize tour dictionary; key = tour/pop. #, value = cost of tour
    newPop = []
    for i in range(0,len(population)):
        tourDistance = findCost(population[i],numOfCities)
        toursDict[i] = tourDistance #update dictionary with tour number (key) and tour cost (value)
    #finished calculating cost with each tour
    #record best tour # and its cost
    #choose top parents
    initialLook = True
    for j in range(0,4): #choose top 3 parent candidates
        shortestTourKey = min(toursDict,key=toursDict.get) #find shortest tour
        if initialLook:
            lowestTourCost = toursDict[shortestTourKey] #store for tracking best overall cost & key
            lowestTourKey = shortestTourKey
            initialLook = False
        newPop.append(population[shortestTourKey]) #insert best tour sequence in new population pool
        del toursDict[shortestTourKey]
    #finishing picking best parental candidates
    parents = random.sample(newPop, 2) #randomly pick two parents out of best pool  
    
    return parents,newPop,lowestTourCost,lowestTourKey
#************** End - Parent Selection*******************************************************************    


##########################################################################################################
##########################################################################################################
####################### Crossover Reproduction Methods ###################################################

#************** Start - Reproduce (w/ single crossover point tested at all points**************************    
def reproduce(parents,numOfCities):
    firstIteration = True #falg
    lowestCost = 0.0 # initialize variable to track cost of shortest route
    parentForCopy = random.randint(0,1) # use number to randomly pick which parent for contiguous chromosome copy
    if parentForCopy == 0: # parent B should be other parent number; parent used to fill unused nodes not copied
        parentForFill = 1
    else:
        parentForFill = 0
    parentA = parents[parentForCopy]
    parentB = parents[parentForFill]
    
    for crossOverPt in range (0,numOfCities): # iterate through multiple crossover points
        copyFromA = parentA[numOfCities-crossOverPt:numOfCities] #will crossover ending portion of parentA & move to start of child
        child = copyFromA #start the child with the tailend of ParentA 
        for j in range(0,numOfCities): # iterate over all possible cities indexes
            if parentB[j] not in copyFromA: 
                child.append(parentB[j]) #only append ordered portions of B that are not already from A (need to avoid duplicates)
        currentCost = findCost(child,numOfCities) # find distance for this child
        if currentCost < lowestCost or firstIteration: # if better child, then save it; else delete it
            bestCrossover = list(child)
            lowestCost = currentCost
            firstIteration = False
    #!!!!!!!!!!!!!! Finished trying a number of single crossover points !!!!!!!!!!!!!!!!!!!!!!!!!111
    #!!!!!!!!!!!!!! Now, select mutation method 
    mutatedChild = neighborMutation(bestCrossover,lowestCost,numOfCities) #mutate this child
    
    return mutatedChild
#************** End - Reproduce *******************************************************************

################################################################################################################
####################### END - Crossover Reproduction Methods ###################################################


###############################################################################################################
#######################         Mutation Methods            ###################################################

#************** Start - Neighbor Mutation *******************************************************************
def neighborMutation(bestMutation,lowestCost,numOfCities):
    tempChild = list(bestMutation) # create a copy of this incoming crossover list; trying to beat it
    startPt = random.randint(0,numOfCities-2)
    for i in range(startPt,numOfCities-1): # 
        temp = tempChild[i-1] #save position 1
        tempChild[i-1] = tempChild[i] # save position 2 into 1's position
        tempChild[i] = temp #put position 1 into position 2
        muteCost = findCost(tempChild,numOfCities) # find cost of this mutated version
        if muteCost < lowestCost: #check to see if it's better
            lowestCost = muteCost
            bestMutation = list(tempChild)
            break
        #else:
        #    tempChild = list(bestMutation) 
    #print ("after mutation = ",lowestCost)
    return bestMutation
      
#************** End - Random Mutation *******************************************************************

#************** Start - Neighbor2 Mutation (perform multiple iterations)**************************************
def neighbor2Mutation(bestMutation,lowestCost,numOfCities):
    
    for j in range(0,5):
        tempChild = list(bestMutation) # create a copy of this incoming crossover list; trying to beat it
        startPt = random.randint(0,numOfCities-2)
        for i in range(startPt,numOfCities-1): # 
            temp = tempChild[i-1] #save position 1
            tempChild[i-1] = tempChild[i] # save position 2 into 1's position
            tempChild[i] = temp #put position 1 into position 2
            muteCost = findCost(tempChild,numOfCities) # find cost of this mutated version
            if muteCost < lowestCost: #check to see if it's better
                lowestCost = muteCost
                bestMutation = list(tempChild)
                break
            #else:
            #    tempChild = list(bestMutation) 
        
    return bestMutation
      
#************** End - Random Mutation *******************************************************************

#************** Start - Random Mutation *******************************************************************
def selectMutation(bestMutation,lowestCost,numOfCities):
    tempChild = list(bestMutation) # create a copy of this incoming crossover list; trying to beat it
    muteCost = 10000 # initialize mutation cost variable to high amount in order to enter loop
    attempt = 0 # initialize mutation attempt counter;
    while muteCost > lowestCost and attempt<100: # if found a better mutation or tried 100 attempts, then quit
        attempt += 1
        randPoints = random.sample(range(0,numOfCities-1), 2) #generate two unique random integers from 1 to 100
        pos1 = randPoints[0]
        pos2 = randPoints[1]
        temp = tempChild[pos1] #save position 1
        tempChild[pos1] = tempChild[pos2] # save position 2 into 1's position
        tempChild[pos2] = temp #put position 1 into position 2
        muteCost = findCost(tempChild,numOfCities) # find cost of this mutated version
        if muteCost < lowestCost: #check to see if it's bpyetter
            lowestCost = muteCost
            bestMutation = list(tempChild)
            break
        else:
            tempChild = list(bestMutation) # reset tempChild to previous crossover list 
           
    return bestMutation
#************** End - Random Mutation *******************************************************************

#************** Start - Random Mutation *******************************************************************
def systematicMutation(bestMutation,lowestCost,numOfCities):
    tempChild = list(bestMutation) # create a copy of this incoming crossover list; trying to beat it
    #for i in range(0,5): # perform a couple of mutations
    pos1 = random.randint(0,99)
    #pos1 = 0
    temp = tempChild[pos1] #save node number that was originally in position 1
    for swapPos in range(-1,99):        
            if swapPos == pos1:
                pass
            else:
                tempChild[pos1] = tempChild[swapPos] # put swapPos# into 1's position
                tempChild[swapPos] = temp #put position 1 into the swap position
                muteCost = findCost(tempChild,numOfCities) # find cost of this mutated version
                if muteCost < 0.01*lowestCost: #check to see if it's better
                    lowestCost = muteCost
                    print ("system worked!!")
                    bestMutation = list(tempChild)
                    break
                else: # if not better, then put nodes back in original spots
                    tempChild[swapPos] = tempChild[pos1]
                    tempChild[pos1] = temp
        
    return bestMutation
#************** End - Random Mutation *******************************************************************

#************** Start - Random Mutation *******************************************************************
def randomMutation(bestMutation,lowestCost,numOfCities):
    attempt = 0
    tempChild = list(bestMutation)
    while attempt<=100: # mutate up to 5 percent of the tour
        attempt += 1
        pos1 = random.randint(0,numOfCities-1)
        pos2 = random.randint(0,numOfCities-1)
        temp = tempChild[pos1] #save position 1
        tempChild[pos1] = tempChild[pos2] # save position 2 into 1's position
        tempChild[pos2] = temp
        muteCost = findCost(tempChild,numOfCities) # find cost of this mutated version
        if muteCost < lowestCost+.01*lowestCost: #check to see if it's better
            #print ("found better!!!")
            lowestCost = muteCost
            bestMutation = list(tempChild)
            return bestMutation # stop looping if found a better mutation
        else:
            tempChild[pos2] = tempChild[pos1]
            tempChild[pos1] = temp
    return tempChild
#************** End - Random Mutation *******************************************************************

###############################################################################################################
#######################     End -    Mutation Methods            ###################################################


#********************* Start - Build Woc Histograms **************************************************
def buildWeightedWocHistograms(gaPaths,gaCosts,numOfCities):
    #initialize variables
    unwFirstIteration = True
    expFirstIteration = True
    percFirstIteration = True
    bestUnwPathCost = 0
    bestExpPathCost = 0
    bestPercPathCost = 0
    worstUnwPathCost = 0
    worstExpPathCost = 0
    worstPercPathCost = 0
    expMapRange = 5 #will map e^-x from 0 to this number
    expWeightOfExperts = []
    percWeightOfExperts = []
    #determine cost ratio that will be used in weighting WoC opinions
    maxCost = max(gaCosts)
    minCost = min(gaCosts)
    totalCostDiff = maxCost - minCost
    unweightedHistogram = [[0 for x in range(numOfCities+1)] for x in range(numOfCities+1)] # create a (n x n) matrix based on num. of cities; +1 because index 0 not used
    expHistogram = [[0 for x in range(numOfCities+1)] for x in range(numOfCities+1)] # create a (n x n) matrix based on num. of cities; +1 because index 0 not used
    percHistogram = [[0 for x in range(numOfCities+1)] for x in range(numOfCities+1)] # create a (n x n) matrix based on num. of cities; +1 because index 0 not used
    # histogram will hold the count of each edge-to-edge connection found in the group of "experts"
    for pathNum in range(0,expertNum): #iterate through all experts
        #Determine the weight of this expert based on distance (cost diff) from best expert
        costDiffFromBest = gaCosts[pathNum] - minCost #find the difference of current expert from best expert
        costRatio = costDiffFromBest / totalCostDiff
        percWeightOfExpert = 1 - costRatio
        percWeightOfExperts.append(percWeightOfExpert) #store weights for plotting
        expValue = costRatio*expMapRange 
        expWeightOfExpert = math.exp(-1*expValue)
        expWeightOfExperts.append(expWeightOfExpert) #store weights for plotting        
        for edgeNum in range(0,numOfCities): #iterate through all nodes
            adjNode = gaPaths[pathNum][edgeNum-1] #get adjacent node
            node = gaPaths[pathNum][edgeNum] #get node
            unweightedEdgeCount = unweightedHistogram[node][adjNode] # read num. of occurences for this edge
            expEdgeCount = expHistogram[node][adjNode] # read num. of occurences for this edge
            percEdgeCount = percHistogram[node][adjNode] # read num. of occurences for this edge
            unweightedEdgeCount = unweightedEdgeCount + 1 # increment the occurence by 1
            expEdgeCount = expEdgeCount + expWeightOfExpert # increment the occurence by exp weight
            percEdgeCount = percEdgeCount + percWeightOfExpert # increment the occurence by exp weight
            unweightedHistogram[node][adjNode] = unweightedEdgeCount # update histogram
            expHistogram[node][adjNode] = expEdgeCount
            percHistogram[node][adjNode] = percEdgeCount
    unwPath,unwPathCost=buildWocPath(unweightedHistogram,numOfCities)
    if unwPathCost < bestUnwPathCost or unwFirstIteration == True: #if current combo of experts produces best path, then store it!
        bestUnwPathCost = unwPathCost
        bestUnwExpertNum = pathNum + 1
        bestUnwPath = unwPath
        unwFirstIteration = False
    expPath,expPathCost=buildWocPath(expHistogram,numOfCities)
    if expPathCost < bestExpPathCost or expFirstIteration == True: #if current combo of experts produces best path, then store it!
        bestExpPathCost = expPathCost
        bestExpExpertNum = pathNum + 1
        bestExpPath = expPath
        expFirstIteration = False
    percPath,percPathCost=buildWocPath(percHistogram,numOfCities)
    if percPathCost < bestPercPathCost or percFirstIteration == True: #if current combo of experts produces best path, then store it!
        bestPercPathCost = percPathCost
        bestPercExpertNum = pathNum + 1
        bestPercPath = percPath
        percFirstIteration = False    
    #Finished building histogram, now build the path
    #Finished with last expert, so capture values associate with entire crowd (i.e. all inputs)
    lastUnwCost = unwPathCost
    lastExpCost = expPathCost
    lastPercCost = percPathCost
    #unweightedHistogram,expHistogram,percHistogram
    return bestUnwPathCost,bestUnwExpertNum,bestUnwPath,bestExpPathCost,bestExpExpertNum,bestExpPath,bestPercPathCost,bestPercExpertNum,bestPercPath,percWeightOfExperts,expWeightOfExperts,lastUnwCost,lastExpCost,lastPercCost
#********************* End - Build WoC Histograms **************************************************   

#********************* Start - Build Woc Paths ************************************************** 
def buildWocPath(histogram,numOfCities):
    #Finished building histogram, now build the path
    firstIteration = True #flag used in greedy approach (if best option not available)
    bestOptionDist = 0 # used to pick best greedy option if best WoC option not available
    firstIteration2 = True # flag used in tracking overall best path cost based on starting city
    bestPathCost = 0 # initialize variable used to track cost of overall best path
    for startCity in range(1,numOfCities+1): #try every city as starting city; pick the one that results in shortest route
        
        wisePath = [startCity] # list to store wisdom path; start with city #1
        currentNode = startCity # keep track of which city we're at while building "wisest path"; initialized with starting city
        for iterNum in range(1,numOfCities): #iterate over every city number to build tour; already added one city (so start at 1)
            bestEdge = 0 # (re)initialize variable to track edge num. which occurs most frequently in histogram
            for cityNum in range(1,numOfCities+1): #go through all possible edges
                testEdge = histogram[currentNode][cityNum] #extract histogram count
                if testEdge > bestEdge and cityNum not in wisePath: #if count > than current best & isn't in wisest path
                    bestEdge = testEdge
                    bestEdgeCity = cityNum
            #finished iterating over list to find best edge; found it!               
            #print ("best Edge = ",bestEdge)
            if bestEdge != 0: #need to ensure a best edge was found; if not, need to use greedy
                wisePath.append(bestEdgeCity) #add best edge to path
                currentNode = bestEdgeCity # now we're going to look for this best edge for this new node
            else: # iterate through all cities; chose remaining city (not in path) that is closest to this node
                for possibleCity in range(1,numOfCities+1): #go through all possible cities
                    if possibleCity not in wisePath: # if city not already in path, then consider it
                        optionDist = dist[currentNode][possibleCity] # find distance from current city to this option
                        if optionDist < bestOptionDist or firstIteration: # if this distance is < current best, then store it
                            bestOptionCity = possibleCity
                            firstIteration = False
                #finished iterating through greedy city options, now append best to path
                wisePath.append(bestOptionCity)
                currentNode = bestOptionCity
                firstIteration = True #flag used in greedy approach (if best option not available)
                bestOptionDist = 0 # used to pick best greedy option if best WoC option not available
            
        #finished building a WoC route using a particular starting city
        #now, record cost of this route and see if it beats the cost of others
        
        #print (wisePath)        
        pathCost = findCost(wisePath,numOfCities) # cost of this path (for this "startCity")
        if pathCost < bestPathCost or firstIteration2 == True: #
            wisestPath = wisePath
            bestPathCost = pathCost
            firstIteration2 = False
    return wisestPath,bestPathCost
#********************* End - Wisdom of Crowd **************************************************

# DML***************** START of EDGE USAGE analysis *******************************************
def calcEdgeUsage(cityCoords,gaPaths):
    edges = []  
    edgesUsage = []    
    #print("citycoords",cityCoords)
    for gaPath in gaPaths:
        #print("gapath: ",gaPath)
        # Create a list of edges and a count of the number of times the edge is used
        for i in range(0,len(gaPath)-2):
            edge = [gaPath[i],gaPath[i+1]]
            backEdge = [gaPath[i+1],gaPath[i]]
            if (edge in edges):
                indx = edges.index(edge)
                edgesUsage[indx]+=1  
            elif (backEdge in edges):
                indx = edges.index(backEdge)
                edgesUsage[indx]+=1                
            else:
                edges.append(edge)
                edgesUsage.append(1)
        # Add the last edge and count to the list
        edge = [gaPath[len(gaPath)-1],gaPath[0]] 
        backEdge = [gaPath[0],gaPath[len(gaPath)-1]]
        if (edge in edges):
            indx = edges.index(edge)
            edgesUsage[indx]+=1  
        elif (backEdge in edges):
            indx = edges.index(backEdge)
            edgesUsage[indx]+=1                 
        else:
            edges.append(edge)
            edgesUsage.append(1)
    #print("Edges",edges)
    #print("EdgesUsage", edgesUsage)
    return edges, edgesUsage
    
def drawEdgeUsage(cityCoords,edges,edgesUsage):
    xcoords = []
    ycoords = []    
    plt.figure(22)
    ax1 = plt.subplot()
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    title = ('WoAC - Agreement - Number of Cities: '+str(cityCount))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.suptitle('Thicker lines represent more agreement')
    amount =max(edgesUsage)
    i =0
    for edge in edges:
        xcoords = [cityCoords[edge[0]][0], cityCoords[edge[1]][0]]
        ycoords = [cityCoords[edge[0]][1], cityCoords[edge[1]][1]]
        c = [float(edgesUsage[i])/float(amount), 0.1, 0.1] #R,G,B
        w = (float(edgesUsage[i])/float(amount)*7)+0.05
        if edgesUsage[i] > (amount/2):
            fig22 = plt.plot(xcoords,ycoords,'or-',linewidth =w)        
        else:
            fig22 = plt.plot(xcoords,ycoords,'ob-',linewidth =w)
        i+=1

#************** Start - Tour Plotter ************************************************************
# This function plots the optimum tour using the coordinates of the cities
def genPlots(cityCoords,numOfCities,gaPaths,gaCosts,genCosts,wocPath,expWocPath,percWocPath,expWeightOfExperts,percWeightOfExperts,bestUnwPathCost,bestExpPathCost,bestPercPathCost):


    #Gather statistical data for evaluation purposes
    bestGaCost = round(min(gaCosts),3)
    bestGaIndex = gaCosts.index(min(gaCosts)) #get index cooresponding to lowest cost GA
    bestGaPath = gaPaths[bestGaIndex]
    meanGaCost = round((sum(gaCosts)/len(gaCosts)),3)
    bestUnwPathCost = round(bestUnwPathCost,3)
    bestExpPathCost = round(bestExpPathCost,3)
    bestPercPathCost = round(bestPercPathCost,3)
    allWocCosts = [bestUnwPathCost, bestExpPathCost, bestPercPathCost]
    bestWocPathCost = min(allWocCosts)
    indexNumOfBest = allWocCosts.index(min(allWocCosts))
    if indexNumOfBest == 0: #then unweighted is best
        bestWocPath = wocPath
    elif indexNumOfBest == 1: #then, exp weighted is best
        bestWocPath = expWocPath
    elif indexNumOfBest == 2: #then, percent weighted is best
        bestWocPath = percWocPath
        
   
    #shortestCost = round(shortestCost,3) # round to 3 decimal places
    xGaCoords = []
    yGaCoords = []
    xBestWocCoords = []
    yBestWocCoords = []
    xWiseCoords = []
    yWiseCoords = []
    xExpWiseCoords = []
    yExpWiseCoords = []
    xPercWiseCoords = []
    yPercWiseCoords = []
    
    for i in range (0,numOfCities): # iterate through tour of cities
        gaCityNum = bestGaPath[i] # extract city # in sequence order (1 at a time)
        bestWocCityNum = bestWocPath[i]
        wiseCityNum = wocPath[i]
        expWiseCityNum = expWocPath[i]
        percWiseCityNum = percWocPath[i]
        xGaCoords.append(cityCoords[gaCityNum][0]) # for this city, extract x coords and place in an array
        yGaCoords.append(cityCoords[gaCityNum][1]) # for this city, extract y coords and place in an array
        xBestWocCoords.append(cityCoords[bestWocCityNum][0]) # for this city, extract x coords and place in an array
        yBestWocCoords.append(cityCoords[bestWocCityNum][1]) # for this city, extract y coords and place in an array
        xWiseCoords.append(cityCoords[wiseCityNum][0]) # for this city, extract x coords and place in an array
        yWiseCoords.append(cityCoords[wiseCityNum][1]) # for this city, extract y coords and place in an array
        xExpWiseCoords.append(cityCoords[expWiseCityNum][0]) # for this city, extract x coords and place in an array
        yExpWiseCoords.append(cityCoords[expWiseCityNum][1]) # for this city, extract y coords and place in an array
        xPercWiseCoords.append(cityCoords[percWiseCityNum][0]) # for this city, extract x coords and place in an array
        yPercWiseCoords.append(cityCoords[percWiseCityNum][1]) # for this city, extract y coords and place in an array
    #Now account for travel back from last node to starting node
    xGaCoords.append(xGaCoords[0]) # so append first coords to end of array for plotting purposes
    yGaCoords.append(yGaCoords[0])
    xBestWocCoords.append(xBestWocCoords[0])
    yBestWocCoords.append(yBestWocCoords[0])
    xWiseCoords.append(xWiseCoords[0])
    yWiseCoords.append(yWiseCoords[0])
    xExpWiseCoords.append(xExpWiseCoords[0])
    yExpWiseCoords.append(yExpWiseCoords[0])
    xPercWiseCoords.append(xPercWiseCoords[0])
    yPercWiseCoords.append(yPercWiseCoords[0])

    
    maxGens = 0 # initialize variable to store max value of x-axis
    plt.figure(1)
    ax1 = plt.subplot()
    textXloc = min(xGaCoords)
    textYloc = min(yGaCoords)
    for j in range(0,len(genCosts)): #iterate through each GA cost array (cost vs gen#)
        plt.plot(genCosts[j], ':')
        numOfGens = len(genCosts[j]) # find how many generations this GA took
        if numOfGens> maxGens:       # record max
            maxGens = numOfGens
    meanGa = [meanGaCost]*maxGens
    wise = [bestWocPathCost]*maxGens
    plt.plot(meanGa, 'r--',linewidth = 3.0,label='Avg. GA Cost')
    plt.plot(wise, 'b-.',linewidth = 4.0,label='Best WoAC Cost')
    plt.legend(prop={'size':8})
    supTitleLabel = str(cityCount)+' Node TSP with ' + str(expertNum) + ' Experts'
    plt.suptitle(supTitleLabel, fontsize=14, fontweight='bold')
    plt.title('Cost of Best GA Pop. Members per Search & Generation')
    plt.xlabel('Generation Number')
    plt.ylabel('Cost')
    plt.text(0.3,0.75, 'Converged Path Cost Comparison', transform=ax1.transAxes,fontweight='bold') #,bbox=dict(facecolor='green', alpha=0.5),va='top')
    plt.text(0.35,0.7, 'Avg. GA Path Cost = '+str(meanGaCost)+'\n'+'Best GA Path Cost = '+str(bestGaCost), transform=ax1.transAxes,bbox=dict(facecolor='red', alpha=0.5),va='top')
    #plt.text(0.35,0.7, 'Best GA Path Cost = '+str(bestGaCost) , transform=ax1.transAxes,bbox=dict(facecolor='red', alpha=0.5),va='top')
    plt.text(0.35,0.55, 'Best WoAC Cost = '+str(bestWocPathCost) , transform=ax1.transAxes,bbox=dict(facecolor='blue', alpha=0.5),va='top')
    #plt.savefig('222tspSearches.png')
    
    #linestyle or ls [ '-' | '--' | '-.' | ':' | 'steps' | ...] 
    plt.figure(2)
    ax2 = plt.subplot()
    textXloc = min(xGaCoords)
    textYloc = min(yGaCoords)
    fig2 = plt.plot(xGaCoords,yGaCoords,'ro--',linewidth=3.0,label='Best GA (out of 30)')
    fig2 = plt.plot(xBestWocCoords,yBestWocCoords, 'b-',label='Best WoAC (out of 3)')
    #fig2 = plt.plot(xExpWiseCoords,yExpWiseCoords, 'g-.',linewidth=2.0,label='WoAC - Exp. Weighted')
    #fig2 = plt.plot(xPercWiseCoords,yPercWiseCoords, 'm:',linewidth=2.0,label='WoAC - Percent Weighted')
    #fig2 = plt.plot(xWiseCoords,yWiseCoords, 'bo-',label='Wisdom of Crowd')
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax2.legend(loc='upper center',prop={'size':8}, bbox_to_anchor=(0.5, -0.1),
        fancybox=True, shadow=True, ncol=5)
    #plt.legend()
    supTitleLabel = str(cityCount)+' Node TSP with ' + str(expertNum) + ' Experts'
    plt.suptitle(supTitleLabel, fontsize=14, fontweight='bold')
    plt.title('Best GA Path and WoAC Aggregration Paths')
    #plt.text(0.35,0.9, 'Best GA Path Cost = '+str(bestGaCost) , transform=ax2.transAxes,bbox=dict(facecolor='red', alpha=0.5),va='top')
    #plt.text(0.35,0.85, 'Avg. GA Path Cost = '+str(meanGaCost) , transform=ax2.transAxes,bbox=dict(facecolor='red', alpha=0.5),va='top')
    #plt.text(0.35,0.8, 'Wisdom Path Cost = '+str(wisestPathCost) , transform=ax2.transAxes,bbox=dict(facecolor='blue', alpha=0.5),va='top')
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    #plt.savefig('222tspPaths.png')
        
    '''
    #For individual plots   
    #linestyle or ls [ '-' | '--' | '-.' | ':' | 'steps' | ...] 
    plt.figure(3)
    ax2 = plt.subplot()
    textXloc = min(xGaCoords)
    textYloc = min(yGaCoords)
    fig2 = plt.plot(xGaCoords,yGaCoords,'ro--',linewidth=3.0,label='Best GA (out of 30)')
    fig2 = plt.plot(xWiseCoords,yWiseCoords, 'b-',label='WoAC - Equally Weighted')
    fig2 = plt.plot(xExpWiseCoords,yExpWiseCoords, 'g-.',linewidth=2.0,label='WoAC - Exp. Weighted')
    fig2 = plt.plot(xPercWiseCoords,yPercWiseCoords, 'm:',linewidth=2.0,label='WoAC - Percent Weighted')
    #fig2 = plt.plot(xWiseCoords,yWiseCoords, 'bo-',label='Wisdom of Crowd')
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax2.legend(loc='upper center',prop={'size':8}, bbox_to_anchor=(0.5, -0.1),
        fancybox=True, shadow=True, ncol=5)
    #plt.legend()
    plt.suptitle('44 Node TSP', fontsize=14, fontweight='bold')
    plt.title('Best GA Path and WoAC Aggregration Paths')
    #plt.text(0.35,0.9, 'Best GA Path Cost = '+str(bestGaCost) , transform=ax2.transAxes,bbox=dict(facecolor='red', alpha=0.5),va='top')
    #plt.text(0.35,0.85, 'Avg. GA Path Cost = '+str(meanGaCost) , transform=ax2.transAxes,bbox=dict(facecolor='red', alpha=0.5),va='top')
    #plt.text(0.35,0.8, 'Wisdom Path Cost = '+str(wisestPathCost) , transform=ax2.transAxes,bbox=dict(facecolor='blue', alpha=0.5),va='top')
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    #plt.savefig('44tspPaths.png')
    '''
    
    
    plt.figure(3)
    ax2 = plt.subplot()
    textXloc = min(xGaCoords)
    textYloc = min(yGaCoords)
    fig2 = plt.plot(xGaCoords,yGaCoords,'ro--',linewidth=3.0,label='Best GA (out of 30)')
    plt.legend(prop={'size':8})
    supTitleLabel = str(cityCount)+' Node TSP with ' + str(expertNum) + ' Experts'
    plt.suptitle(supTitleLabel, fontsize=14, fontweight='bold')
    plt.title('Best GA Path')
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    
    '''    
    plt.figure(3)
    ax3 = plt.subplot()
    textXloc = min(xWiseCoords)
    textYloc = min(yWiseCoords)
    fig2 = plt.plot(xWiseCoords,yWiseCoords, 'bo-',label='30 Nodes')
    plt.legend()
    plt.title('Wisdom of Crowd - Unweighted')
    #plt.text(0.35,0.9, 'Best GA Path Cost = '+str(bestGaCost) , transform=ax2.transAxes,bbox=dict(facecolor='red', alpha=0.5),va='top')
    #plt.text(0.35,0.85, 'Avg. GA Path Cost = '+str(meanGaCost) , transform=ax2.transAxes,bbox=dict(facecolor='red', alpha=0.5),va='top')
    #plt.text(0.35,0.8, 'Wisdom Path Cost = '+str(wisestPathCost) , transform=ax2.transAxes,bbox=dict(facecolor='blue', alpha=0.5),va='top')
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    '''
    
    plt.figure(4)
    ax3 = plt.subplot()
    textXloc = min(xExpWiseCoords)
    textYloc = min(yExpWiseCoords)
    fig2 = plt.plot(xExpWiseCoords,yExpWiseCoords, 'bo-',label='44 Nodes')
    plt.legend(prop={'size':8})
    supTitleLabel = str(cityCount)+' Node TSP with ' + str(expertNum) + ' Experts'
    plt.suptitle(supTitleLabel, fontsize=14, fontweight='bold')
    plt.title('WoAC - Exp. Weighted Path')
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    
    plt.figure(5)
    ax3 = plt.subplot()
    textXloc = min(xPercWiseCoords)
    textYloc = min(yPercWiseCoords)
    fig2 = plt.plot(xPercWiseCoords,yPercWiseCoords, 'bo-',label='44 Nodes')
    plt.legend(prop={'size':8})
    supTitleLabel = str(cityCount)+' Node TSP with ' + str(expertNum) + ' Experts'
    plt.suptitle(supTitleLabel, fontsize=14, fontweight='bold')
    plt.title('WoAC - Percentage Weighted Path')
    plt.xlabel('x Coordinate')
    plt.ylabel('y Coordinate')
    
    plt.figure(6)
    exp = sorted(expWeightOfExperts, reverse=True)
    perc = sorted(percWeightOfExperts, reverse=True)
    numOfExperts = len(exp)
    x = list(range(1,numOfExperts+1))
    fig6a = plt.scatter(x,exp,c="b")    
    fig6b = plt.scatter(x,perc,c="k",marker="s",facecolors='none',s=80)
    plt.legend((fig6a, fig6b),('Exponential Weight', 'Percentage Weight'),
           scatterpoints=1,
           loc='lower left',
           ncol=3,
           fontsize=8)
    supTitleLabel = str(cityCount)+' Node TSP with ' + str(expertNum) + ' Experts'
    plt.suptitle(supTitleLabel, fontsize=14, fontweight='bold')
    plt.title('Weight Distribution of Individual GA Outcomes')
    plt.xlabel('Number of GA Outcomes')
    plt.ylabel('Weight of GA Outcome')
    plt.axis([0, numOfExperts+0.5, -0.02, 1.02])
    #plt.savefig('222tspWeights.png')
    
    return
#************** End - Tour Plotter ************************************************************

#******************* Start - Main *****************************************
#initialize variables
filename = ''
cityCount = 0
print("argv count",len(sys.argv),sys.argv)
if (len(sys.argv)>1):
    cityCount = sys.argv[1]
else:
    cityCount = 22
fileName = 'Random'+str(cityCount)+'.tsp'
popSize = 20 #population size
expertNum = 100 # number of ga's that will be considered in 'Wisdom of Crowd' analysis

#gaTimes = [] #list to hold each ga time
#gaPaths = [] #list to hold all the 'expert' ga paths
#gaCosts = [] #list to hold the cost of each ga path
#genCosts = [] #list to hold each array of cost per generation (iteration)

edges = []
edgesUsage = []

cities,cityCoords,numOfCities = readFile(fileName) #read text-file and get arrays with cities & coordinates
dist = calcAllDistances(numOfCities,cityCoords) # setup distance matrix for quick look-up table access


if __name__ == '__main__':
	

	manager = Manager()
	gaPaths = manager.list()
	gaCosts = manager.list()
	genCosts = manager.list()
	procs = []
	
	groupGA_startTime = time.clock() #timer for tracking time to create group of experts
	for i in range(0,expertNum): # perform multiple GA iterations to obtain different tours    
		# Start Genetric Algorithm
		#startTime = time.clock()
		population = genRandomPop(numOfCities,popSize) #generate random population pool w/ size = "popSize" initialized above
		
		p = Process (target=gaTSP, args=(population,numOfCities, gaPaths, gaCosts, genCosts)) #perform genetic algorithm using population;
		procs.append(p)
		p.start()
		
		#singleGA_stopTime = time.clock() #ending time for single GA
		#singleGAtime = singleGA_stopTime - startTime
		#gaTimes.append(singleGAtime)
		#returns best path, its cost, and the cost of each generation
		#gaPaths.append(gaPath) #store ga path
		#gaCosts.append(gaCost) #store ga path cost
		#genCosts.append(genCost) #store the array of costs per generation
		
		
		#print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
		#print ("singleGAtime",singleGAtime)
		#print ("gaPath",gaPath)
		#print ("gaCost",gaCost)
		#print ("genCost",genCost)

	for p in procs:
		p.join()
	
	#finished completed the group of GA searches        
	groupGA_stopTime = time.clock() #ending time for group of 'expert GAs'
	timeNow = time.strftime("%Y%m%d_%H%M%S")

	#########  DML calling functions for eges
	edges, edgesUsage = calcEdgeUsage(cityCoords,gaPaths)
	drawEdgeUsage(cityCoords,edges, edgesUsage)
	##=========================================
  
	#Start "Wisdom of Crowds" Analysis
	crowd_startTime = time.clock()
	bestUnwPathCost,bestUnwExpertNum,bestUnwPath,bestExpPathCost,bestExpExpertNum,bestExpPath,bestPercPathCost,bestPercExpertNum,bestPercPath,percWeightOfExperts,expWeightOfExperts,lastUnwCost,lastExpCost,lastPercCost = buildWeightedWocHistograms(gaPaths,gaCosts,numOfCities)
	crowd_stopTime = time.clock() #ending time to perform WoC 
	bestGaCost = round(min(gaCosts),3)
	meanGaCost = round((sum(gaCosts)/len(gaCosts)),3)

	#Calculate ending Times
	groupGAtime = groupGA_stopTime - groupGA_startTime
	wisdomTime = crowd_stopTime - crowd_startTime
	#avgGaTime = round((sum(gaTimes)/len(gaTimes)),3)

	#******************* Data Logging *****************************************
	#**** All relevant data should be saved to file for analysis later ********
	print ("Avg GA = ",meanGaCost)
	print ("Best GA = ",round(bestGaCost,3))
	print ("Unweighted = ",round(bestUnwPathCost,3),bestUnwExpertNum)
	print ("Last Unweighted = ", round(lastUnwCost,3))
	print ("Exp Weighted = ",round(bestExpPathCost,3),bestExpExpertNum)
	print ("Last Exp = ", round(lastExpCost,3))
	print ("Percent Weighted = ",round(bestPercPathCost,3),bestPercExpertNum)
	print ("Last Percent = ", round(lastPercCost,3))
	#print ("Avg GA Agent Time = ",round(avgGaTime,3))
	print ("Group GA Time = ",round(groupGAtime,3))
	print ("WoC Time = ",round(wisdomTime,3))
        
	outFilename = 'Data/gaData_'+str(cityCount)+'Cities_'+timeNow+'.txt' #Build a filename with current date time to seconds accuracy
	with open(outFilename, 'w') as f: #store Cummualitive results for all GAs in a file.   
		outString = 'numOfCities:'+str(numOfCities)
		write_data = f.write(outString+'\n')  
		outString = 'gaPaths:'+str(gaPaths)
		write_data = f.write(outString+'\n')  
		outString = 'gaCosts:'+str(gaCosts)
		write_data = f.write(outString+'\n') 
		outString = 'genCosts:'+str(genCosts)
		write_data = f.write(outString+'\n')  
		outString = 'bestUnwPath:'+str(bestUnwPath)
		write_data = f.write(outString+'\n')  
		outString = 'bestExpPath:'+str(bestExpPath)
		write_data = f.write(outString+'\n')  
		outString = 'bestPercPath:'+str(bestPercPath)
		write_data = f.write(outString+'\n')  
		outString = 'expWeightOfExperts:'+str(expWeightOfExperts)
		write_data = f.write(outString+'\n')  
		outString = 'percWeightOfExperts:'+str(percWeightOfExperts)
		write_data = f.write(outString+'\n')  
		outString = 'cityCoords:'+str(cityCoords)
		write_data = f.write(outString+'\n')        
		outString = 'edges:'+str(edges)
		write_data = f.write(outString+'\n')        
		outString = 'edgesUsage:'+str(edgesUsage)
		write_data = f.write(outString+'\n')        
		outString = 'Avg GA:'+str(meanGaCost)
		write_data = f.write(outString+'\n')        
		outString = 'Best GA:'+str(round(bestGaCost,3))
		write_data = f.write(outString+'\n')        
		outString = 'Unweighted:'+str(round(bestUnwPathCost,3))+","+str(bestUnwExpertNum)
		write_data = f.write(outString+'\n')        
		outString = 'Last Unweighted:'+str(round(lastUnwCost,3))
		write_data = f.write(outString+'\n')        
		outString = 'Exp Weighted:'+str(round(bestExpPathCost,3))+","+str(bestExpExpertNum)
		write_data = f.write(outString+'\n')        
		outString = 'Last Exp:'+str(round(lastExpCost,3))
		write_data = f.write(outString+'\n')         
		outString = 'Percent Weighted:'+str(round(bestPercPathCost,3))+","+str(bestPercExpertNum)
		write_data = f.write(outString+'\n')         
		outString = 'Last Percent:'+str(round(lastPercCost,3))
		write_data = f.write(outString+'\n')
		#outString = 'Avg GA Agent Time:'+str(round(avgGaTime,3))
		#write_data = f.write(outString+'\n')  
		outString = 'Group GA Time:'+str(round(groupGAtime,3))
		write_data = f.write(outString+'\n')  
		outString = 'WoC Time:'+str(round(wisdomTime,3))
		write_data = f.write(outString+'\n')  
	#prepare plots
	genPlots(cityCoords,numOfCities,gaPaths,gaCosts,genCosts,bestUnwPath,bestExpPath,bestPercPath,expWeightOfExperts,percWeightOfExperts,bestUnwPathCost,bestExpPathCost,bestPercPathCost) 
	plt.show()

