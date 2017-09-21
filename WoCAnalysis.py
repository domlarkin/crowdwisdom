#!/usr/bin/python
import sys





#******************* Start - Main *****************************************
if __name__ == "__main__":
    filename = 'gaData_22Cities_20170907_233001.txt'
    cityCount = 0
    if (len(sys.argv)>1):
        filename = sys.argv[1]
    fileName = 'Data/'+filename
    print("Using filename: "+fileName)
    
    ''' numOfCities              
            gaPaths              
            gaCosts             
            genCosts              
            bestUnwPath              
            bestExpPath              
            bestPercPath              
            expWeightOfExperts              
            percWeightOfExperts              
            cityCoords                    
            edges                    
            edgesUsage                    
            meanGaCost                    
            bestGaCost                    
            bestUnwPathCost,bestUnwExpertNum                    
            lastUnwCost                    
            bestExpPathCost,bestExpExpertNum                    
            lastExpCost                     
            bestPercPathCost,bestPercExpertNum                     
            lastPercCost            
            avgGaTime              
            groupGAtime              
            wisdomTime'''    
    
    with open(fileName, 'r') as f: #store Cummualitive results for all GAs in a file.   
            numOfCities=f.readline().split(":")[1]
            print(numOfCities)
            gaPaths=f.readline().split(":")[1]
            #print(gaPaths)                         
            gaCosts=f.readline().split(":")[1] 
            print(gaCosts)            
            genCosts=f.readline().split(":")[1]
            bestUnwPath=f.readline().split(":")[1]
            bestExpPath=f.readline().split(":")[1]
            bestPercPath=f.readline().split(":")[1]
            expWeightOfExperts=f.readline().split(":")[1]
            percWeightOfExperts=f.readline().split(":")[1]
            cityCoords=f.readline().split(":")[1]
            edges=f.readline().split(":")[1]
            edgesUsage=f.readline().split(":")[1]
            meanGaCost=f.readline().split(":")[1]
            bestGaCost=f.readline().split(":")[1]
            #bestUnwPathCost,
            bestUnwExpertNum=f.readline().split(":")[1]
            print(bestUnwExpertNum)     
            lastUnwCost=f.readline().split(":")[1]
            bestExpPathCost,bestExpExpertNum                    
            lastExpCost=f.readline().split(":")[1]
            bestPercPathCost,bestPercExpertNum=f.readline().split(":")[1]
            lastPercCost=f.readline().split(":")[1]
            avgGaTime=f.readline().split(":")[1]
            groupGAtime=f.readline().split(":")[1]
            wisdomTime=f.readline().split(":")[1]
