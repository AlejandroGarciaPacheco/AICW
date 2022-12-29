import numpy as np
import matplotlib.pyplot as plt
import math
from queue import PriorityQueue
import heapq

class RobotGame :
    
    
    def setup(self):
        
        self.gridHeight = int(input("\nEnter the height of the grid? "))
        self.gridWidth = int(input("\nEnter the width of the grid? "))    
    
        self.posx = 0
        self.posy = 0
        self.robot_moves = []
        self.goal = (self.gridWidth-1,self.gridHeight-1)
        
    def makeGrid(self):
        width = self.gridWidth
        height = self.gridHeight
        #creates a grid with random values between 0 and 9 and gets 
        self.grid = map = np.random.randint(0,10, size = (width, height))
        return self.grid
    
    
    def getGrid(self):
        return self.grid
        #print(map[5][5])
        
    def startingPosition(self):
        self.posx = 0
        self.posy = 0
        

    def myAlgorithm(self):
        self.startingPosition()
        currentGrid = self.getGrid()
        width = self.gridWidth-1
        height = self.gridHeight-1
        
        m = len(currentGrid)
        n = len(currentGrid[0])
        #newPath = np.array(m)
        newArray = np.copy(currentGrid)
        #print (newArray)
        
        #print(newArray)
        #for i in range(m):
         #   newPath[i] = np.array(n)
        #newPath[0][0] = currentGrid[0][0]
        
        #Calculates values in x-axis at y =0 grid 
        for i in range (1, m):
            newArray[i][0] = newArray[i-1][0] + currentGrid[i][0]
            #print(newArray)
        
        #Calculates values in y axis at x = 0 in grid
        for i in range (1, n):
            newArray[0][i] = newArray[0][i-1] + currentGrid[0][i]
            #print(newArray)
            
        for i in range (1, m):
            for j in range (1, n):
                newArray[i][j] = min(newArray[i-1][j], newArray[i][j-1]) + currentGrid[i][j]
                
        #print(currentGrid)
            
        return newArray[m-1][n-1]
            
            
        #print (newPath[0][0], "and ", currentGrid[0][0])
        
        
    
    #Code for dijktra's algorithm
    def dijkstra(self):
        x= input("enter the x starting position \n")
        y= input("enter the y starting position \n")
        #x goal
        xg = input("Enter the x finishing position \n")
        yg = input("Enter the y finishing position \n")
        
        width = self.gridWidth
        height = self.gridHeight
        grid = self.getGrid()
        visited=np.zeros((width,height),dtype=bool)
        
        x,y=np.int(0),np.int(0)
        count = 0
        
        #List with x and y directions
        drow = [-1, 1, 0, 0]
        dcol = [0, 0, -1, 1]
        print (grid)
        
        queue = [(grid[0][0], 0, 0)]
        while queue:
            count+=1
            
            dist, x, y = heapq.heappop(queue)
      
            if visited[x][y]:
                continue
            visited[x][y] = True
            
            if x==width-1 and y==height-1:
                return dist
            
            for i in range(len(drow)):
              ni = x + drow[i]
              nj = y + dcol[i]
              if 0 <= ni < width and 0 <= nj < height:
                  heapq.heappush(queue, (dist + grid[ni][nj], ni, nj))   
        
        return dist 
        
        


rg = RobotGame()
rg.setup()
rg.makeGrid()
print(rg.myAlgorithm())
print("the length is: ", rg.dijkstra())

'''
cost=np.ones((self.gridWidth, self.gridHeight),dtype=int)*np.Infinity

cost[0][0] = 0


#originmap=np.ones((max_val,max_val),dtype=int)*np.nan
visited=np.zeros((self.gridWidth,self.gridHeight),dtype=bool)
finished = False
x, y = self.posx, self.posy

count=0

while not finished:
  # move to x+1,y
      if x < width:
            if cost[x+1,y]>currentGrid[x+1,y]+cost[x,y] and not visited[x+1,y]:
                cost[x+1,y]=currentGrid[x+1,y]+cost[x,y]
              
      # move to x-1,y
      if x>0:
            if cost[x-1,y]>currentGrid[x-1,y]+cost[x,y] and not visited[x-1,y]:
                cost[x-1,y]=currentGrid[x-1,y]+cost[x,y]
              
      # move to x,y+1
      if y < height:
            if cost[x,y+1]>currentGrid[x,y+1]+cost[x,y] and not visited[x,y+1]:
                cost[x,y+1]=currentGrid[x,y+1]+cost[x,y]
              
      # move to x,y-1
      if y>0:
            if cost[x,y-1]>currentGrid[x,y-1]+cost[x,y] and not visited[x,y-1]:
                cost[x,y-1]=currentGrid[x,y-1]+cost[x,y]
              
        
      print("x: ", x , "y: ", y)
      visited[x,y]=True
      temp=cost
      temp[np.where(visited)]=np.Infinity
      # now we find the shortest path so far
      minval=np.unravel_index(np.argmin(temp),np.shape(temp))
      x,y=minval[0],minval[1]
      
      if x==width and y==height:
        finished=True
      count=count+1

print('The path length is: '+np.str(cost[width,height]))
print("The count to find the shortest path is: ", count)
print('The dump/mean path should have been: '+np.str(9*((width + height)/2)))
'''

