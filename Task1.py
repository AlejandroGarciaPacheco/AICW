import numpy as np
import matplotlib.pyplot as plt

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
        self.grid = map = np.random.randint(0,9, size = (width, height))
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
        
    def dijakstra():
        
        return

    #HAVE A FUNCTION THAT SUGGESTS A BETTER ALTERNATIVE
        
        


rg = RobotGame()
rg.setup()
rg.makeGrid()
rg.getGrid()[4][4]
rg.myAlgorithm()