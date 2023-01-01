import numpy as np
import matplotlib.pyplot as plt

#from queue import PriorityQueue
import heapq
import time

class Game :
    
    
    
    
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
        print("We have  3 modes to allocate each number to the cell")
        print("1) randomly allocates numbers to every cell")
        print("2) you are given the choice to give each number the probability of a number to appear on the cells. All numbers have to add up to 1, otherwise it wont work. e.g 0.1")
        
        number = int(input("Enter a number: "))
        #creates a grid with random values between 0 and 9 and gets 
        while not(number==1 or number==2):
            number = int(input("Enter a number: "))
        
        if (number==1):
            self.grid = map = np.random.randint(0,10, size = (width, height))
            return self.grid
        if (number==2):
            print("enter a number on the probability for each number as percentages: ")
            zero=float(input("0: "))
            one=float(input("1: "))
            two=float(input("2: "))
            three=float(input("3: "))
            four=float(input("4: "))
            five=float(input("5: "))
            six=float(input("6: "))
            seven=float(input("7: "))
            eight=float(input("8: "))
            nine = 1-(zero+one+two+three+four+five+six+seven+eight)
            self.grid = np.random.choice([0,1,2,3,4,5,6,7,8,9],(width,height),p=[zero, one, two, three, four, five, six, seven, eight, nine])
            #print(np.count_nonzero(self.grid))
            return self.grid
    
    
    def getGrid(self):
        return self.grid
        #print(map[5][5])
        
    def startingPosition(self):
        self.posx = 0
        self.posy = 0
        
    def meanPath(self):
        grid = self.getGrid()
        return np.mean(grid)
    
    
        
        

    def myAlgorithm(self):
        #start = time.time()
        self.startingPosition()
        currentGrid = self.getGrid()
        width = self.gridWidth-1
        height = self.gridHeight-1
        
        m = len(currentGrid)
        n = len(currentGrid[0])
        #newPath = np.array(m)
        newArray = np.copy(currentGrid)
        #print (newArray)
        
        #CALCULATES PATH FROM TOP LEFT TO BOTTOM RIGHT
        
        
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
                
        #end = time.time()
        #print(f"Solved grid with size {m*n} in {1000*(end - start):.3f}ms")
        print (currentGrid)
        
        
        return newArray
            

    #Code for dijktra's algorithm
    def dijkstra(self):
        #startT = time.time()
        width = self.gridWidth
        height = self.gridHeight
        grid = self.getGrid()
        visited=np.zeros((width,height),dtype=bool)
        distance = []
        
        print ("enter values >=0 and keep in mind values entered cannot be the same as values entered for grid size")
        x= int(input("enter the x starting position "))
        y= int(input("enter the y starting position "))
        #x goal
        xg = int(input("Enter the x finishing position "))
        #y goal
        yg = int(input("Enter the y finishing position "))
        
        #print("The mean path of this grid is: ", np.mean(grid))
        #print("The average path of this grid is: ", np.average(grid))
        
        while ((x or y or xg or yg) < 0):
            print ("enter values >=0")
            x= int(input("enter the x starting position "))
            y= int(input("enter the y starting position "))
            #x goal
            xg = int(input("Enter the x finishing position "))
            #y goal
            yg = int(input("Enter the y finishing position "))
        
        while ((x or xg) >= width) or ((y or yg) >= height):
            print ("enter values within grid size")
            x= int(input("enter the x starting position "))
            y= int(input("enter the y starting position "))
            
            xg = int(input("Enter the x finishing position "))
            
            yg = int(input("Enter the y finishing position "))
        

        #print ("current value " , x, y, " ",grid[x][y], ", to: ", grid[xg][yg])
        
        #x,y=np.int(0),np.int(0)
        count = 0
        
        #List with x and y directions
        drow = [-1, 1, 0, 0]
        dcol = [0, 0, -1, 1]
        print (grid)
        
        queue = [(grid[x][y], x, y)]
        while queue:
            count+=1
            
            dist, x, y = heapq.heappop(queue)
            distance.append(dist)
      
            if visited[x][y]:
                continue
            visited[x][y] = True
            
            if x==xg and y==yg:
                break
                #return dist
            
            for i in range(len(drow)):
              ni = x + drow[i]
              nj = y + dcol[i]
              if 0 <= ni < width and 0 <= nj < height:
                  heapq.heappush(queue, (dist + grid[ni][nj], ni, nj))
        #endT = time.time()   
        #print(f"Solved grid with size {width*height} in {1000*(endT - startT):.3f}s")
        print("The shortest path length is: ", dist)
        print("The mean value is: ", np.mean(grid))
        
        
        
        return dist, distance
    
while (True):
    
    print("Welcome...")
    print("To initialize the game follow the instruction in the console")
    rg = Game()
    rg.setup()
    rg.makeGrid()
    print("We have two game modes: 1) heuristic algorithm and 2) uses Dijakstras algorithm using priority queues")
    print("Choose you game mode: ")
    number = int(input("Enter a number between 1 and 2\n"))
    while not((number == 1) or (number == 2)):
        number = int(input("Enter a number between 1 and 2\n"))
        
        
    if (number == 1):
        width = rg.gridWidth-1
        height = rg.gridHeight-1
        
        startTime = time.time()
        result = rg.myAlgorithm()        
        endTime = time.time()
        length= result[width][height]
        
        print(f"Solved grid with size {rg.gridWidth*rg.gridHeight} in {1000*(endTime - startTime):.5f}ms")
        print("The path length of this algorithm is: ", length)
        print("Plotting Graph \n ")
        mean = rg.meanPath()
        print("The mean path is: ", mean)
        
        size = np.arange(0, rg.gridWidth*rg.gridHeight)
        y = result.flatten()
        plt.title("heuristic Algorithm")
        plt.xlabel("size")
        plt.ylabel("length")
        plt.plot(size, y, color ="green")
        plt.show()
        
        
        #print(timeit.Timer('for i in xrange(10): oct(i)').timeit())
        #print(p)
    
    if (number == 2):
        startT = time.time()
        length, distance = rg.dijkstra()
        endT = time.time()
        
        print(f"Solved grid with size {rg.gridWidth*rg.gridHeight} in {1000*(endT - startT):.5f}ms")
        print("Plotting Graph \n ")
        mean = rg.meanPath()
        #n = (rg.gridWidth*rg.gridHeight)
        size = np.arange(0, len(distance))
        y = distance
        plt.title("Dijakstra Algorithm")
        plt.xlabel("size")
        plt.ylabel("length")
        plt.plot(size, y, color ="green")
        plt.show()
        
        #length = rg.dijkstra()
        
        

    


