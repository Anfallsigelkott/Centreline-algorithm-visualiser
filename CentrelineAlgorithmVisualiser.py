import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import ezdxf
from scipy.spatial import Voronoi, voronoi_plot_2d
import math
from tkinter import filedialog
from tkinter import simpledialog
from VoronoiAlgorithm import VoronoiAlgorithm
import time

class CentrelineAlgorithmVisualiser:
    def __init__(self):
        self.figure, self.axis = plt.subplots()
        self.canvas = self.figure.canvas
        
        self.axis.set_title("Centreline Algorithm Visualiser")
        self.axis.set_xlabel("n = next/new polyline, b = previous polyline, u = undo, t = save test case, r = read file, v = display Voronoi, l = clear diagram, d = convert dxf to polyline format, a = run algorithm, c = automatic closing lines, y = even out polylines, lm = polyline, rm = closing line, g = load nodegraph, p = clear nodegraph")

        self.axis.set_xlim(0, 100)
        self.axis.set_ylim(0, 100)

        self.polylineIndex = 0
        self.polyLines = []
        self.polylineVerticesX = []
        self.polylineVerticesY = []

        self.storedPolylineIndex = 0
        self.storedPolyLines = []
        self.storedPolylineVerticesX = []
        self.storedPolylineVerticesY = []

        self.centrelineNodes = {}
        self.centrelinePlots = []

        self.loadedNodegraphNodes = {}
        self.loadedNodegraphPlots = []

        self.closingLines = []
        self.closingLineStartEnds = [] # I've done these lines in three different ways... Oh well, this is only a prototyping tool

        self.infiniteLines = []

        self.canvas.mpl_connect('draw_event', self.onDraw)
        self.canvas.mpl_connect('button_press_event', self.onPressedButton)
        self.canvas.mpl_connect('key_press_event', self.onPressedKey)
        self.canvas.mpl_connect('scroll_event', self.scrollbarZoom)

        self.voronoiAlgorithm = VoronoiAlgorithm(6, 4, math.pi/4) # Algorithm input parameters adjusted here

        plt.subplots_adjust(left=0.04, bottom=0.06, right=0.97, top=0.94)
        plt.show()


    def addVertex(self, x, y):
        if (self.polylineIndex == len(self.polyLines)):
            self.newPolyline(x, y)
            return
        self.polylineVerticesX[self.polylineIndex].append(x)
        self.polylineVerticesY[self.polylineIndex].append(y)
        self.polyLines[self.polylineIndex] = self.axis.plot(self.polylineVerticesX[self.polylineIndex], self.polylineVerticesY[self.polylineIndex], marker='o', markerfacecolor='r', color='black', animated=True)

    def newPolyline(self, x, y):
        self.polylineVerticesX.append([x])
        self.polylineVerticesY.append([y])
        self.polyLines.append(self.axis.plot(self.polylineVerticesX[self.polylineIndex], self.polylineVerticesY[self.polylineIndex], marker='o', markerfacecolor='r', color='black', animated=True))

    def incrementPolylineIndex(self):
        if self.polylineIndex == len(self.polyLines):
            return
        self.polylineIndex = self.polylineIndex + 1

    def decrementPolylineIndex(self):
        if self.polylineIndex == 0:
            return
        self.polylineIndex = self.polylineIndex - 1

    def undo(self):
        if len(self.polylineVerticesX[self.polylineIndex]) == 0:
            return
        self.polylineVerticesX[self.polylineIndex].pop()
        self.polylineVerticesY[self.polylineIndex].pop()
        self.polyLines[self.polylineIndex] = self.axis.plot(self.polylineVerticesX[self.polylineIndex], self.polylineVerticesY[self.polylineIndex], marker='o', markerfacecolor='r', color='black', animated=True)
        self.canvas.draw()

    def onDraw(self, event):
        self.background = self.canvas.copy_from_bbox(self.axis.bbox)
        for polyLine in self.polyLines:
            self.axis.draw_artist(polyLine[0])
        for centreline in self.centrelinePlots:
            self.axis.draw_artist(centreline[0])
        for closingLine in self.closingLines:
            self.axis.draw_artist(closingLine[0])
        for infiniteLine in self.infiniteLines:
            self.axis.draw_artist(infiniteLine[0])
        for loadedNodegraphline in self.loadedNodegraphPlots:
            self.axis.draw_artist(loadedNodegraphline[0])

    def onPressedButton(self, event):
        if (event.inaxes is None or (event.button != MouseButton.LEFT and event.button != MouseButton.RIGHT)):
            return
        if event.button == MouseButton.LEFT:
            self.addVertex(event.xdata, event.ydata)
        elif event.button == MouseButton.RIGHT:
            if len(self.closingLineStartEnds) == 0:
                self.closingLineStartEnds.append([[event.xdata, event.ydata]])
            elif len(self.closingLineStartEnds[-1]) == 1:
                self.closingLineStartEnds[-1].append([event.xdata, event.ydata])
                self.closingLines.append(self.axis.plot([self.closingLineStartEnds[-1][0][0], self.closingLineStartEnds[-1][1][0]], [self.closingLineStartEnds[-1][0][1], self.closingLineStartEnds[-1][1][1]], color='gold', linewidth = 4, animated=True))
            else:
                self.closingLineStartEnds.append([[event.xdata, event.ydata]])

        self.canvas.draw()

    def onPressedKey(self, event):
        if event.key == 'n':
            self.incrementPolylineIndex()
        if event.key == 'b':
            self.decrementPolylineIndex()
        if event.key == 'u':
            self.undo()
        if event.key == 'v':
            self.displayVoronoi()
        if event.key == 't':
            self.saveTestcase()
        if event.key == 'r':
            self.readPolylineFile()
        if event.key == 'l':
            self.clearDiagram()
        if event.key == 'd':
            self.convertDXFToPolylineFile()
        if event.key == 'a':
            self.runVoronoiAlgorithm()
        if event.key == 'c':
            self.automaticClosingLineCreation()
        if event.key == 'y':
            self.evenOutPolylines()
        if event.key == '1':
            self.demoStep1()
        if event.key == '2':
            self.demoStep2()
        if event.key == '3':
            self.demoStep3()
        if event.key == '4':
            self.demoStep4()
        if event.key == '5':
            self.demoStep5()
        if event.key == '6':
            self.demoStep6()
        if event.key == 'g':
            self.loadNodegraph()
        if event.key == 'p':
            self.loadedNodegraphPlots.clear()
            self.canvas.draw()

    def readPolylineFile(self):
        input = list[str]()
        with open(filedialog.askopenfilename(), "r") as file:
            for row in file:
                input.append(row)
        
        xmin = math.inf
        xmax = -math.inf
        for row in input:
            xys = row.split(";")
            for xy in xys:
                self.addVertex(float(xy.split(",")[0]), float(xy.split(",")[1]))
                if float(xy.split(",")[0]) < xmin:
                    xmin = float(xy.split(",")[0])
                if float(xy.split(",")[0]) > xmax:
                    xmax = float(xy.split(",")[0])
            self.incrementPolylineIndex()
        
        self.axis.set_xlim(xmin, xmax)
        self.axis.set_ylim(xmin, xmax)
        self.canvas.draw()

    def convertDXFToPolylineFile(self):
        dxfFile = ezdxf.readfile(filedialog.askopenfilename()) # Prints a lot of "Found non-unique entity handle #XXXXX, data validation is required." but still seems to work
        outputFile = open(simpledialog.askstring("Output Filename", "insert name of file to be created here"), "w")

        modelspace = dxfFile.modelspace()
        for polyline in modelspace.query("POLYLINE"):
            polylineRow = ""
            vertices = polyline.points()
            for vertex in vertices:
               polylineRow = polylineRow + str(vertex.x) + ", " + str(vertex.y) + "; " # The axis of these may sometimes have to be changed depending on the DXF file (x + z instead of x + y for example)
            polylineRow = polylineRow[:-2] + "\n"
            outputFile.write(polylineRow)
        
        outputFile.close()

    def addCentrelineNode(self, nodeID: int, x: float, y: float, connections: list[int]):
        self.centrelineNodes[nodeID] = [x, y, connections]

    def displayCentreline(self):
        lines = {}
        for centrelineNodeID, centrelineNode in self.centrelineNodes.items():
            for nodeID in centrelineNode[2]:
                if str(centrelineNodeID) + ":" + str(nodeID) in lines or str(nodeID) + ":" + str(centrelineNodeID) in lines:
                    continue
                lines[str(centrelineNodeID) + ":" + str(nodeID)] = [[centrelineNode[0], self.centrelineNodes[nodeID][0]], [centrelineNode[1], self.centrelineNodes[nodeID][1]]]
        
        for _,line in lines.items():
            self.centrelinePlots.append(self.axis.plot(line[0], line[1],  marker='o', color='red', animated=True))

        self.canvas.draw()
    
    def displayVoronoi(self):
        points = []
        for i in range(0, len(self.polylineVerticesX)):
            for j in range(0, len(self.polylineVerticesX[i])):
                points.append([self.polylineVerticesX[i][j], self.polylineVerticesY[i][j]])
        voronoi = Voronoi(points)

        voronoi_plot_2d(voronoi, self.axis, line_colors='purple', line_alpha=0.7)
        self.canvas.draw()

    def saveTestcase(self):
        outputFile = open(simpledialog.askstring("Output Filename", "insert name of file to be created here"), "w")

        for i in range(0, len(self.polylineVerticesX)):
            polylineRow = ""
            for j in range(0, len(self.polylineVerticesX[i])):
                polylineRow = polylineRow + str(self.polylineVerticesX[i][j]) + ", " + str(self.polylineVerticesY[i][j]) + "; "
            polylineRow = polylineRow[:-2] + "\n"
            outputFile.write(polylineRow)

        outputFile.close()

    def clearDiagram(self):
        self.axis.cla()

        self.axis.set_title("Centreline Algorithm Visualiser")
        self.axis.set_xlabel("n = next/new polyline, b = previous polyline, u = undo, t = save test case, r = read file, v = display Voronoi, l = clear diagram, d = convert dxf to polyline format, a = run algorithm, c = automatic closing lines, y = even out polylines, lm = polyline, rm = closing line, g = load nodegraph, p = clear nodegraph")

        self.axis.set_xlim(0, 100)
        self.axis.set_ylim(0, 100)

        self.polylineIndex = 0
        self.polyLines = []
        self.polylineVerticesX = []
        self.polylineVerticesY = []

        self.centrelineNodes = {}
        self.centrelinePlots = []
        self.infiniteLines = []

        self.closingLines = []
        self.closingLineStartEnds = []

        self.storedPolylineIndex = 0
        self.storedPolyLines = []
        self.storedPolylineVerticesX = []
        self.storedPolylineVerticesY = []

        self.canvas.draw()

    def scrollbarZoom(self, event):
        scaleFactor = 2
        currentRangeX = (self.axis.get_xlim()[1] - self.axis.get_xlim()[0])/2
        currentRangeY = (self.axis.get_ylim()[1] - self.axis.get_ylim()[0])/2
        if event.button == 'up':
            self.axis.set_xlim(event.xdata - currentRangeX*(1/scaleFactor), event.xdata + currentRangeX*(1/scaleFactor))
            self.axis.set_ylim(event.ydata - currentRangeY*(1/scaleFactor), event.ydata + currentRangeY*(1/scaleFactor))
        if event.button == 'down':
            self.axis.set_xlim(event.xdata - currentRangeX*scaleFactor, event.xdata + currentRangeX*scaleFactor)
            self.axis.set_ylim(event.ydata - currentRangeY*scaleFactor, event.ydata + currentRangeY*scaleFactor)
        
        self.canvas.draw()

    def runVoronoiAlgorithm(self):
        self.centrelineNodes = {}
        self.centrelinePlots = []
        self.infiniteLines = []

        self.voronoiAlgorithm.setPolylines(self.polylineVerticesX, self.polylineVerticesY)
        self.voronoiAlgorithm.setClosingLines(self.closingLineStartEnds)
        timestamp = time.time()
        centrelines, infiniteLines = self.voronoiAlgorithm.calculateCentreline()
        print("Time:", time.time() - timestamp)
        for centreline in centrelines:
            self.addCentrelineNode(centreline[0], centreline[1], centreline[2], centreline[3])
        for infiniteLine in infiniteLines:
            self.infiniteLines.append(self.axis.plot([infiniteLine[0][0], infiniteLine[1][0]], [infiniteLine[0][1], infiniteLine[1][1]], color='red', linestyle='dashed', animated=True))
        self.displayCentreline()
        print("Voronoi Algorithm Completed")

    def automaticClosingLineCreation(self):
        timestamp = time.time()
        maxClosingDistance = 25.0
        points = []
        for i in range(0, len(self.polylineVerticesX)):
            points.append([self.polylineVerticesX[i][0], self.polylineVerticesY[i][0]])
            if len(self.polylineVerticesX[i]) > 1:
                points.append([self.polylineVerticesX[i][-1], self.polylineVerticesY[i][-1]])

        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def closestPair(points):
            if len(points) < 2:
                return ([0, 0], [0, 0]), -1

            points.sort()

            def closestPairRecursion(start, end):
                if end - start <= 3:
                    minDist = math.inf
                    closestPoints = None
                    for i in range(start, end):
                        for j in range(i+1, end):
                            d = distance(points[i], points[j])
                            if d < minDist:
                                minDist = d
                                closestPoints = (points[i], points[j])
                    return closestPoints, minDist
                
                mid = (start + end) // 2
                leftPair, leftDist = closestPairRecursion(start, mid)
                rightPair, rightDist = closestPairRecursion(mid, end)

                if leftDist < rightDist:
                    minDist = leftDist
                    closestPoints = leftPair
                else:
                    minDist = rightDist
                    closestPoints = rightPair
                
                midX = points[mid][0]
                strip = [p for p in points[start:end] if abs(p[0] - midX) < minDist]
                strip.sort(key = lambda p: p[1])

                for i in range(len(strip)):
                    for j in range(i + 1, min(i + 7, len(strip))):
                        d = distance(strip[i], strip[j])
                        if d < minDist:
                            minDist = d
                            closestPoints = (strip[i], strip[j])
                
                return closestPoints, minDist
            
            return closestPairRecursion(0, len(points))
        
        while True:
            print("Closing lines loading, max # points left:", len(points))
            newClosingLinePoints, newClosingLineLength = closestPair(points)
            if newClosingLineLength > maxClosingDistance or newClosingLineLength == -1:
                break
            self.closingLineStartEnds.append(newClosingLinePoints)
            self.closingLines.append(self.axis.plot([self.closingLineStartEnds[-1][0][0], self.closingLineStartEnds[-1][1][0]], [self.closingLineStartEnds[-1][0][1], self.closingLineStartEnds[-1][1][1]], color='gold', linewidth = 4, animated=True))
            points.remove(newClosingLinePoints[0])
            points.remove(newClosingLinePoints[1])
        print("Time:", time.time() - timestamp)
        self.canvas.draw()

    def cutOutTestCasePart(self):
        print("Cutting out part")
        minX = -2500 # Adjust these values manually for the X and Y bounds of the area to cut out
        minY = 1416
        maxX = 120
        maxY = 3000
        newPolylineVerticesX = []
        newPolylineVerticesY = []
        for i in range(0, len(self.polylineVerticesX)):
            print("Loading cut: ", i, "/", len(self.polylineVerticesX))
            newPolylineVerticesX.append([])
            newPolylineVerticesY.append([])
            for j in range(0, len(self.polylineVerticesX[i])):
                if minX < self.polylineVerticesX[i][j] < maxX and minY < self.polylineVerticesY[i][j] < maxY:
                    newPolylineVerticesX[i].append(self.polylineVerticesX[i][j])
                    newPolylineVerticesY[i].append(self.polylineVerticesY[i][j])
        
        self.polylineIndex = 0
        self.polyLines = []
        self.polylineVerticesX = []
        self.polylineVerticesY = []
        for i in range(0, len(newPolylineVerticesX)):
            for j in range(0, len(newPolylineVerticesX[i])):
                self.addVertex(newPolylineVerticesX[i][j], newPolylineVerticesY[i][j])
            self.incrementPolylineIndex()

        self.canvas.draw()

    # Lerps new points between every polyline point when the distance between points is too high
    def evenOutPolylines(self):
        def distance(x1, y1, x2, y2):
            return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        def lerp2D(x1, y1, x2, y2, f):
            x = (x1 * (1.0 - f)) + (x2 * f)
            y = (y1 * (1.0 - f)) + (y2 * f)
            return x, y
        
        timestamp = time.time()
        # Allows toggling between evened and unevened versions of the mine map
        if len(self.storedPolyLines) > 0:
            self.polylineIndex = self.storedPolylineIndex
            self.polyLines = self.storedPolyLines[:]
            self.polylineVerticesX = self.storedPolylineVerticesX[:]
            self.polylineVerticesY = self.storedPolylineVerticesY[:]
            self.storedPolylineIndex = 0
            self.storedPolyLines = []
            self.storedPolylineVerticesX = []
            self.storedPolylineVerticesY = []

            self.canvas.draw()
            return
        
        self.storedPolylineIndex = self.polylineIndex
        self.storedPolyLines = self.polyLines[:]
        self.storedPolylineVerticesX = self.polylineVerticesX[:]
        self.storedPolylineVerticesY = self.polylineVerticesY[:]

        minPointDistance = 1 # 1 meter seems to work well, lower and higher both gave worse results
        newPolylineVerticesX = []
        newPolylineVerticesY = []

        for i in range(0, len(self.polylineVerticesX)):
            print("Evening out polyline: ", i, "/", len(self.polylineVerticesX))
            newPolylineVerticesX.append([])
            newPolylineVerticesY.append([])
            for j in range(0, len(self.polylineVerticesX[i])-1):
                newPolylineVerticesX[i].append(self.polylineVerticesX[i][j])
                newPolylineVerticesY[i].append(self.polylineVerticesY[i][j])
                dist = distance(self.polylineVerticesX[i][j], self.polylineVerticesY[i][j], self.polylineVerticesX[i][j+1], self.polylineVerticesY[i][j+1])
                if dist > minPointDistance:
                    pieces = round(dist / minPointDistance)
                    fraction = 1 / pieces
                    for pieceNum in range(1, pieces):
                        x, y = lerp2D(self.polylineVerticesX[i][j], self.polylineVerticesY[i][j], self.polylineVerticesX[i][j+1], self.polylineVerticesY[i][j+1], fraction * pieceNum)
                        newPolylineVerticesX[i].append(x)
                        newPolylineVerticesY[i].append(y)
            
            newPolylineVerticesX[i].append(self.polylineVerticesX[i][-1])
            newPolylineVerticesY[i].append(self.polylineVerticesY[i][-1])
        
        print("Time:", time.time() - timestamp)

        self.polylineIndex = 0
        self.polyLines = []
        self.polylineVerticesX = []
        self.polylineVerticesY = []
        for i in range(0, len(newPolylineVerticesX)):
            for j in range(0, len(newPolylineVerticesX[i])):
                self.addVertex(newPolylineVerticesX[i][j], newPolylineVerticesY[i][j])
            self.incrementPolylineIndex()

        self.canvas.draw()

    def loadNodegraph(self):
        input = ""
        with open(filedialog.askopenfilename(), "r") as file:
            for row in file:
                input = row

        nodes = input.split(";")
        for node in nodes:
            nodedata = node.split(",")
            self.loadedNodegraphNodes[nodedata[0]] = [float(nodedata[1]), float(nodedata[2]), nodedata[4:]]

        lines = {}
        for loadedGraphNodeID, loadedGraphNode in self.loadedNodegraphNodes.items():
            for nodeID in loadedGraphNode[2]:
                if str(loadedGraphNodeID) + ":" + str(nodeID) in lines or str(nodeID) + ":" + str(loadedGraphNodeID) in lines:
                    continue
                lines[str(loadedGraphNodeID) + ":" + str(nodeID)] = [[loadedGraphNode[0], self.loadedNodegraphNodes[nodeID][0]], [loadedGraphNode[1], self.loadedNodegraphNodes[nodeID][1]]]
        
        for _,line in lines.items():
            self.loadedNodegraphPlots.append(self.axis.plot(line[0], line[1],  marker='^', color='blue', linestyle='dashed', animated=True))

        self.canvas.draw()



    ##### FUNCTIONS BELOW ARE FOR DEMO PURPOSES #####
    def demoStep1(self):
        self.centrelineNodes = {}
        self.centrelinePlots = []
        self.infiniteLines = []

        self.voronoiAlgorithm.setPolylines(self.polylineVerticesX, self.polylineVerticesY)
        self.voronoiAlgorithm.setClosingLines(self.closingLineStartEnds)
        timestamp = time.time()
        centrelines, infiniteLines = self.voronoiAlgorithm.demoStep1()
        print("Time:", time.time() - timestamp)
        for centreline in centrelines:
            self.addCentrelineNode(centreline[0], centreline[1], centreline[2], centreline[3])
        for infiniteLine in infiniteLines:
            self.infiniteLines.append(self.axis.plot([infiniteLine[0][0], infiniteLine[1][0]], [infiniteLine[0][1], infiniteLine[1][1]], color='red', linestyle='dashed', animated=True))
        self.displayCentreline()
        print("Step 1 Completed")
       

    def demoStep2(self): # Can be run without running step 1
        self.centrelineNodes = {}
        self.centrelinePlots = []
        self.infiniteLines = []

        self.voronoiAlgorithm.setPolylines(self.polylineVerticesX, self.polylineVerticesY)
        self.voronoiAlgorithm.setClosingLines(self.closingLineStartEnds)
        timestamp = time.time()
        centrelines, infiniteLines = self.voronoiAlgorithm.demoStep2()
        print("Time:", time.time() - timestamp)
        for centreline in centrelines:
            self.addCentrelineNode(centreline[0], centreline[1], centreline[2], centreline[3])
        for infiniteLine in infiniteLines:
            self.infiniteLines.append(self.axis.plot([infiniteLine[0][0], infiniteLine[1][0]], [infiniteLine[0][1], infiniteLine[1][1]], color='red', linestyle='dashed', animated=True))
        self.displayCentreline()
        print("Step 2 Completed")

    def demoStep3(self):
        self.centrelineNodes = {}
        self.centrelinePlots = []

        timestamp = time.time()
        centrelines, _ = self.voronoiAlgorithm.demoStep3()
        print("Time:", time.time() - timestamp)
        for centreline in centrelines:
            self.addCentrelineNode(centreline[0], centreline[1], centreline[2], centreline[3])

        self.displayCentreline()
        print("Step 3 Completed")

    def demoStep4(self):
        self.centrelineNodes = {}
        self.centrelinePlots = []

        timestamp = time.time()
        centrelines, _ = self.voronoiAlgorithm.demoStep4()
        print("Time:", time.time() - timestamp)
        for centreline in centrelines:
            self.addCentrelineNode(centreline[0], centreline[1], centreline[2], centreline[3])

        self.displayCentreline()
        print("Step 4 Completed")

    def demoStep5(self):
        self.centrelineNodes = {}
        self.centrelinePlots = []

        timestamp = time.time()
        centrelines, _ = self.voronoiAlgorithm.demoStep5()
        print("Time:", time.time() - timestamp)
        for centreline in centrelines:
            self.addCentrelineNode(centreline[0], centreline[1], centreline[2], centreline[3])

        self.displayCentreline()
        print("Step 5 Completed")

    def demoStep6(self):
        self.centrelineNodes = {}
        self.centrelinePlots = []

        timestamp = time.time()
        centrelines, _ = self.voronoiAlgorithm.demoStep6()
        print("Time:", time.time() - timestamp)
        for centreline in centrelines:
            self.addCentrelineNode(centreline[0], centreline[1], centreline[2], centreline[3])

        self.displayCentreline()
        print("Step 6 Completed")
                            

algorithmVisualiser = CentrelineAlgorithmVisualiser()

