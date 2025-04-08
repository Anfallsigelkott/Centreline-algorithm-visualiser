import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy
import sys

class VoronoiAlgorithm:
    def __init__(self, minPathLength, minLengthBetweenNodes, threeWayCrossingToleranceRad):
        self.polylines = []
        self.points = []
        self.closingLines = []
        self.centrelines = []

        self.minPathLength = minPathLength
        self.minLengthBetweenNodes = minLengthBetweenNodes
        self.threeWayCrossingToleranceRad = threeWayCrossingToleranceRad

    def setPolylines(self, polylineXs, polylineYs):
        self.polylines = []
        for i in range(0, len(polylineXs)):
            self.polylines.append([])
            for j in range(0, len(polylineXs[i])):
                self.polylines[i].append([polylineXs[i][j], polylineYs[i][j]])

    def setClosingLines(self, closinglineStartEnds): # Special lines to close off segments, are only checked in the complex intersecting edges step (and could in the future have its intersections noted in some way to be used for reconstructing the map from segments)
        if len(closinglineStartEnds) == 0:
            self.closingLines = []
            return
        if len(closinglineStartEnds[-1]) < 2:
            closinglineStartEnds = closinglineStartEnds[:-1]
        self.closingLines = closinglineStartEnds

    def isCounterclockwise(self, pointA: list[float], pointB: list[float], pointC: list[float]):
        return (pointC[1]-pointA[1]) * (pointB[0]-pointA[0]) > (pointB[1]-pointA[1]) * (pointC[0]-pointA[0])
    
    def isIntersecting(self, line1Start: list[float], line1End: list[float], line2Start: list[float], line2End: list[float]): # This code doesn't handle colinearity but hopefully that won't be necessary
        return self.isCounterclockwise(line1Start, line2Start, line2End) != self.isCounterclockwise(line1End, line2Start, line2End) and self.isCounterclockwise(line1Start, line1End, line2Start) != self.isCounterclockwise(line1Start, line1End, line2End)
    
    def clearInfiniteEdges(self, voronoiEdges):
        cleanedEdges = []
        for edge in voronoiEdges:
            if edge[0] == -1 or edge[1] == -1:
                continue
            cleanedEdges.append(edge)

        return cleanedEdges
    
    def isNeighboringIndexInSamePolyline(self, index1, index2):
        index1Polylineindex = -1
        index2Polylineindex = -1

        polylineIndex = 0
        i = 0
        for polyline in self.polylines:
            if i + len(polyline) < index1 - 1:
                polylineIndex = polylineIndex + 1
                i = i + len(polyline)
                continue
            for polylineVertex in polyline:
                if i == index1:
                    index1Polylineindex = polylineIndex
                if i == index2:
                    index2Polylineindex = polylineIndex
                if index1Polylineindex != -1 and index2Polylineindex != -1:
                    return index1Polylineindex == index2Polylineindex
                i = i + 1
            polylineIndex = polylineIndex + 1

        return index1Polylineindex == index2Polylineindex


    # Loop through the polyline vertices that define the voronoi diagram and remove any voronoi edges that lie between it and its polyline vertex neighbors, removes a majority of the intersecting polylines
    # Not used since doing line sweep straight away is quicker.
    def removeSimpleIntersectingEdges(self, voronoiEdges, voronoiEdgepoints):
        remainingIndices = set(range(0, len(voronoiEdges)))
        for i in range(0, len(voronoiEdgepoints)):
            if voronoiEdgepoints[i][0] == voronoiEdgepoints[i][1]-1 or voronoiEdgepoints[i][0] == voronoiEdgepoints[i][1]+1:
                if self.isNeighboringIndexInSamePolyline(voronoiEdgepoints[i][0], voronoiEdgepoints[i][1]):
                    remainingIndices.remove(i)
            if i % 100 == 0:
                print("Loading Bar 1:", i, "/", len(voronoiEdgepoints))
        
        print("Remaining: ", remainingIndices)
        remainingEdges = []
        remainingEdgepoints = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
            remainingEdgepoints.append(voronoiEdgepoints[index])
        
        return remainingEdges, remainingEdgepoints
    
    def removeComplexIntersectingEdges(self, voronoiVertices, voronoiEdges): #Old algorithm, has been replaced by the line sweep below this for improved time complexity
        remainingIndices = set(range(0, len(voronoiEdges)))
        for i in range(0, len(voronoiEdges)):
            shouldBreak = False # I do not like this kind of construction...
            if voronoiEdges[i][0] == -1:
                continue
            for polyline in self.polylines:
                if shouldBreak:
                    break
                for j in range(0, len(polyline)-1):
                    if self.isIntersecting(voronoiVertices[voronoiEdges[i][0]], voronoiVertices[voronoiEdges[i][1]], polyline[j], polyline[j+1]):
                        remainingIndices.remove(i)
                        shouldBreak = True
                        break
            if i % 100 == 0:
                print("Loading Bar 2:", i, "/", len(voronoiEdges))

        remainingEdges = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        
        return remainingEdges
    
    def removeComplexIntersectingEdgesLineSweep(self, voronoiVertices, voronoiEdges, voronoiEdgepoints): # Went from around 11 hours to under 2 minutes for this step on Boliden-Kriberg, very much worth it
        #Setup for the infinite edge calculation
        center = numpy.mean(self.points, axis = 0)
        minmaxPointLocations = numpy.ptp(self.points, axis = 0)
        events = []
        edgeIndexVertexMapping = {} # This construction is made neccessary by the infinite edges
        i = 0
        for voronoiEdge in voronoiEdges:
            if voronoiEdge[0] == -1:
                # Finds the tangent and normal
                t = [self.points[voronoiEdgepoints[i][1]][0] - self.points[voronoiEdgepoints[i][0]][0], self.points[voronoiEdgepoints[i][1]][1] - self.points[voronoiEdgepoints[i][0]][1]]
                t = t / numpy.linalg.norm(t)
                n = numpy.array([-t[1], t[0]])
                midpoint = numpy.mean([self.points[voronoiEdgepoints[i][0]], self.points[voronoiEdgepoints[i][1]]], axis = 0)
                subMidpointCenter = [midpoint[0] - center[0], midpoint[1] - center[1]]
                direction = numpy.sign(numpy.dot(subMidpointCenter, n)) * n
                infinitePoint = [voronoiVertices[voronoiEdge[1]][0] + direction[0] * numpy.max(minmaxPointLocations[0]), voronoiVertices[voronoiEdge[1]][1] + direction[1] * numpy.max(minmaxPointLocations[1])]

                if infinitePoint[0] >= voronoiVertices[voronoiEdge[1]][0]:
                    events.append((voronoiVertices[voronoiEdge[1]], "edgeStart", i))
                    events.append((infinitePoint, "edgeEnd", i))
                else:
                    events.append((infinitePoint, "edgeStart", i))
                    events.append((voronoiVertices[voronoiEdge[1]], "edgeEnd", i))
                edgeIndexVertexMapping[i] = (infinitePoint, voronoiVertices[voronoiEdge[1]])
            else:
                if voronoiVertices[voronoiEdge[1]][0] >= voronoiVertices[voronoiEdge[0]][0]:
                    events.append((voronoiVertices[voronoiEdge[0]], "edgeStart", i))
                    events.append((voronoiVertices[voronoiEdge[1]], "edgeEnd", i))
                else:
                    events.append((voronoiVertices[voronoiEdge[1]], "edgeStart", i))
                    events.append((voronoiVertices[voronoiEdge[0]], "edgeEnd", i))
                edgeIndexVertexMapping[i] = (voronoiVertices[voronoiEdge[0]], voronoiVertices[voronoiEdge[1]])
            i = i + 1

        i = 0
        polylineSegments = []
        for polyline in self.polylines:
            for j in range(0, len(polyline)-1):
                if polyline[j+1][0] >= polyline[j][0]:
                    events.append((polyline[j], "polyStart", i))
                    events.append((polyline[j+1], "polyEnd", i))
                else:
                    events.append((polyline[j+1], "polyStart", i))
                    events.append((polyline[j], "polyEnd", i))
                polylineSegments.append((polyline[j], polyline[j+1]))
                i = i + 1
        for closingLine in self.closingLines: # For now we simply treat these like normal polylines
            if closingLine[1][0] >= closingLine[0][0]:
                events.append((closingLine[0], "polyStart", i))
                events.append((closingLine[1], "polyEnd", i))
            else:
                events.append((closingLine[1], "polyStart", i))
                events.append((closingLine[0], "polyEnd", i))
            polylineSegments.append((closingLine[0], closingLine[1]))
            i = i + 1


        events.sort(key= lambda x: x[0][0])

        activePolySegments = set()
        activeEdgeSegments = set()

        remainingIndices = set(range(0, len(voronoiEdges)))

        for i in range(0, len(events)):
            _, eventType, segmentID = events[i]
            if eventType == "edgeStart":
                for activeSegment in activePolySegments:
                    if self.isIntersecting(edgeIndexVertexMapping[segmentID][0], edgeIndexVertexMapping[segmentID][1], polylineSegments[activeSegment][0], polylineSegments[activeSegment][1]):
                        if segmentID in remainingIndices:
                            remainingIndices.remove(segmentID)
                        break
                activeEdgeSegments.add(segmentID)
            if eventType == "edgeEnd":
                activeEdgeSegments.remove(segmentID)
            if eventType == "polyStart":
                for activeSegment in activeEdgeSegments:
                    if self.isIntersecting(edgeIndexVertexMapping[activeSegment][0], edgeIndexVertexMapping[activeSegment][1], polylineSegments[segmentID][0], polylineSegments[segmentID][1]):
                        if activeSegment in remainingIndices:
                            remainingIndices.remove(activeSegment)
                activePolySegments.add(segmentID)
            if eventType == "polyEnd":
                activePolySegments.remove(segmentID)
            if i % 100 == 0:
                print("Loading Bar Linesweep:", i, "/", len(events))

        print("Remaining lines after removing intersections: ", len(remainingIndices))
        remainingEdges = []
        infinitePoints = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
            if voronoiEdges[index][0] == -1:
                infinitePoints.append(edgeIndexVertexMapping[index][0])
        return remainingEdges, infinitePoints
    
    def populateConnectionDictionary(self, voronoiEdges: list[list[int]], connections: dict[list[list[int]]]):
        i = 0
        for edge in voronoiEdges:
            if edge[0] in connections:
                connections[edge[0]].append([edge[1], i])
            else:
                connections[edge[0]] = [[edge[1], i]]
            if edge[1] in connections:
                connections[edge[1]].append([edge[0], i])
            else:
                connections[edge[1]] = [[edge[0], i]]
            i += 1
    
    def removeConnectedEdges(self, connections: dict[list[list[int]]], remainingIndices: set, pointIndex):
        for connection in connections[pointIndex]:
            if connection[1] in remainingIndices:
                remainingIndices.remove(connection[1])
                self.removeConnectedEdges(connections, remainingIndices, connection[0])

    # 1. Loop through the centreline points
    # 2. For each centreline point, check the number of connections. If it's more than 2 it is a crossing.
    # 3. On a crossing: Recursively check the distance of each connecting path, if the distance exceeds a minimum value (say: 10 meters): break; If the path ends without exceeding the value: Remove the path
    def removeShortPathsFromCrossings(self, centrelines, minLength):
        remainingIndices = set(range(0, len(centrelines)))
        for centreline in centrelines:
            if len(centreline[3]) > 2: # The voronoi vertex is a crossing
                for connection in centreline[3]:
                    distance = math.sqrt((centreline[1] - centrelines[connection][1])**2 + (centreline[2] - centrelines[connection][2])**2)
                    self.doesLengthOfPathExceedMinimum(centrelines, connection, minLength, distance, centreline[0], remainingIndices)

        remainingCentrelines = []
        for index in remainingIndices:
            remainingCentrelines.append(centrelines[index])

        removedIndices = set(range(0, len(centrelines))).difference(remainingIndices)
        print("Removed", len(removedIndices), "short path nodes")
        for centreline in remainingCentrelines:
            newConnections = centreline[3][:]
            for connection in centreline[3]:
                if connection in removedIndices:
                    newConnections.remove(connection)
            centreline[3] = newConnections

        # Remove centreline nodes with no connections
        remainingCentrelineNodes = set(range(0, len(remainingCentrelines)))
        for i in range(0,len(remainingCentrelines)):
            if len(remainingCentrelines[i][3]) == 0:
                remainingCentrelineNodes.remove(i)
        newCentrelines = []
        for centrelineNodeIndex in remainingCentrelineNodes:
            newCentrelines.append(remainingCentrelines[centrelineNodeIndex])
        remainingCentrelines = newCentrelines

        return remainingCentrelines
    
    def doesLengthOfPathExceedMinimum(self, centrelines, nodeID, minLength, length, latestNode, remainingIndices):
        if length >= minLength:
            return True
        
        if len(centrelines[nodeID][3]) == 1: # End of a path
            if nodeID in remainingIndices:
                remainingIndices.remove(nodeID)
            return False
        
        longEnough = False
        for connection in centrelines[nodeID][3]:
            if connection != latestNode:
                distance = math.sqrt((centrelines[nodeID][1] - centrelines[connection][1])**2 + (centrelines[nodeID][2] - centrelines[connection][2])**2)
                if self.doesLengthOfPathExceedMinimum(centrelines, connection, minLength, length+distance, nodeID, remainingIndices):
                    longEnough = True
        
        if longEnough:
            return True
        else:
            if nodeID in remainingIndices:
                remainingIndices.remove(nodeID)
            return False
    

    # 1. Assign each centreline node a line ID of -1
    # 2. For each centreline node with ID -1: Assign a new ID [0, 1, ...] and explore all nodes connected to the node, assigning them the same ID.
    # 3. Count the number of nodes assigned to each ID
    # 4. Remove every node not belonging to the ID with max elements
    def removeNonMaximalCentreline(self, centrelines):
        lineIDs = {}
        nodeIDtoIndex = {}
        i = 0
        for centreline in centrelines:
            lineIDs[centreline[0]] = -1
            nodeIDtoIndex[centreline[0]] = i
            i = i + 1
        lineID = 0
        indicesToExplore = []
        for i in range(0, len(centrelines)):
            if lineIDs[centrelines[i][0]] == -1:
                lineIDs[centrelines[i][0]] = lineID
                currID = centrelines[i][0]
                for connection in centrelines[i][3]:
                    if lineIDs[connection] == -1:
                        indicesToExplore.append(connection)

                while len(indicesToExplore) != 0:
                    currID = indicesToExplore.pop()
                    for connection in centrelines[nodeIDtoIndex[currID]][3]:
                        if lineIDs[connection] == -1:
                            indicesToExplore.append(connection)
                    lineIDs[currID] = lineID

                lineID = lineID + 1
        
        indexOfMaxLengthLine = -1
        maxLengthSeen = 0
        for i in range(0, lineID+1):
            lineLength = list(lineIDs.values()).count(i)
            if lineLength > maxLengthSeen:
                maxLengthSeen = lineLength
                indexOfMaxLengthLine = i

        print("The max length line index is line", indexOfMaxLengthLine+1, "out of", lineID)
        
        remainingIndices = set(range(0, len(centrelines)))
        i = 0
        for lineID in lineIDs.values():
            if lineID != indexOfMaxLengthLine:
                remainingIndices.remove(i)
            i = i + 1

        remainingCentrelines = []
        for index in remainingIndices:
            remainingCentrelines.append(centrelines[index])
        
        print("The length of the centreline is", len(remainingCentrelines), "nodes. Removed", len(centrelines)-len(remainingCentrelines), "nodes")
        return remainingCentrelines
    
    # 1. Find all crossings and endpoints of the centreline and set these as nodes
    # 2. Recursively explore the centreline nodes between each connection between two crossings/endpoints
    # 3. Split the line consisting of the connected centreline nodes into segments with a length equal to the ideal distance between nodes (e.g. 5m)
    # 4. Perform linear regression on the nodes of each segment and place a node connected to the neighboring segments/endpoint/crossing in the "middle" of the aquired line
    def constructNodeGraphFromCentreline(self, centrelines, minLengthBetweenNodes):
        centrelineNodeIDtoIndex = {}
        for i in range(0, len(centrelines)):
            centrelineNodeIDtoIndex[centrelines[i][0]] = i

        crossingsAndEndpoints = []
        nodeGraph = {}
        for centreline in centrelines:
            if len(centreline[3]) > 2 or len(centreline[3]) == 1: # The voronoi vertex is a crossing or an endpoint
                crossingsAndEndpoints.append(centreline)
                nodegraphCrossingEndpoint = centreline[:]
                nodegraphCrossingEndpoint[3] = []
                nodeGraph[nodegraphCrossingEndpoint[0]] = nodegraphCrossingEndpoint

        exploredCrossingAdjacentCentrelineNodes = set() # The first and last centreline node of the nodes between a crossing/endpoint and crossing/endpoint is added to this in order not to create duplicate paths in the node graph while allowing for things like a crossing that connects to itself (a roundabout)
        for crossing in crossingsAndEndpoints:
            for connection in crossing[3]:
                if connection not in exploredCrossingAdjacentCentrelineNodes:
                    nodesBetweenCrossings = []
                    otherCrossingID = self.findCentrelineNodesBetweenCrossings(centrelines[centrelineNodeIDtoIndex[connection]], crossing[0], nodesBetweenCrossings, centrelines, centrelineNodeIDtoIndex)
                    if len(nodesBetweenCrossings) == 0:
                        if otherCrossingID not in nodeGraph[crossing[0]][3]:
                            nodeGraph[crossing[0]][3].append(otherCrossingID)
                            nodeGraph[otherCrossingID][3].append(crossing[0])
                        continue 

                    exploredCrossingAdjacentCentrelineNodes.add(nodesBetweenCrossings[0][0])
                    exploredCrossingAdjacentCentrelineNodes.add(nodesBetweenCrossings[-1][0])
                    centrelineNodeSegments = self.splitCentrelineNodesIntoSegments(nodesBetweenCrossings, minLengthBetweenNodes, crossing, nodeGraph[otherCrossingID])
                    if len(centrelineNodeSegments) == 0:
                        if otherCrossingID not in nodeGraph[crossing[0]][3]:
                            nodeGraph[crossing[0]][3].append(otherCrossingID)
                            nodeGraph[otherCrossingID][3].append(crossing[0])
                        continue
                    
                    # Take out the centre node of each centreline segment
                    nodesBetweenCrossings = []
                    for segment in centrelineNodeSegments:
                        node = segment[len(segment)//2][:]
                        node[3] = []
                        nodesBetweenCrossings.append(node)
                    # Connect the nodes from the segments
                    nodeGraph[crossing[0]][3].append(nodesBetweenCrossings[0][0])
                    nodesBetweenCrossings[0][3].append(crossing[0])
                    for i in range(0, len(nodesBetweenCrossings)-1):
                        nodesBetweenCrossings[i][3].append(nodesBetweenCrossings[i+1][0])
                        nodesBetweenCrossings[i+1][3].append(nodesBetweenCrossings[i][0])
                    nodesBetweenCrossings[-1][3].append(otherCrossingID)
                    nodeGraph[otherCrossingID][3].append(nodesBetweenCrossings[-1][0])

                    for node in nodesBetweenCrossings:
                        nodeGraph[node[0]] = node
        
        nodeGraphList = []
        for node in nodeGraph.values():
            nodeGraphList.append(node)
        return nodeGraphList
    

    def findCentrelineNodesBetweenCrossings(self, currentCentrelineNode, previousNodeID, nodesBetweenCrossings, centrelines, centrelineNodeIDtoIndex):
        if len(currentCentrelineNode[3]) > 2 or len(currentCentrelineNode[3]) == 1: # A crossing or endpoint has been reached
            return currentCentrelineNode[0]
        
        nodesBetweenCrossings.append(currentCentrelineNode)
        for connection in currentCentrelineNode[3]:
            if connection != previousNodeID:
                return self.findCentrelineNodesBetweenCrossings(centrelines[centrelineNodeIDtoIndex[connection]], currentCentrelineNode[0], nodesBetweenCrossings, centrelines, centrelineNodeIDtoIndex)


    def splitCentrelineNodesIntoSegments(self, nodesBetweenCrossings, minLength, crossing, otherCrossing):
        # This is a trivial solution that just traverses the nodes and cuts out a new segment when the minimum length has been passed. If this causes problems with nodes being too close together or far apart at the ends it may need to be changed
        segments = []
        segment = [nodesBetweenCrossings[0]]

        if math.sqrt((segment[0][1] - crossing[1])**2 + (segment[0][2] - crossing[2])**2) >= minLength:
            segments.append(segment)
            segment = []

        length = 0
        for i in range(1, len(nodesBetweenCrossings)):
            segment.append(nodesBetweenCrossings[i])
            distance = math.sqrt((nodesBetweenCrossings[i][1] - nodesBetweenCrossings[i-1][1])**2 + (nodesBetweenCrossings[i][2] - nodesBetweenCrossings[i-1][2])**2)
            length += distance
            if length >= minLength:
                segments.append(segment)
                segment = []
                length = 0

        totalNodes = 0
        for seg in segments:
            totalNodes += len(seg)

        return segments
    

    def adjustThreeWayCrossings(self, nodegraph, parallelAngleRadians):
        def lerp2D(x1, y1, x2, y2, f):
            x = (x1 * (1.0 - f)) + (x2 * f)
            y = (y1 * (1.0 - f)) + (y2 * f)
            return x, y

        nodeGraphIDtoIndex = {}
        for i in range(0, len(nodegraph)):
            nodeGraphIDtoIndex[nodegraph[i][0]] = i
        
        threeWayCrossings = []
        for node in nodegraph:
            if len(node[3]) == 3:
                threeWayCrossings.append(node) 

        for crossing in threeWayCrossings:
            cX = crossing[1]
            cY = crossing[2]
            mostParallelAngleFound = math.inf
            parAngX1 = 0
            parAngY1 = 0
            parAngX2 = 0
            parAngY2 = 0
            for connection1 in crossing[3]:
                for connection2 in crossing[3]:
                    if connection1 == connection2:
                        continue

                    x1 = nodegraph[nodeGraphIDtoIndex[connection1]][1]
                    y1 = nodegraph[nodeGraphIDtoIndex[connection1]][2]
                    x2 = nodegraph[nodeGraphIDtoIndex[connection2]][1]
                    y2 = nodegraph[nodeGraphIDtoIndex[connection2]][2]
                    angle1 = math.atan2(y1-cY, x1-cX)
                    angle2 = math.atan2(y2-cY, x2-cX)
                    angleParallel = abs((angle1%(2*math.pi)-angle2%(2*math.pi)) - math.pi)

                    if angleParallel < mostParallelAngleFound:
                        mostParallelAngleFound = angleParallel
                        parAngX1 = x1
                        parAngY1 = y1
                        parAngX2 = x2
                        parAngY2 = y2

            if mostParallelAngleFound <= parallelAngleRadians:
                length1 = math.sqrt((parAngX1 - cX)**2 + (parAngY1 - cY)**2)
                length2 = math.sqrt((parAngX2 - cX)**2 + (parAngY2 - cY)**2)
                lengthFraction = length1/(length1+length2)
                cX, cY = lerp2D(parAngX1, parAngY1, parAngX2, parAngY2, lengthFraction)
                nodegraph[nodeGraphIDtoIndex[crossing[0]]][1] = cX
                nodegraph[nodeGraphIDtoIndex[crossing[0]]][2] = cY

        return nodegraph    

    def calculateCentreline(self):
        sys.setrecursionlimit(100000)
        points = []
        for polyline in self.polylines:
            for point in polyline:
                points.append(point)
        self.points = points
        
        voronoi = Voronoi(points)    
        voronoiVertices = voronoi.vertices
        voronoiEdges = voronoi.ridge_vertices
        voronoiEdgepoints = voronoi.ridge_points

        voronoiEdges, infinitePoints = self.removeComplexIntersectingEdgesLineSweep(voronoiVertices, voronoiEdges, voronoiEdgepoints)

        remainingIndices = set(range(0, len(voronoiEdges)))
        connections = {}
        self.populateConnectionDictionary(voronoiEdges, connections)
        self.removeConnectedEdges(connections, remainingIndices, -1)
        remainingEdges = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        voronoiEdges = remainingEdges
        print("Cut out centrelines that are outside the mine tunnels. Remaining lines:", len(remainingEdges))

        centrelines = []
        for i in range(0, len(voronoiVertices)):
            centrelines.append([i, voronoiVertices[i][0], voronoiVertices[i][1], []])
        
        infinteLines = [] # When "removeConnectedEdges" is run this will always be empty
        infiniteIndex = 0
        for connection in voronoiEdges:
            if connection[0] != -1:
                centrelines[connection[0]][3].append(connection[1])
                centrelines[connection[1]][3].append(connection[0])
            else:
                infinteLines.append([infinitePoints[infiniteIndex], voronoiVertices[connection[1]]])
                infiniteIndex += 1

        centrelines = self.removeShortPathsFromCrossings(centrelines, self.minPathLength)

        centrelines = self.removeNonMaximalCentreline(centrelines)

        centrelines = self.constructNodeGraphFromCentreline(centrelines, self.minLengthBetweenNodes) 
        centrelines = self.constructNodeGraphFromCentreline(centrelines, self.minLengthBetweenNodes) # Running it twice is a bit of a hack, but it makes for a better graph

        centrelines = self.adjustThreeWayCrossings(centrelines, self.threeWayCrossingToleranceRad)

        return centrelines, infinteLines
    
    ######## THE FUNCTIONS BELOW ARE ONLY FOR DEMO PURPOSES, THEY SEPARATE calculateCentreline INTO STEPS#######
    def demoStep1(self): # Removes all line that cross mine tunnel walls
        sys.setrecursionlimit(100000)
        points = []
        for polyline in self.polylines:
            for point in polyline:
                points.append(point)
        self.points = points
        
        voronoi = Voronoi(points)    
        voronoiVertices = voronoi.vertices
        voronoiEdges = voronoi.ridge_vertices
        voronoiEdgepoints = voronoi.ridge_points

        voronoiEdges, infinitePoints = self.removeComplexIntersectingEdgesLineSweep(voronoiVertices, voronoiEdges, voronoiEdgepoints)

        remainingIndices = set(range(0, len(voronoiEdges)))
        connections = {}
        self.populateConnectionDictionary(voronoiEdges, connections)
        #self.removeConnectedEdges(connections, remainingIndices, -1)
        remainingEdges = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        voronoiEdges = remainingEdges
        print("Cut out centrelines that are outside the mine tunnels. Remaining lines:", len(remainingEdges))

        centrelines = []
        for i in range(0, len(voronoiVertices)):
            centrelines.append([i, voronoiVertices[i][0], voronoiVertices[i][1], []])
        
        infinteLines = [] # When "removeConnectedEdges" is run this will always be empty
        infiniteIndex = 0
        for connection in voronoiEdges:
            if connection[0] != -1:
                centrelines[connection[0]][3].append(connection[1])
                centrelines[connection[1]][3].append(connection[0])
            else:
                infinteLines.append([infinitePoints[infiniteIndex], voronoiVertices[connection[1]]])
                infiniteIndex += 1

        self.centrelines = centrelines

        return centrelines, infinteLines
    
    def demoStep2(self): # Could be immediately, it's the same as step1 except it removes lines connected to infinity 
        sys.setrecursionlimit(100000)
        points = []
        for polyline in self.polylines:
            for point in polyline:
                points.append(point)
        self.points = points
        
        voronoi = Voronoi(points)    
        voronoiVertices = voronoi.vertices
        voronoiEdges = voronoi.ridge_vertices
        voronoiEdgepoints = voronoi.ridge_points

        voronoiEdges, infinitePoints = self.removeComplexIntersectingEdgesLineSweep(voronoiVertices, voronoiEdges, voronoiEdgepoints)

        remainingIndices = set(range(0, len(voronoiEdges)))
        connections = {}
        self.populateConnectionDictionary(voronoiEdges, connections)
        self.removeConnectedEdges(connections, remainingIndices, -1)
        remainingEdges = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        voronoiEdges = remainingEdges
        print("Cut out centrelines that are outside the mine tunnels. Remaining lines:", len(remainingEdges))

        centrelines = []
        for i in range(0, len(voronoiVertices)):
            centrelines.append([i, voronoiVertices[i][0], voronoiVertices[i][1], []])
        
        infinteLines = [] # When "removeConnectedEdges" is run this will always be empty
        infiniteIndex = 0
        for connection in voronoiEdges:
            if connection[0] != -1:
                centrelines[connection[0]][3].append(connection[1])
                centrelines[connection[1]][3].append(connection[0])
            else:
                infinteLines.append([infinitePoints[infiniteIndex], voronoiVertices[connection[1]]])
                infiniteIndex += 1

        self.centrelines = centrelines

        return centrelines, infinteLines
    
    def demoStep3(self):
        self.centrelines = self.removeShortPathsFromCrossings(self.centrelines, self.minPathLength)
        return self.centrelines, []

    def demoStep4(self):
        self.centrelines = self.removeNonMaximalCentreline(self.centrelines)
        return self.centrelines, []
    
    def demoStep5(self):
        self.centrelines = self.constructNodeGraphFromCentreline(self.centrelines, self.minLengthBetweenNodes)
        self.centrelines = self.constructNodeGraphFromCentreline(self.centrelines, self.minLengthBetweenNodes) # Running it twice is a bit of a hack, but it makes for a better graph
        return self.centrelines, []
    
    def demoStep6(self):
        self.centrelines = self.adjustThreeWayCrossings(self.centrelines, self.threeWayCrossingToleranceRad)
        return self.centrelines, []

        


