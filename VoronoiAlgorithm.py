import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy
import sys

class VoronoiAlgorithm:
    def __init__(self):
        self.polylines = []
        self.points = []
        self.closingLines = []
    def setPolylines(self, polylineXs, polylineYs):
        self.polylines = []
        for i in range(0, len(polylineXs)):
            self.polylines.append([])
            for j in range(0, len(polylineXs[i])):
                self.polylines[i].append([polylineXs[i][j], polylineYs[i][j]])

    def setClosingLines(self, closinglineStartEnds): # Special lines to close off segments, are only checked in the complex intersecting edges step (and might in the future have its intersections noted in some way to be used for reconstructing the map from segments)
        if len(closinglineStartEnds) == 0:
            self.closingLines = []
            return
        if len(closinglineStartEnds[-1]) < 2:
            closinglineStartEnds = closinglineStartEnds[:-1]
        self.closingLines = closinglineStartEnds

    def isCounterclockwise(self, pointA: list[float], pointB: list[float], pointC: list[float]):
        return (pointC[1]-pointA[1]) * (pointB[0]-pointA[0]) > (pointB[1]-pointA[1]) * (pointC[0]-pointA[0])
    
    # If this turns out to be too slow then one could try line-sweep
    def isIntersecting(self, line1Start: list[float], line1End: list[float], line2Start: list[float], line2End: list[float]): # This code doesn't handle colinearity but hopefully that won't be necessary
        return self.isCounterclockwise(line1Start, line2Start, line2End) != self.isCounterclockwise(line1End, line2Start, line2End) and self.isCounterclockwise(line1Start, line1End, line2Start) != self.isCounterclockwise(line1Start, line1End, line2End)
    
    def clearInfiniteEdges(self, voronoiEdges): # Should one remove the infinite edges at ones? If they make a poor fit for determining inside-outside (which might be the case, see simple examples) then they probably should. Besides, including them would make intersection calculation more complicated.
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
    
    def removeComplexIntersectingEdges(self, voronoiVertices, voronoiEdges): # TODO: Maybe line sweep this for improved time complexity
        remainingIndices = set(range(0, len(voronoiEdges)))
        for i in range(0, len(voronoiEdges)):
            shouldBreak = False # I do not like this kind of construction...
            if voronoiEdges[i][0] == -1:
                continue # TODO: Deal with these later
            for polyline in self.polylines:
                if shouldBreak:
                    break
                for j in range(0, len(polyline)-1):
                    #if i == 13:
                        #print("The problem", voronoiEdges[i], voronoiVertices[voronoiEdges[i][0]], voronoiVertices[voronoiEdges[i][1]], polyline[j], polyline[j+1])
                    if self.isIntersecting(voronoiVertices[voronoiEdges[i][0]], voronoiVertices[voronoiEdges[i][1]], polyline[j], polyline[j+1]):
                        #print("removing: ", i, voronoiEdges[i], voronoiVertices[voronoiEdges[i][0]], voronoiVertices[voronoiEdges[i][1]], polyline[j], polyline[j+1])
                        remainingIndices.remove(i)
                        shouldBreak = True
                        break
            if i % 100 == 0:
                print("Loading Bar 2:", i, "/", len(voronoiEdges))

        print("Remaining: ", remainingIndices)
        remainingEdges = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        
        return remainingEdges
    
    def removeComplexIntersectingEdgesLineSweep(self, voronoiVertices, voronoiEdges, voronoiEdgepoints): # Went from around 11 hours to under 2 minutes for this step on Boliden-Kriberg, very much worth it
        #Setup for the infinite edge calculation, TODO: make this more generalised
        center = numpy.mean(self.points, axis = 0)
        minmaxPointLocations = numpy.ptp(self.points, axis = 0)
        events = []
        edgeIndexVertexMapping = {} # This construction is made neccessary by the infinite edges
        i = 0
        for voronoiEdge in voronoiEdges:
            if voronoiEdge[0] == -1:
                # Finds the tangent and normal
                print(voronoiEdgepoints[i])
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
                        #print("edge-poly intersect:", voronoiVertices[voronoiEdges[segmentID][0]], voronoiVertices[voronoiEdges[segmentID][1]], polylineSegments[activeSegment][0], polylineSegments[activeSegment][1], segmentID, activeSegment)
                        if segmentID in remainingIndices:
                            remainingIndices.remove(segmentID)
                        break
                activeEdgeSegments.add(segmentID)
            if eventType == "edgeEnd":
                activeEdgeSegments.remove(segmentID)
            if eventType == "polyStart":
                for activeSegment in activeEdgeSegments:
                    if self.isIntersecting(edgeIndexVertexMapping[activeSegment][0], edgeIndexVertexMapping[activeSegment][1], polylineSegments[segmentID][0], polylineSegments[segmentID][1]):
                        #print("poly-edge intersect:", voronoiVertices[voronoiEdges[activeSegment][0]], voronoiVertices[voronoiEdges[activeSegment][1]], polylineSegments[segmentID][0], polylineSegments[segmentID][1], segmentID, activeSegment)
                        if activeSegment in remainingIndices:
                            remainingIndices.remove(activeSegment)
                activePolySegments.add(segmentID)
            if eventType == "polyEnd":
                activePolySegments.remove(segmentID)
            if i % 100 == 0:
                print("Loading Bar 2:", i, "/", len(events))

        print("Remaining: ", remainingIndices)
        print("Remaining lines: ", len(remainingIndices))
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
        #print(voronoiEdges, pointIndex)
        # for i in range(0, len(voronoiEdges)):
        #     if i not in remainingIndices:
        #         continue
        #     if voronoiEdges[i][0] == pointIndex:
        #         if i in remainingIndices:
        #             remainingIndices.remove(i)
        #         self.removeConnectedEdges(voronoiEdges, remainingIndices, voronoiEdges[i][1])
        #     elif voronoiEdges[i][1] == pointIndex:
        #         if i in remainingIndices:
        #             remainingIndices.remove(i)
        #         self.removeConnectedEdges(voronoiEdges, remainingIndices, voronoiEdges[i][0])

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
        print("removed indices:", removedIndices)
        for centreline in remainingCentrelines:
            newConnections = centreline[3][:]
            for connection in centreline[3]:
                if connection in removedIndices:
                    newConnections.remove(connection)
            centreline[3] = newConnections
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

        #voronoiEdges, voronoiEdgepoints = self.removeSimpleIntersectingEdges(voronoiEdges, voronoiEdgepoints)
        #print(voronoiVertices)
        #print(voronoiEdgepoints)
        #print(voronoiEdges)


        #voronoiEdges = self.removeComplexIntersectingEdges(voronoiVertices, voronoiEdges)
        voronoiEdges, infinitePoints = self.removeComplexIntersectingEdgesLineSweep(voronoiVertices, voronoiEdges, voronoiEdgepoints)
        #print(voronoiEdges)
        #print(infinitePoints)
        remainingIndices = set(range(0, len(voronoiEdges)))
        connections = {}
        self.populateConnectionDictionary(voronoiEdges, connections)
        self.removeConnectedEdges(connections, remainingIndices, -1)
        remainingEdges = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        voronoiEdges = remainingEdges
        print("Cut out centrelines")

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
        
        centrelines = self.removeShortPathsFromCrossings(centrelines, 10)

        return centrelines, infinteLines

