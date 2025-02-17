from scipy.spatial import Voronoi, voronoi_plot_2d
import math
from pynput import keyboard

class VoronoiAlgorithm:
    def __init__(self):
        self.polylines = []
        self.points = []

    def setPolylines(self, polylineXs, polylineYs):
        self.polylines = []
        for i in range(0, len(polylineXs)):
            self.polylines.append([])
            for j in range(0, len(polylineXs[i])):
                self.polylines[i].append([polylineXs[i][j], polylineYs[i][j]])

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
    
    def isNeighboringIndexInSamePolyline(self, index):
        negativeNeighborPolylineIndex = -1
        positiveNeighborPolylineIndex = -1
        indexPolylineIndex = -1

        polylineIndex = 0
        i = 0
        for polyline in self.polylines:
            if i + len(polyline) < index - 1:
                polylineIndex = polylineIndex + 1
                i = i + len(polyline)
                continue
            for polylineVertex in polyline:
                if i == index-1:
                    negativeNeighborPolylineIndex = polylineIndex
                elif i == index:
                    indexPolylineIndex = polylineIndex
                elif i == index+1:
                    positiveNeighborPolylineIndex = polylineIndex
                    return positiveNeighborPolylineIndex == indexPolylineIndex or negativeNeighborPolylineIndex == indexPolylineIndex
                i = i + 1
            polylineIndex = polylineIndex + 1

        return positiveNeighborPolylineIndex == indexPolylineIndex or negativeNeighborPolylineIndex == indexPolylineIndex


    # Loop through the polyline vertices that define the voronoi diagram and remove any voronoi edges that lie between it and its polyline vertex neighbors, removes a majority of the intersecting polylines
    def removeSimpleIntersectingEdges(self, voronoiEdges, voronoiEdgepoints):
        remainingIndices = set(range(0, len(voronoiEdges)))
        for i in range(0, len(voronoiEdgepoints)):
            if voronoiEdgepoints[i][0] == voronoiEdgepoints[i][1]-1 or voronoiEdgepoints[i][0] == voronoiEdgepoints[i][1]+1:
                if self.isNeighboringIndexInSamePolyline(i):
                    remainingIndices.remove(i)
            if i % 100 == 0:
                print("Loading Bar 1:", i, "/", len(voronoiEdgepoints))
        
        print("Remaining: ", remainingIndices)
        remainingEdges = [] # Consider also if the edgePoints may be neccessary later and update them here
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        
        return remainingEdges
    
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
        remainingEdges = [] # Consider also if the edgePoints may be neccessary later and update them here
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        
        return remainingEdges
    
    def removeComplexIntersectingEdgesLineSweep(self, voronoiVertices, voronoiEdges): # Went from around 11 hours to under 2 minutes for this step on Boliden-Kriberg, worth it
        events = []
        i = 0
        for voronoiEdge in voronoiEdges:
            if voronoiEdge[0] == -1:
                events.append((voronoiVertices[voronoiEdge[1]], "infEdgeStart", i)) # TODO: Deal with these later
            else:
                if voronoiVertices[voronoiEdge[1]][0] >= voronoiVertices[voronoiEdge[0]][0]:
                    events.append((voronoiVertices[voronoiEdge[0]], "edgeStart", i))
                    events.append((voronoiVertices[voronoiEdge[1]], "edgeEnd", i))
                else:
                    events.append((voronoiVertices[voronoiEdge[1]], "edgeStart", i))
                    events.append((voronoiVertices[voronoiEdge[0]], "edgeEnd", i))
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

        events.sort(key= lambda x: x[0][0])
        #print(events)

        activePolySegments = set()
        activeEdgeSegments = set()

        remainingIndices = set(range(0, len(voronoiEdges)))

        for i in range(0, len(events)):
            point, eventType, segmentID = events[i]
            if eventType == "edgeStart":
                for activeSegment in activePolySegments:
                    if self.isIntersecting(voronoiVertices[voronoiEdges[segmentID][0]], voronoiVertices[voronoiEdges[segmentID][1]], polylineSegments[activeSegment][0], polylineSegments[activeSegment][1]):
                        #print("edge-poly intersect:", voronoiVertices[voronoiEdges[segmentID][0]], voronoiVertices[voronoiEdges[segmentID][1]], polylineSegments[activeSegment][0], polylineSegments[activeSegment][1], segmentID, activeSegment)
                        if segmentID in remainingIndices:
                            remainingIndices.remove(segmentID)
                        break
                activeEdgeSegments.add(segmentID)
            if eventType == "edgeEnd":
                activeEdgeSegments.remove(segmentID)
            if eventType == "polyStart":
                for activeSegment in activeEdgeSegments:
                    if self.isIntersecting(voronoiVertices[voronoiEdges[activeSegment][0]], voronoiVertices[voronoiEdges[activeSegment][1]], polylineSegments[segmentID][0], polylineSegments[segmentID][1]):
                        #print("poly-edge intersect:", voronoiVertices[voronoiEdges[activeSegment][0]], voronoiVertices[voronoiEdges[activeSegment][1]], polylineSegments[segmentID][0], polylineSegments[segmentID][1], segmentID, activeSegment)
                        if activeSegment in remainingIndices:
                            remainingIndices.remove(activeSegment)
                activePolySegments.add(segmentID)
            if eventType == "polyEnd":
                activePolySegments.remove(segmentID)
            if i % 100 == 0:
                print("Loading Bar 2:", i, "/", len(events))

        print("Remaining: ", remainingIndices)
        remainingEdges = []
        for index in remainingIndices:
            remainingEdges.append(voronoiEdges[index])
        
        return remainingEdges

    def calculateCentreline(self):
        points = []
        for polyline in self.polylines:
            for point in polyline:
                points.append(point)
        
        voronoi = Voronoi(points)    
        voronoiVertices = voronoi.vertices
        voronoiEdges = voronoi.ridge_vertices
        voronoiEdgepoints = voronoi.ridge_points

        voronoiEdges = self.removeSimpleIntersectingEdges(voronoiEdges, voronoiEdgepoints)
        #print(voronoiVertices)
        #print(voronoiEdgepoints)
        #print(voronoiEdges)


        #voronoiEdges = self.removeComplexIntersectingEdges(voronoiVertices, voronoiEdges)
        voronoiEdges = self.removeComplexIntersectingEdgesLineSweep(voronoiVertices, voronoiEdges)
        print(voronoiEdges)

        centrelines = []
        for i in range(0, len(voronoiVertices)):
            centrelines.append([i, voronoiVertices[i][0], voronoiVertices[i][1], []])
        
        for connection in voronoiEdges:
            if connection[0] != -1:
                centrelines[connection[0]][3].append(connection[1])
                centrelines[connection[1]][3].append(connection[0])

        return centrelines

