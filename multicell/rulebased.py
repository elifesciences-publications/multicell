import dolfin
import pyurdme
import heapq
import math
import numpy
import random
import vtk
import json
import math as Math
import sys
from scipy import stats
#sys.setrecursionlimit(10000)

class Cell(pyurdme.URDMEModel):

    def __init__(self, color="red",index=None, time_to_division=None):
        
        pyurdme.URDMEModel.__init__(self,name="test")

        self.properties = {}
        self.properties["color"] = color
        self.color = color
        self.index = index
        self.time_to_next_event = None
        self.time_to_division = time_to_division
        self.points = []
    
    
    def __cmp__(self,other):
        return self.time_to_division > other.time_to_division



class RuleBasedModel():

    def __init__(self,mesh_resolution=40):

        circle1 = dolfin.Circle(0,0,1)
        #circle2 = dolfin.Circle(-0.2,-0.2,0.2)
        self.lattice = pyurdme.URDMEMesh(mesh=dolfin.Mesh(circle1-circle2,mesh_resolution))
        #circle = dolfin.Circle(0,0,1)
        #self.lattice = pyurdme.URDMEMesh(mesh=dolfin.Mesh(circle,80))
        
        self.connectivity_graph = self._connectivityGraph()

        self.clock = 0.0
                
        self.state = numpy.zeros((1,self.lattice.num_vertices())).flatten()

        self.cells = []
        
        # Find center voxel
        coords = self.lattice.coordinates()
        shape = coords.shape
        
        point = [0.0,0.0]
        reppoint = numpy.tile(point, (shape[0], 1))
        dist = numpy.sqrt(numpy.sum((coords-reppoint)**2, axis=1))
        ix = numpy.argmin(dist)
        
        heapq.heappush(self.cells,Cell(color="red",index=ix, time_to_division=self.cell_division_time()))
        self.state[ix]=1
        
        
        neighbours = self.connectivity_graph[ix]

        point = [0.0,0.1]
        reppoint = numpy.tile(point, (shape[0], 1))
        dist = numpy.sqrt(numpy.sum((coords-reppoint)**2, axis=1))
        ix = numpy.argmin(dist)


        heapq.heappush(self.cells,Cell(color="green",index=ix, time_to_division=self.cell_division_time()))
        self.state[ix]=1
        
        point = [0.1,0.0]
        reppoint = numpy.tile(point, (shape[0], 1))
        dist = numpy.sqrt(numpy.sum((coords-reppoint)**2, axis=1))
        ix = numpy.argmin(dist)
        
        heapq.heappush(self.cells,Cell(color="blue",index=ix, time_to_division=self.cell_division_time()))
        self.state[ix]=1
        
        point = [0.1,0.1]
        reppoint = numpy.tile(point, (shape[0], 1))
        dist = numpy.sqrt(numpy.sum((coords-reppoint)**2, axis=1))
        ix = numpy.argmin(dist)
        
        heapq.heappush(self.cells,Cell(color="yellow",index=ix, time_to_division=self.cell_division_time()))
        self.state[ix]=1

        color = ["black"]*len(self.state)
        for cell in self.cells:
            color[cell.index] = cell.color



    def _connectivityGraph(self):
        """ Create a connectivity graph. """
        
        self.lattice.init()
        
        # incidence relation between vertices and edges
        conn = self.lattice.topology()(0,self.lattice.topology().dim()-1)
        edg2vtx = self.lattice.topology()(self.lattice.topology().dim()-1,0)
        
        connectivity = []
        for vertex in range(self.lattice.num_vertices()):
            temp = []
            for edg in conn(vertex):
                for vtx in edg2vtx(edg):
                    if vtx not in temp:
                        temp.append(vtx)

            temp.remove(vertex)
            connectivity.append(temp)

        return connectivity

    
    def cell_migration_time(self,cell):
        """ Sample time to cell migration = cell swapping position with neighbour. """
    
    def cell_migration(self, cell, other_cell):
        """ Execute cell migration event.  """
    
    def cell_death_time(self, cell):
        """ Time to spontaneous cell death of an individual. """
    
    def cell_death(self, cell):
        """ Execute cell death. """
    
    
    def cell_division_time(self, mean=0.5):
        """ Sample time until next cell division. """
        #return self.clock + mean
        return self.clock +random.gauss(mean,0.01*mean)
        #return self.clock - math.log(random.random())/mean
    
    def divide_cell(self, cell):
        """ Select the next cell to divide and perform the
            cell division. """
    
        cell_index = cell.index
        # Grab all neighbour sites of the present cells
        neighbours = self.connectivity_graph[cell_index]
        # And of those, select the ones that are not occupied
        free = [x for x in neighbours if self.state[x]==0]
        
        # Try to divide the cell so the progeny fills an empty, adjacent lattice site.
        try:
            r = random.randint(0,len(free)-1)
            new_cell = Cell(color=cell.color,index=free[r], time_to_division=self.cell_division_time())
            self.state[free[r]] = 1
            cell.time_to_division = self.cell_division_time()
            heapq.heappush(self.cells, new_cell)
            heapq.heappush(self.cells, cell)
        except ValueError,e:
            # There are no free neighbouring cells and the cell becomes quiesient.
            cell.time_to_division = "inf"
            heapq.heappush(self.cells, cell)


# TODO: Rewrite iteratively.

    def divide_cell_pushing(self,cell):
        """ Select the next cell to divide and perform the
        cell division. """
        
        # Grab all neighbour sites of the present cells
        neighbours = self.connectivity_graph[cell.index]
        # And of those, select the ones that are not occupied
        free = [x for x in neighbours if self.state[x]==0]
        
        # Assign weights for the division direction
        #w = numpy.ones((1,len(neighbours)))
        #w = w.flatten()
        # More likely to fill a free site
        #for i,n in enumerate(neighbours):
        #    if self.state[n]==0:
        #        w[i] = 10
        #w = w/numpy.sum(w)
        #p = stats.rv_discrete(name="fd",values=(neighbours, w))
        #r = p.rvs(size=1)
        
        # Draw a random division direction. Preference are given to free sites.
        if len(free)>0:
            r = free[random.randint(0,len(free)-1)]
        else:
            r = neighbours[random.randint(0,len(neighbours)-1)]
        
        # Create a new daugther cell
        new_cell = Cell(color=cell.color,index=r, time_to_division=self.cell_division_time())
    
        # Draw a new division time for the mother cell and put it back on the heap
        cell.time_to_division = self.cell_division_time()
        heapq.heappush(self.cells, cell)
        
        # If the location r is empty, just place the new cell there.
        if r in free:
            self.state[new_cell.index]=1
            heapq.heappush(self.cells, new_cell)
        else:
            for c in self.cells:
                if c.index == r:
                    # Displace the cell on the new position...
                    try:
                        self._displace(c, cell)
                    except RuntimeError,e:
                        raise CellDivisionError, e
        
            # And add the new cell to the now free location
            self.state[new_cell.index] = 1
            heapq.heappush(self.cells, new_cell)


    def _displace(self, cell, mother):

        nn = self.connectivity_graph[cell.index]
        neighbours = [x for x in nn if x != mother.index]
        free = [x for x in neighbours if self.state[x]==0]
        
        # Assign weights for the division direction
        # w = numpy.ones((1,len(neighbours)))
        #w = w.flatten()
        # More likely to fill a free site
        #for i,n in enumerate(neighbours):
        #    if self.state[n]==0:
        #        w[i] = 10
        #w = w/numpy.sum(w)
        #p = stats.rv_discrete(name="fd",values=(neighbours, w))
        #r = p.rvs(size=1)
        r = neighbours[random.randint(0,len(neighbours)-1)]
        # Draw a random division direction
        if len(free)>0:
            r = free[random.randint(0,len(free)-1)]
        else:
            r = neighbours[random.randint(0,len(neighbours)-1)]
        
        # If r is a free location, just add the cell there
        if r in free:
            self.state[r] = 1
            cell.index = r
        else:
            # And if not, displace the cell at location r...
            for c in self.cells:
                if c.index == r:
                    self._displace(c, cell)
            # and move this cell to the now empty location
            cell.index = r
                

    def run(self, time=None):
        """ Advance the system for a duration time. """
        i=0
        while self.clock < time:
            cell = heapq.heappop(self.cells)
            current_time = cell.time_to_division
            try:
                self.divide_cell_pushing(cell)
            except CellDivisionError:
                print "Failed to divide cell."
                break
            self.clock = current_time
            i=i+1
        print "Number of iterations "+str(i)



    def counterclockwise(self,vertex, points):
        angles = []
        #coords = self.lattice.coordinates()

        reference_point = vertex
        reference_vector = [points[0][0]-reference_point[0],points[0][1]-reference_point[1]]
        
        for i,c in enumerate(points):

            vector = []
            vector.append(c[0]-reference_point[0])
            vector.append(c[1]-reference_point[1])
            a = math.atan2(reference_vector[1],reference_vector[0])
            b = math.atan2(vector[1], vector[0])
            
            if a < 0:
                a +=  2*math.pi

            if b < 0:
                b +=  2*math.pi

            val = b-a
            angles.append(val)

        idx = numpy.argsort(numpy.array(angles))


        sorted_points = list(numpy.array(points)[idx])
        return sorted_points

    def draw(self):
        
        
        import numpy as np
        import matplotlib
        from matplotlib.patches import Circle, Wedge, Polygon
        from matplotlib.collections import PatchCollection
        import matplotlib.pyplot as plt


        fig, ax = plt.subplots()

        coords = self.lattice.coordinates()
        patches = []
        vtx2cell = self.lattice.topology()(0,self.lattice.topology().dim())
        triangles = self.lattice.topology()(self.lattice.topology().dim(),0)

        facecolors = []
        for cell in self.cells:
            vertex = coords[cell.index,:]
            neig = self.connectivity_graph[cell.index]
            points = []

            for n in neig:
                points.append(coords[n,:])
            points = midpoints(vertex, points)
            
            
            for triangle in vtx2cell(cell.index):
                tvtx = triangles(triangle)
                points.append(centroid(coords[tvtx,:]))
            
            points = self.counterclockwise(vertex, points)
            polygon = Polygon(points,fill=True)
            facecolors.append(cell.color)
            patches.append(polygon)

        p = PatchCollection(patches, facecolor = facecolors)
        ax.add_collection(p)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        plt.show()


def centroid(points):
    
    return numpy.mean(points, axis=0)


def midpoints(vertex, points):
    newpoints = []
    for point in points:
        newpoints.append([(vertex[0]+point[0])/2, (vertex[1]+point[1])/2] )
    return newpoints


def dot(x,y):
    return x[0]*y[0]+x[1]*y[1]

def norm(x):
    return math.sqrt(dot(x,x))

class CellDivisionError(Exception):
    pass

if __name__ == '__main__':

    system = RuleBasedModel()
    system.run(time=6)

    color = ["black"]*len(system.state)
    for cell in system.cells:
        color[cell.index] = cell.color
    
    print "Number of cells ", str(len(system.cells))
    print "Number of voxels ", str(system.lattice.num_vertices())


    system.draw()



