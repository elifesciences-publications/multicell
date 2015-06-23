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
import os
import numpy.linalg


try:
    # This is only needed if we are running in an Ipython Notebook
    import IPython.display
except:
    pass

sys.setrecursionlimit(10000)

class Cell(pyurdme.URDMEModel):
    """ Model for the indivuduals/agents """
    
    def __init__(self, color="red",index=None, mean_division_time=None, time_to_migration=None, time_to_division=None):
        
        pyurdme.URDMEModel.__init__(self,name="test")

        self.properties = {}
        self.properties["color"] = color
        self.color = color
        self.index = index
        
        self.mean_division_time = mean_division_time
        self.time_to_migration = time_to_migration
    
        self.time_to_division = time_to_division
        self.update()

        self.points = []
            
    def update(self):
        self.time_to_next_event = min(self.time_to_migration, self.time_to_division)
    
    def __cmp__(self,other):
        return self.time_to_next_event > other.time_to_next_event



class RuleBasedModel():

    def __init__(self,cell_size=0.01, mesh_type="boxmesh", mesh_resolution=80, mean_division_time=0.42, variance_division_time=10, mean_migration_time=None, pushing_factor=0.5, gradient=None, polarization_strength=1.0):
        
        
        self.mean_division_time = mean_division_time
        self.variance_division_time = variance_division_time*mean_division_time/100.0
        if mean_migration_time == None:
            self.mean_migration_time = mean_division_time/3.0
        else:
            self.mean_migration_time = mean_migration_time


        # Pushing parameter
        self.c = pushing_factor
        # Variance in cell division rate
        self.f = variance_division_time/mean_division_time
        # "Polarization strength"
        self.b = polarization_strength
        
        #circle1 = dolfin.Circle(0,0,1)
        #sphere = dolfin.Sphere(dolfin.Point(0,0,0.15),0.5)
        
        sphere = dolfin.Sphere(dolfin.Point(0,0,0.15),0.5)
        # eye1 = dolfin.Sphere(dolfin.Point(0.3,0,0),0.05)
        #eye2 = dolfin.Sphere(dolfin.Point(-0.3,0,0),0.05)
        brain = dolfin.Sphere(dolfin.Point(0,0,0),0.3)
        #brain_bubble=brain+eye1+eye2
        domain = sphere - brain
        #domain = dolfin.Box(0,0,0,1,0.2,1)
        x = 1
        y=0.2
        z=1
        nx = int(x/cell_size)
        ny = int(y/cell_size)
        nz = int(z/cell_size)
       
        self.lattice = pyurdme.URDMEMesh(mesh=dolfin.Mesh(domain,mesh_resolution))
        
        #circle = dolfin.Circle(0,0,1)
        #self.lattice = pyurdme.URDMEMesh(mesh=dolfin.Mesh(circle,80))
        self.gradient = pyurdme.DolfinFunctionWrapper(self.lattice.get_function_space())
        #self.set_gradient_onesided_linear()
        #self.set_gradient_onesided_linear()
        #self.set_gradient_brain()
        if gradient == None:
            self.set_gradient_uniform()
        elif gradient == "radial":
            self.set_gradient_brain()
        

        self.connectivity_graph = self._connectivity_graph()

        # Init state
        self.clock = 0.0
        (Nvox,dim) = numpy.shape(self.lattice.coordinates())
        print Nvox
        self.state=[None]*Nvox
        
        self.cells = []
    # self.deposit_cells_one_cell()
        self.deposit_cells_on_brain()

    def reset(self):
        """ Reset the heap so that the simulation can start from t=0 again. """
        (Nvox,dim) = numpy.shape(self.lattice.coordinates())
        self.state = [None]*Nvox
        while True:
            try:
                heapq.heappop(self.cells)
            except:
                break
        self.clock = 0.0
        self.deposit_cells_on_brain()

    def _connectivity_graph(self):
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


    def deposit_cells_one_cell(self, cell_size):
        """ Deposit cells in a plane """
        
        x0 = 0.5
        y0 = 0.0
        z0 = 0.5
     
        rho = cell_size
        
        coords = self.lattice.coordinates()
        shape = coords.shape

        point = [x0,y0,z0]
        reppoint = numpy.tile(point, (shape[0], 1))
        dist = numpy.sqrt(numpy.sum((coords-reppoint)**2, axis=1))
        ix = numpy.argmin(dist)
        # print coords[ix]
        
        heapq.heappush(self.cells,Cell(color="red",index=ix, time_to_migration=self.sample_migration_time(), time_to_division=self.cell_division_time()))

    def deposit_cells(self, cell_size):
        """ Deposit cells in a plane """
        
        x0 = 0.4
        z0 = 0.4
        rho = cell_size
        
        x = x0+numpy.linspace(0,20*rho,10)
        z = z0+numpy.linspace(0,20*rho,10)
        
        import itertools
        space = itertools.product(x,z)
        #cols = [5 10 15 20]
        coords = self.lattice.coordinates()
        shape = coords.shape
        for i,p in enumerate(space):
            point = [p[0],0.0,p[1]]
            reppoint = numpy.tile(point, (shape[0], 1))
            dist = numpy.sqrt(numpy.sum((coords-reppoint)**2, axis=1))
            ix = numpy.argmin(dist)
            if i == 22:
                heapq.heappush(self.cells,Cell(color="red",index=ix, time_to_division=self.cell_division_time(),time_to_migration=self.sample_migration_time()))
            elif  i==36:
                heapq.heappush(self.cells,Cell(color="green",index=ix, time_to_division=self.cell_division_time(),time_to_migration=self.sample_migration_time()))
            elif i==68:
                heapq.heappush(self.cells,Cell(color="cyan",index=ix, time_to_division=self.cell_division_time(),time_to_migration=self.sample_migration_time()))
            elif i==72:
                heapq.heappush(self.cells,Cell(color="yellow",index=ix, time_to_division=self.cell_division_time(),time_to_migration=self.sample_migration_time()))
            else:
                heapq.heappush(self.cells,Cell(color="white",index=ix, time_to_division=self.cell_division_time(),time_to_migration=self.sample_migration_time()))
            self.state[ix]=1

    def deposit_cells_on_brain(self):
        
        #r = 0.3
        
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0
        
        h = numpy.mean(self.lattice.get_mesh_size())
        print "Mean mesh size", h
        
        coords = self.lattice.coordinates()
        shape = coords.shape
        
        thetas = numpy.linspace(-math.pi/20,math.pi/20,6)
        phis   = numpy.linspace(0,math.pi/20,6)
        rho = [0.3,0.3+h,0.3+2*h]
        for r in rho:
            for i,theta in enumerate(thetas):
                for j,phi in enumerate(phis):
                    point = [x0+r*math.cos(theta)*math.sin(phi),y0+r*math.sin(theta)*math.cos(phi),z0+r*math.cos(phi)]
                    reppoint = numpy.tile(point, (shape[0], 1))
                    dist = numpy.sqrt(numpy.sum((coords-reppoint)**2, axis=1))
                    ix = numpy.argmin(dist)
                    if not self.state[ix]:
                        cell = Cell(color="white",index=ix, time_to_migration=self.sample_migration_time(), time_to_division=self.cell_division_time())
                        if j ==4:
                            if i==2:
                                cell.color = "cyan"
                        if j==2:
                            if i==3:
                                cell.color = "red"
                        
                        if j==3:
                            if i==4:
                                cell.color = "lilac"
                                    
                        if j==1:
                            if i==1:
                                cell.color = "yellow"
                    
                        # add reference to cell
                        try:
                            self.state[ix]=cell
                        except SystemError,e:
                            print str(e)
                            raise
                        heapq.heappush(self.cells,cell)


    def gradient_dof_to_vertex(self):
        vertex_to_dof = dolfin.vertex_to_dof_map(self.lattice.get_function_space())
        print vertex_to_dof

    def set_gradient_linear(self):
    
        vec = self.gradient.vector()
        p = self.lattice.get_voxels()
        np, dim = numpy.shape(p)
        for i in xrange(np):
            x = p[i,:]
            #r = numpy.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
            r = math.fabs(x[1]-0.2/2)
            vec[i] = r

    def set_gradient_brain(self):
    
        vec = self.gradient.vector()
        p = self.lattice.get_voxels()
        np, dim = numpy.shape(p)
        for i in xrange(np):
            x = p[i,:]
            r = numpy.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
            #r = math.fabs(x[1]-0.2/2)
            vec[i] = r



    def set_gradient_uniform(self):
        
        vec = self.gradient.vector()
        p = self.lattice.get_voxels()
        np, dim = numpy.shape(p)
        for i in xrange(np):
            vec[i] = 1.0


    def set_gradient_onesided_linear(self):
    
        vec = self.gradient.vector()
        p = self.lattice.get_voxels()
        np, dim = numpy.shape(p)
        for i in xrange(np):
            x = p[i,:]
            #r = numpy.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
            #r = max(x[1]-0.2/2,0)
            r = x[1]
            vec[i] = r


    def set_gradient_twosided_quadratic(self):
        
        vec = self.gradient.vector()
        p = self.lattice.get_voxels()
        np, dim = numpy.shape(p)
        for i in xrange(np):
            x = p[i,:]
            #r = numpy.sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])
            r = 1000*((x[1]-0.2/2)*(x[1]-0.2/2))
            vec[i] = r


    
    def sample_migration_time(self):
        """ Sample time to cell migration = cell swapping position with neighbour. """
        return self.clock + random.gauss(self.mean_migration_time,self.mean_migration_time*0.01)

    def _sample_neighbour_uniform(self, cell):
        """ Sample a neighbour cell from a uniform distribution. """

    def cell_migration(self, cell):
        """ Execute cell migration event.  """

        # pick a random neigbor site to migrate to.
        neighbours = self.connectivity_graph[cell.index]
        N = len(neighbours)
      
        n = neighbours[random.randint(0,N-1)]
        other_cell = self.state[n]
        
        if other_cell:
            #   if not other_cell:
            ##    print self.previous_event
            #   self.check_consistency()
            #    raise CellMigrationError("the selected neighbor does not appear to exist. ")
            index = cell.index
            self.state[index] = other_cell
            cell.index = other_cell.index
            self.state[n] = cell
            other_cell.index = index
            
            # Swap references
            #   self.state[cell.index] = other_cell
        #    self.state[n] = cell
        

            #other_cell.time_to_migration = self.sample_migration_time()
            #other_cell.update()
        else:
            # Move into the non-occupied space
            self.state[n]=cell
            self.state[cell.index] = None
            cell.index = n

        cell.time_to_migration = self.sample_migration_time()
        cell.update()
        heapq.heappush(self.cells, cell)
        #print "Migration after",  len(self.cells)



    def cell_death_time(self, cell):
        """ Time to spontaneous cell death of an individual. """
    
    def cell_death(self, cell):
        """ Execute cell death. """
    
    def cell_division_time(self):
        """ Sample time until next cell division. """
    #return self.clock + self.mean_division_time
        return self.clock + random.gauss(self.mean_division_time,self.variance_division_time)
        #return self.clock - math.log(random.random())/mean
    

    def get_weights(self,cell, neighbours):
        # Assign weights for the division direction
        data = self.gradient.vector()
        coords = self.lattice.coordinates()
        
        w = []
        for n in neighbours:
            edge = coords[n]-coords[cell.index]
            #    print data[n],data[cell.index]
            w.append(max((data[n]-data[cell.index])/numpy.linalg.norm(edge),0.0))
                # print w
        maxw = w[numpy.argmax(w)]
        # print numpy.argmax(w),maxw
        if maxw == 0:
            maxw=1
        for i,we in enumerate(w):
            w[i] = (we/maxw)**self.b
            #print w
        
        w = numpy.cumsum(w)
        
        if w[-1]==0:
            raise CellDivisionError("All division weights are zero")
        w = w/w[-1]
        return w

    def sample_division_direction(self, cell):
        """ Sample a proposed division direction. """
        # Grab all neighbour sites of the present cell
        neighbours = self.connectivity_graph[cell.index]
        w = self.get_weights(cell, neighbours)
        #print "w",w
        u = random.random()
        #print "u", u
        r = numpy.digitize([u], w)
        r = r[0]
        #print "r",r
        return neighbours[r]

    def sample_pushing_direction(self, cell, mother):
        """ Sample a proposed displacement direction due to a pushing event. """
        
        # Grab all neighbour sites of the present cell
        neighbours = self.connectivity_graph[cell.index]
            # if cell.index in neighbours:
            #    print "This is not great"
        coords = self.lattice.coordinates()
        #print cell, neighbours

        # The displacement vector is defined to be the connecting edge between the mother and
        # daughter cell.
        mv = coords[mother.index]
        dv = coords[cell.index]
        d = dv-mv

        # Now, we assign weights to the neighbours of the daughter by computing scalar projections on
        # the displacement vectors. Hence, the more aligned a direction is to the
        # pushing direction, the more likey it is that that cell will be displaced.

        # We can also modify the weight to make it less likely to push a cell to an occupied cell.
        # c is a number between 0 and 1 that dictates how much an occupied cell hinders the pushing.
        # c = 0 means no hindrance at all and 1 means that a cell can't be displaced.
        w = []
        for n in neighbours:
            nd  = coords[n] - coords[cell.index]
            nd = nd/numpy.linalg.norm(nd)
            a = numpy.dot(d, nd)
            tt=0.0
            if self.state[n]:
                tt=1.0
            
            w.append(max(a,0.0)*(1-self.c*tt))

        w = numpy.cumsum(w)
        
        # If all weights are zero, we will not be able to push the neighbouring cell.
        # Then we just give up.
        if w[-1] == 0:
            print "d:", d, mv, dv
            print cell, mother, cell.index, mother.index
            #for c in self.cells:
            #    print c, c.index
            raise PushingError("Could not find a valid pushing direction.")
        
        w = w/w[-1]

        # Sample a pushing direction
        u = random.random()
        r = numpy.digitize([u], w)
        r = r[0]

        if neighbours[r] == cell.index:
            print "Sampled self cell"
        
        if neighbours[r] == mother.index:
            print "Sampled mother cell"
        
        
        return neighbours[r]

    def divide_cell(self,cell):
        """ Select the next cell to divide and perform the
        cell division. """
        
        #print "Dividing cell", cell
        
        # Sample division direction
        r = self.sample_division_direction(cell)
        
        # Create a new daugther cell
        new_cell = Cell(color=cell.color,index=r,time_to_division=self.cell_division_time(), time_to_migration=self.sample_migration_time())
        
        
        # If the location r is empty, just place the new cell there....
        c= self.state[r]
        if not c:
            self.state[r]=new_cell
        else:
            # if not, displace the cell that is there to make room for it...
            try:
                self._displace(c, cell)
            except CellDivisionError,e:
                #print "Cell: ", cell.index
                #print "Neighbours: ", self.connectivity_graph[cell.index]
                #print "State:", self.state[self.connectivity_graph[cell.index]]
                raise CellDivisionError("Failed to divide cell:"+str(e))
        
            # ...add then insert the new cell in the now free location
            self.state[r] = new_cell

        heapq.heappush(self.cells, new_cell)
        # Draw a new division time for the mother cell and put it back on the heap
        cell.time_to_division = self.cell_division_time()
        cell.update()
        heapq.heappush(self.cells, cell)


    def _displace(self, cell, mother):
        """ Displace "cell", leaving its current location empty. """
            
        try:
            r = self.sample_pushing_direction(cell, mother)
        except PushingError,e:
            raise CellDivisionError("Failed to push the cell."+str(e))
        
        # If the newly sampled location r is unoccupied, just add cell there...
        c = self.state[r]
        if not c:
            self.state[r] = cell
            self.state[cell.index]=None
            cell.index = r
        else:
            # And if not, displace the cell at location r...
            try:
                self._displace(c, cell)
                # then insert this cell in the now empty location, leaving its previous
                # site unoccupied.
                self.state[cell.index]=None
                cell.index = r
                self.state[r]=cell
            except RuntimeError,e:
                raise CellDivisionError("Failed to divide cell:"+str(e))



    def check_consistency(self):
        """ Check that the state and cell list is consistent. """
        
        ll = []
        for c in self.cells:
            ll.append(c.index)
        
        if len(list_duplicates(ll))> 0:
            raise InconsistencyError("Two cells in the same place. ")


        for c in self.cells:
            if self.state[c.index] == 0:
                print c, c.index
                raise InconsistencyError("Heap not consistent, cell exists but not in state")
                    
        cons = True
        for i,v in enumerate(self.state):
            cons = False
            if v:
                for c in self.cells:
                    if c.index == i:
                        cons = True
                        break
                if not cons:
                    raise InconsistencyError("Heap not consistent, state is one but cell is missing")
    
    
    
    def run(self, time=None):
        """ Advance the system for a duration time. """
        i=0
        print "Stated with nr cells:",len(self.cells)
            #try:
            #    self.check_consistency()
            #except InconsistencyError,e:
            # print "inconsistent initially"+str(e)
            #raise
        while self.clock < time:
            
            cell = heapq.heappop(self.cells)
            current_time = cell.time_to_next_event
            try:
                if cell.time_to_division < cell.time_to_migration:
                    self.divide_cell(cell)
                    self.previous_event = "Division"
                        #try:
                        #    self.check_consistency()
                        #except InconsistencyError,e:
                        #   print "inconsistent after cell division"+str(e)
                        # raise
                else:
                    #print "Cell migration event"
                    self.cell_migration(cell)
                    self.previous_event = "Migration"

                        #try:
                        #    self.check_consistency()
                        #except InconsistencyError,e:
                        #  print "Inconsistent after migration"+str(e)
                        # raise
            except CellDivisionError,e:
                print str(e)
                raise
            self.clock = current_time

            i=i+1
    
        print "Number of iterations: ", str(i), "Total number of cells: ", str(len(self.cells))



    def system_order(self):

        coords = self.lattice.coordinates()
        shape = coords.shape

        pos = []
        for cell in self.cells:
            if cell.color == "red":
                pos.append(coords[cell.index])

        return pos

        #for color on self.colors:
    


    def counterclockwise(self,vertex, points):

        angles = []
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


    def write_particles(self):
        coords = system.lattice.coordinates()
        celldatalist = []
        for cell in system.cells:
            center = coords[cell.index,:]
            celldata = list(center) + [0.01] +[color[cell.index]]
            celldatalist.append(celldata)
    
        with open('3dcellstest.dat','w') as fh:
            fh.write(str(celldatalist))

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


    def _export_to_particle_js(self, opacity):
        """ Create a html string for displaying the particles as small spheres. """
        import random
        with open(os.path.dirname(os.path.abspath(__file__))+"/particles.html",'r') as fd:
            template = fd.read()
        
        coordinates = self.lattice.coordinates()

        dims = numpy.shape(coordinates)
        if dims[1]==2:
            is3d = 0
            vtxx = numpy.zeros((dims[0],3))
            for i, v in enumerate(coordinates):
                vtxx[i,:]=(list(v)+[0])
            coordinates = vtxx
        else:
            is3d = 1
        
        h = self.lattice.get_mesh_size()
        
        x=[];
        y=[];
        z=[];
        c=[];
        radius = []
        
        for cell in self.cells:
            #if cell.color != "white":
            x.append(coordinates[cell.index,0])
            y.append(coordinates[cell.index,1])
            z.append(coordinates[cell.index,2])
            radius.append(h[cell.index])
            c.append(cell.color)
        
        
        #Brain
        x.append(0)
        y.append(0)
        z.append(0)
        c.append("grey")
        radius.append(0.3)
        
        x.append(0.3)
        y.append(0)
        z.append(0)
        c.append("grey")
        radius.append(0.07)
        x.append(-0.3)
        y.append(0)
        z.append(0)
        c.append("grey")
        radius.append(0.07)
        
        template = template.replace("__OPACITY__",str(opacity))
        template = template.replace("__X__",str(x))
        template = template.replace("__Y__",str(y))
        template = template.replace("__Z__",str(z))
        template = template.replace("__COLOR__",str(c))
        template = template.replace("__RADIUS__",str(radius))


        return template

    def display_particles(self, opacity=0.1):
        hstr = self._export_to_particle_js(opacity)
        import uuid
        displayareaid=str(uuid.uuid4())
        hstr = hstr.replace('###DISPLAYAREAID###',displayareaid)
        
        html = '<div id="'+displayareaid+'" class="cell"></div>'
        IPython.display.display(IPython.display.HTML(html+hstr))



def list_duplicates(seq):
    seen = set()
    seen_add = seen.add
    # adds all elements it doesn't know yet to seen and all other to seen_twice
    seen_twice = set( x for x in seq if x in seen or seen_add(x) )
    # turn the set into a list (as requested)
    return list( seen_twice )

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

class PushingError(Exception):
    pass

class CellMigrationError(Exception):
    pass

class InconsistencyError(Exception):
    pass

if __name__ == '__main__':

    system = RuleBasedModel()
    system.run(time=5)

    color = ["black"]*len(system.state)
    for cell in system.cells:
        color[cell.index] = cell.color
    
    print "Number of cells ", str(len(system.cells))
    print "Number of voxels ", str(system.lattice.num_vertices())

    coords = system.lattice.coordinates()
    celldatalist = []
    for cell in system.cells:
        center = coords[cell.index,:]
        celldata = list(center) + [0.01] +[color[cell.index]]
        celldatalist.append(celldata)

    with open('3dcellstest.dat','w') as fh:
        fh.write(str(celldatalist))




