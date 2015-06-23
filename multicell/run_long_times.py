from rulebased3D import RuleBasedModel
import numpy
import math
import pickle

def analyze_system(system):
    dist1 = chain_reaction(system,"lilac")
    dist2 = chain_reaction(system,"yellow")
    dist3 = chain_reaction(system,"cyan")
    dist4 = chain_reaction(system,"red")
    dist = dist1+dist2+dist3+dist4
    
    dist = numpy.array(dist)/0.0078
    dist1=numpy.array(dist1)/0.0078
    dist2=numpy.array(dist2)/0.0078
    dist3=numpy.array(dist3)/0.0078
    dist4=numpy.array(dist4)/0.0078
    
    means = [numpy.mean(dist1),numpy.mean(dist2),numpy.mean(dist3),numpy.mean(dist4)]
    stds = [numpy.std(dist1),numpy.std(dist2),numpy.std(dist3),numpy.std(dist4)]
    return (dist, means, stds)


def record_simulation(system):
    pos_cyan=get_positions(system,"cyan")
    pos_yellow=get_positions(system,"yellow")
    pos_red=get_positions(system,"red")
    pos_lilac=get_positions(system,"lilac")
    pos_white = get_positions(system, "white")
    return {"cyan":pos_cyan, "yellow":pos_yellow, "red":pos_red, "lilac":pos_lilac, "white":pos_white}

def get_positions(system,clone):
    # Get a copy of the heap for the given clone/color
    pos = []
    vtx = system.lattice.coordinates()
    
    for cell in system.cells:
        if cell.color == clone:
            pos.append(vtx[cell.index])
    
    return pos

def chain_reaction(system,clone):
    """ This funciton mimics how the expeimental data analysis was done """
    # Get a copy of the heap for the given clone/color
    pos = []
    vtx = system.lattice.coordinates()
    
    for cell in system.cells:
        if cell.color == clone:
            pos.append(vtx[cell.index])
    
    # Find a cell on the circumference of the clone
    p1 = numpy.mean(pos, axis=0)
    maxdist = 10^9
    dists = []
    for p in pos:
        dists.append(math.sqrt((p1[0]-p[0])*(p1[0]-p[0])+ (p1[1]-p[1])*(p1[1]-p[1])+ (p1[2]-p[2])*(p1[2]-p[2])))
    start_cell_ix = numpy.argmax(numpy.array(dists))
    
    cell_index = start_cell_ix
    d = []
    cell_count=0
    while len(pos) > 1 and cell_count <30:
        p1 = pos.pop(cell_index)
        dists = []
        for p in pos:
            dists.append(math.sqrt((p1[0]-p[0])*(p1[0]-p[0])+ (p1[1]-p[1])*(p1[1]-p[1])+ (p1[2]-p[2])*(p1[2]-p[2])))
            ix = numpy.argmin(numpy.array(dists))
        d.append(dists[ix])
        cell_count += 1
        cell_index = ix
    return d

# Mesh resolution factor 80 corresponds roughly to 10mum cell radius.
system = RuleBasedModel(cell_size=0.02, mesh_resolution=110,mesh_type="unstructured", variance_division_time=10, pushing_factor=0.1,mean_migration_time=1e9,gradient="radial",polarization_strength=1000)

results = {}
positions=record_simulation(system)
results["initial_condition"]=positions
with open("initial_condition.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

# Run the first experiment, No migration, no polarization, just random cell divisions and pushing
N = 1
Tf=3.5
# Since we have 4 clones with unique colors in each simulation, this results in 24 different clones,
#close to the 22 in the experimental dataset.
dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.1
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=1e20
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c01_no_polarization_no_migration_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

# Run the second experiment, No migration, no polarization, just random cell divisions and pushing
# Since we have 4 clones with unique colors in each simulation, this results in 24 different clones,
#close to the 22 in the experimental dataset.
dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.5
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=1e20
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c05_no_polarization_no_migration_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

# Run the second experiment, No migration, no polarization, just random cell divisions and pushing
# Since we have 4 clones with unique colors in each simulation, this results in 24 different clones,
#close to the 22 in the experimental dataset.
dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.9
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=1e20
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c09_no_polarization_no_migration_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

# Run the first experiment, No migration, no polarization, just random cell divisions and pushing
# Since we have 4 clones with unique colors in each simulation, this results in 24 different clones,
#close to the 22 in the experimental dataset.
dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.1
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=0.42/3.0
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c01_no_polarization_migration3_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

# Run the second experiment, No migration, no polarization, just random cell divisions and pushing
# Since we have 4 clones with unique colors in each simulation, this results in 24 different clones,
#close to the 22 in the experimental dataset.
dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.5
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=0.42/3.0
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c05_no_polarization_migration3_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

# Run the second experiment, No migration, no polarization, just random cell divisions and pushing
# Since we have 4 clones with unique colors in each simulation, this results in 24 different clones,
#close to the 22 in the experimental dataset.
dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.9
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=0.42/3.0
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c09_no_polarization_migration3_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.1
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=0.42/6.0
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c01_no_polarization_migration6_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

# Run the second experiment, No migration, no polarization, just random cell divisions and pushing
# Since we have 4 clones with unique colors in each simulation, this results in 24 different clones,
#close to the 22 in the experimental dataset.
dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.5
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=0.42/6.0
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c05_no_polarization_migration6_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))

# Run the second experiment, No migration, no polarization, just random cell divisions and pushing
# Since we have 4 clones with unique colors in each simulation, this results in 24 different clones,
#close to the 22 in the experimental dataset.
dists=[]
stds = []
results = {}
for i in xrange(N):
    system.reset()
    system.b = 0
    system.c = 0.9
    system.mean_division_time=0.42 # -->10hr on average
    system.mean_migration_time=0.42/6.0
    system.run(Tf)
    positions=record_simulation(system)
    results["simulation_{0}".format(i)]=positions
    print i

with open("c09_no_polarization_migration6_Tf4.pyb","wb") as fh:
    fh.write(pickle.dumps(results))





