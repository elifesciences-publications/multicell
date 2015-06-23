from rulebased_columns import RuleBasedModel
import numpy
import math
import pickle

def cell_position_by_color(system):
    cell_positions={}
    coords= system.lattice.coordinates()
    
    for cell in system.cells:
        try:
            cell_positions[cell.color].append(coords[cell.index,:])
        except:
            cell_positions[cell.color]=[coords[cell.index,:]]
    
    return cell_positions

def cartilage_score_function(positions):
    cell_count=0
    score = 0.0
    for color,pos in positions.items():
        if not color=="white":
            v= numpy.array(pos)
            y = v[0:,1]
            cell_count += len(v)
            i = numpy.argsort(y)
            v= v[i,:]
            for x in numpy.diff(v,axis=0):
                score += abs((numpy.dot(x/numpy.linalg.norm(x),[0,1,0])))
    return score/cell_count

# Mesh resolution factor 160 corresponds roughly to 7.6mum cell radius.
system = RuleBasedModel(cell_size=0.02, mesh_resolution=160, mesh_type="unstructured", variance_division_time=0.01,
                        pushing_factor=0.1,mean_migration_time=1e20,gradient="onesided_linear",polarization_strength=1)


coeffs = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
N = 10

for c in coeffs:
    s=[]
    for i in xrange(N):
        system.reset()
        system.c=c
        system.run(1)
        results=cell_position_by_color(system)
        s.append(cartilage_score_function(results))
        print s

        with open("cartilage/data/cartilage_c_no_polarization_{0}_{1}.pyb".format(c,i),"wb") as fh:
            fh.write(pickle.dumps(results))
        print "Competed one iteration"



