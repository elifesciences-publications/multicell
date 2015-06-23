from rulebased_columns import *
import numpy
import math

def cartilage_score_function(positions):
    vector_count=0
    score = 0.0
    for color,pos in positions.items():
        if not color=="white":
            v= numpy.array(pos)
            y = v[0:,1]
            vector_count += len(v)-1
            i = numpy.argsort(y)
            v= v[i,:]
            for x in numpy.diff(v,axis=0):
                score += abs((numpy.dot(x/numpy.linalg.norm(x),[0,1,0])))
    return score/vector_count

# Mesh resolution factor 80 corresponds roughly to 10mum cell radius.
system = RuleBasedModel(cell_size=0.02, mesh_resolution=160, mesh_type="unstructured",       variance_division_time=0.01,
                        pushing_factor=0.1,mean_migration_time=1e20,gradient="twosided_linear",polarization_strength=1000)

result = system.cell_position_by_color()
print_html_free(result,0.05, "initial_condition_snapshot.html")

#system.reset()
system.b = 1000
system.c = 0.99
system.mean_division_time = 0.42 # -->10hr on average
system.run(1)
result = system.cell_position_by_color()
print_html_free(result, 0.05)

print cartilage_score_function(result)
#system.print_html()









