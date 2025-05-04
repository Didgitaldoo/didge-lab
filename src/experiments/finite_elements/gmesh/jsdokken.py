# https://jsdokken.com/src/tutorial_gmsh.html


from dolfinx.io import XDMFFile
from dolfinx.mesh import meshtags_from_entities
from dolfinx.cpp.mesh import cell_entity_type
from dolfinx.io import distribute_entity_data
from dolfinx.graph import adjacencylist
from dolfinx.mesh import create_mesh
from dolfinx.cpp.mesh import to_type
from dolfinx.cpp.io import perm_gmsh
import numpy
from dolfinx.io import gmshio
from mpi4py import MPI
from dolfinx.io.gmshio import model_to_mesh
import numpy as np
import gmsh
import warnings

warnings.filterwarnings("ignore")
gmsh.initialize()

gmsh.model.add("DFG 3D")
L, B, H, r = 2.5, 0.41, 0.41, 0.05

import math 

def main():
    gmsh.initialize()
    
    # alias to facilitate code writing
    factory = gmsh.model.geo
    
    # default mesh size
    lc = 1.
    
    # Geometry
    # points
    p1 = factory.addPoint(0., 0., 0., lc)
    p2 = factory.addPoint(10., 0., 0., lc)
    p3 = factory.addPoint(0., 10., 0., lc)
    p4 = factory.addPoint(4., 0., 0., lc)
    p5 = factory.addPoint(0., 4., 0., lc)
    p6 = factory.addPoint(4., 4., 0., lc)
    angle = math.pi/4.
    p7 = factory.addPoint(10*math.cos(angle), 10*math.sin(angle),       0., lc)
    
    # lines
    l1 = factory.addLine(p5, p6)
    l2 = factory.addLine(p6, p4)
    l3 = factory.addLine(p4, p1)
    l4 = factory.addLine(p1, p5)
    l5 = factory.addLine(p4, p2)
    l6 = factory.addLine(p5, p3)
    l7 = factory.addLine(p6, p7)
    l8 = factory.addCircleArc(p2, p1, p7)
    l9 = factory.addCircleArc(p7, p1, p3)
    
    # curve loops
    cl1 = factory.addCurveLoop([l3, l4, l1, l2])
    cl2 = factory.addCurveLoop([l7, l9, -l6, l1])
    cl3 = factory.addCurveLoop([l5, l8, -l7, l2])
    
    # surfaces
    s1 = factory.addPlaneSurface([cl1])
    s2 = factory.addPlaneSurface([cl2])
    s3 = factory.addPlaneSurface([cl3])
    
    # extrusions
    dx = 5.
    num_els_z = 10
    factory.extrude([(2, s1), (2, s2), (2, s3)], 0., 0., dx,
                    numElements=[num_els_z], recombine=True)
    
    factory.synchronize()
    
    # Meshing
    meshFact = gmsh.model.mesh
    
    # transfinite curves
    n_nodes = 10
    # "Progression" 1 is default
    meshFact.setTransfiniteCurve(l1, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l2, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l3, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l4, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l5, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l6, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l7, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l8, numNodes=n_nodes)
    meshFact.setTransfiniteCurve(l9, numNodes=n_nodes)
    # transfinite surfaces
    meshFact.setTransfiniteSurface(s1)
    meshFact.setTransfiniteSurface(s2)
    meshFact.setTransfiniteSurface(s3)
    
    # mesh
    meshFact.generate(2)
    meshFact.recombine()
    meshFact.generate(3)
    
    gmsh.fltk.run()
    
    gmsh.finalize()


if __name__ == "__main__":
    main()
