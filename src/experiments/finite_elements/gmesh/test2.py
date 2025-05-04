import gmsh
gmsh.initialize()
gmsh.clear()


center_x = 0
center_y = 0
center_z = 0
radius = 10

gmsh.model.add("circle_extrusion")
circle = gmsh.model.occ.addCircle(center_x,center_y,center_z,radius,tag=-1)
surface = gmsh.model.occ.addPlaneSurface([circle],tag = -1)
gmsh.model.occ.synchronize()
base_surf_pg = gmsh.model.addPhysicalGroup(2,[surface],tag = 100,name="lower_surface")
h = 1
subdivision = [10]
extrusion = gmsh.model.occ.extrude([(2, surface)], 0, 0, h, subdivision)
gmsh.model.occ.synchronize()

volume = gmsh.model.addPhysicalGroup(3, [extrusion1[1][1]], name="volume")
lateral_surf_group = gmsh.model.addPhysicalGroup(2, [extrusion[2][1]], tag = 101, name="lateral_surface")
upper_surf_group = gmsh.model.addPhysicalGroup(2, [extrusion[0][1]], tag = 102, name="upper_surface")
gmsh.model.mesh.generate(3)
gmsh.model.mesh.refine()
gmsh.option.setNumber("Mesh.MshFileVersion", 2) #save in ASCII 2 format
gmsh.write("filename.msh")
