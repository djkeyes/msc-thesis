import numpy as np
import skimage.io as skio
from mayavi import mlab
import trimesh


def main():

  print 'loading TSDF volume...'
  src = mlab.pipeline.open('/home/daniel/Downloads/tsdf/fire.mhd')
  print 'loaded volume, extracting isosurface...'

  my_obj = mlab.pipeline.iso_surface(src, contours=[0])
  mlab.show()

  print 'Isosurface mesh successful. Extracting raw vertex data...'

  my_actor = my_obj.actor.actors[0]
  poly_data_object = my_actor.mapper.input

  vertices = np.zeros((len(poly_data_object.points), 3), np.float64)
  for i in range(len(poly_data_object.points)):
    vertices[i, :] = poly_data_object.points[i]
  print '\tvertices done!'
  faces = np.zeros((len(poly_data_object.polys.data) / 4, 3), np.int)
  for i in range(len(poly_data_object.polys.data) / 4):
    # these are always triangles, so poly_data_object.polys.data[4*i] should always be 3
    assert (poly_data_object.polys.data[4 * i] == 3)
    faces[i, 0] = poly_data_object.polys.data[4 * i + 1]
    faces[i, 1] = poly_data_object.polys.data[4 * i + 2]
    faces[i, 2] = poly_data_object.polys.data[4 * i + 3]
  print '\tfaces done!'

  print 'Data extracted. Building Trimesh object...'

  mesh = trimesh.Trimesh(vertices, faces)
  print 'Trimesh built. Exporting to file...'
  mesh.export('mesh.stl')
  print 'COMPLETE!'

  # so we have the mesh (defined by points and tuples of points).
  # we just need to save that, the load it in something that can do raycasting.
  # Then to test, we:
  # pick 2 images.
  # For every point in the first image, we raycast.
  #    If the ray didn't hit anything, continue
  #    otherwise the ray hit a cell
  #    check every vertex in the cell to see if one projects onto the ray's origin pixel
  #    if so, project the vertex into the second image, then raycast from the second image (to check for obscurance)
  #    if visible in both images, it is a true positive
  #    compute descriptor in first image, and descriptor in second image. Also search in second image for
  #       NN descriptor to descriptor in first image (but exclude radius around matched point)
  #    if NN is the original descriptor, we win (found true positive). if it isn't, we lose (correspondence is false
  #       negative, and the NN is a false positive)
  #
  # note: this will be super slow, since we must compute all descriptors in second image. gross.
  # computing takes W*H*D, as does exhaustive comparison. So 39mil per image ~ 1 second. RIP. maybe we can do a few thousand.
  # but also note: we don't need ALL the descriptors in the second image. just all the ones satisfying the aforementioned
  # self-projection test.


if __name__ == "__main__":
  main()
