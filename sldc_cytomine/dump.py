
import os

from shapely import wkt
from skimage import io
from shapely.affinity import affine_transform
from cytomine.models import ImageInstance, Annotation
from cytomine.models._utilities.pattern_matching import resolve_pattern

from sldc_cytomine.image import CytomineSlide
from sldc_cytomine.tile_builder import CytomineTileBuilder
from sldc_cytomine.tile import CytomineIIPTile


def _infer_image_region(zone, zoom_level=0, slide_class=CytomineSlide):
  """Infer the sldc.Image that must be extracted based on the parameters

  Parameters
  ----------
  zone: int|str|Annotation|ImageInstance
    Integer and string are interpreted as an image instance identifier. When the argument is a Cytomine Annotation
    the resulting zone is a crop of the annotation in the image it belongs to. Otherwise, the zone is considered to 
    be the whole image. 
  zoom_level: int
    The zoom level
  slide_class:
    A subclass of AbstractCytomineSlide in which to encapsulate the image.
  """
  if isinstance(zone, int) or isinstance(zone, str):
    image = ImageInstance().fetch(zone)
  elif isinstance(zone, ImageInstance):
    image = zone
  elif isinstance(zone, Annotation):
    if not hasattr(zone, "image"):
      raise ValueError("the annotation is missing the attribute `image` containing the image identifier")
    image = ImageInstance().fetch(zone.image)
  else:
    raise TypeError(f"unknown region input type, found '{type(zone)}'")

  slide = slide_class(image, zoom_level=zoom_level)
  if not isinstance(zone, Annotation):
    return slide

  if not hasattr(zone, "location"):
    raise ValueError("the annotation is missing the attribute `wkt` containing polygon coordinates")

  polygon = wkt.loads(zone.location)
  polygon = affine_transform(polygon, [1, 0, 0, -1, 0, image.height])
  polygon = affine_transform(polygon, [1 / 2 ** zoom_level, 0, 0, 1 / 2 ** zoom_level, 0, 0])
  return slide.window_from_polygon(polygon)


def dump_region(
  zone, 
  dest_pattern: str, 
  slide_class=CytomineSlide, 
  tile_class=CytomineIIPTile, 
  zoom_level: int=0, n_jobs=0, 
  working_path=None, plugin=None
):
  """Dump an image from a Cytomine server by downloading it tile by tile (in parallel)

  Parameters
  ----------
  zone: int|str|ImageInstance|Annotation
    The area of the image to dump. For a whole image pass the image id (as an int or str) or 
    an cytomine.models.ImageInstance. For a region, pass a cytomine.models.Annotation.
  dest_pattern: str
    The destination path pattern. Can contain placeholder '{property}' to be replaced with attributes 
    of 'zone'.
  slide_class: 
    A subclass of AbstractCytomineSlide
  tile_class:
    A subclass of CytomineDownloadableTile 
  zoom_level: int
    The zoom level at which the image must be dumped
  working_path: str
    The working path where to store the downloaded tiles. This path should exist until the dump function
    has exited.
  plugin:
    A plugin for saving the image (see `skimage.io.imsave`)
  """
  if working_path is None:
    working_path = os.getcwd()

  dump_paths = resolve_pattern(dest_pattern, zone)
  if len(dump_paths) != 1:
    raise ValueError("pattern '{}' does not resolve into a unique path".format(dest_pattern))
  dump_path = dump_paths[0]

  # defined dump region
  region = _infer_image_region(zone, zoom_level=zoom_level, slide_class=slide_class)
  tile_builder = CytomineTileBuilder(working_path, tile_class=tile_class, n_jobs=n_jobs)

  # this tile represents the image to dump
  tile = tile_builder.build(region, (0, 0), region.width, region.height)
  
  # load in memory 
  # TODO infilew riting
  img = tile.np_image
  io.imsave(dump_path, img,  check_contrast=False, plugin=plugin)
  return dump_path
  