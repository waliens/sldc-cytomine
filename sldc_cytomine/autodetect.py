from tempfile import TemporaryDirectory
from cytomine.models import ImageInstance, SliceInstance
from sldc import TileExtractionException
from sldc_cytomine import CytomineSlide
from sldc_cytomine.image import LegacyCytomineSlide
from sldc_cytomine.tile import CytomineIIPTile, CytomineZoomifyTile

from sldc_cytomine.tile_builder import CytomineTileBuilder


def _infer_has_slice(image_instance: ImageInstance):
  slice = image_instance.reference_slice()
  return isinstance(slice, SliceInstance)


def infer_protocols(image_instance):
  """Detect whether or not the slice endpoints exists and whether or not iip is available
  Parameters
  ----------
  image_instance: ImageInstance|id|str
    A reference to the image instance 

  Returns
  -------
  slide_class: 
    A subclass of AbstractCytomineSlide that can be used to instantiate a SLDC slide
  tile_class:
    A subclass of CytomineDownloadableTile that can be used to fetch tiles with SLDC-Cytomine
  """
  if not isinstance(image_instance, ImageInstance):
    image_instance = ImageInstance().fetch(image_instance)
  
  if image_instance is None or not image_instance:
    raise ValueError("cannot infer protocols, invalid image instance") 

  with TemporaryDirectory() as tmpdir:
    if _infer_has_slice(image_instance):
      slide_class = CytomineSlide
    else: 
      return LegacyCytomineSlide, CytomineZoomifyTile

    slide = slide_class(image_instance) 
    
    # attempt to fetch a tile with iip to check whether or not the 
    # protocol is supported, otherwise fallback to Zoomify
    try:
      iip_builder = CytomineTileBuilder(tmpdir, tile_class=CytomineIIPTile, n_jobs=1)
      tile = iip_builder.build(slide, (0, 0), 256, 256)
      tile.np_image  # if this succeeds
      return slide_class, CytomineIIPTile
    except TileExtractionException:
      return slide_class, CytomineZoomifyTile
