import os
import numpy as np

from skimage.io import imread
from abc import abstractmethod
from cytomine import Cytomine
from cytomine.models._utilities import parallel
from sldc import Tile, TileExtractionException, TileTopology, alpha_rasterize
from sldc_cytomine import CytomineSlide


class CytomineDownloadableTile(Tile):
  """An abstract tile implementation that downloads the tile from a server and caches it in a local folder"""
  def __init__(self, working_path, parent, offset, width, height, tile_identifier=None, polygon_mask=None):
    Tile.__init__(self, parent, offset, width, height, tile_identifier=tile_identifier, polygon_mask=polygon_mask)
    self._working_path = working_path

  @abstractmethod
  def download_tile_image(self):
    pass

  @property
  def cache_filename(self):
    image_instance = self.base_image.image_instance
    x, y, width, height = self.abs_offset_x, self.abs_offset_y, self.width, self.height
    zoom = self.base_image.zoom_level
    cache_filename_format = "{id}-{zoom}-{x}-{y}-{w}-{h}.png"
    return cache_filename_format.format(id=image_instance.id, x=x, y=y, w=width, h=height, zoom=zoom)

  @property
  def cache_filepath(self):
    return os.path.join(self._working_path, self.cache_filename)

  @property
  def np_image(self):
    try:
      if not os.path.exists(self.cache_filepath) and not self.download_tile_image():
        raise TileExtractionException("Cannot fetch tile at for '{}'.".format(self.cache_filename))

      np_array = imread(self.cache_filepath)

      # if np_array.shape[:2] != (self.height, self.width) or  np_array.shape[2] != self.base_image.channels:
      #   raise TileExtractionException("Fetched image has invalid size : {} instead "
      #                                 "of {}".format(np_array.shape, (self.width, self.height, self.channels)))

      if np_array.shape[2] == 4:
        np_array = np_array[:, :, 3]
      np_array = np_array.astype("uint8")
      return self.add_polygon_mask(np_array)
    except IOError as e:
      raise TileExtractionException(str(e))

  def add_polygon_mask(self, image):
    try:
      return alpha_rasterize(image, self.polygon_mask)
    except:
      return image


class CytomineZoomifyTile(CytomineDownloadableTile):
  """Tile fetch using the Zoomify protocol (for older Cytomine versions)"""
  def download_tile_image(self):
    slide = self.base_image
    col_tile = self.abs_offset_x // 256
    row_tile = self.abs_offset_y // 256
    _slice = slide.image_instance
    response = Cytomine.get_instance().get('imaging_server.json', None)
    imageServerUrl = response['collection'][0]['url']
    return Cytomine.get_instance().download_file(imageServerUrl + "/image/tile", self.cache_filepath, False, payload={
      "zoomify": _slice.fullPath,
      "mimeType": _slice.mime,
      "x": col_tile,
      "y": row_tile,
      "z": slide.api_zoom_level
    })


class CytomineIIPTile(CytomineDownloadableTile):
  def download_tile_image(self):
    slide = self.base_image
    if not isinstance(slide, CytomineSlide):
      raise TypeError(f"CytomineIIP tile should be used in conjunction with CytomineSlide only (as base image), found `{type(slide)}`")
    iip_topology = TileTopology(slide, None, max_width=256, max_height=256, overlap=0)
    col_tile = self.abs_offset_x // 256
    row_tile = self.abs_offset_y // 256
    iip_tile_index = col_tile + row_tile * iip_topology.tile_horizontal_count
    _slice = slide.slice_instance
    return Cytomine.get_instance().download_file(_slice.imageServerUrl + "/slice/tile", self.cache_filepath, False, payload={
      "fif": _slice.path,
      "mimeType": _slice.mime,
      "tileIndex": iip_tile_index,
      "z": slide.api_zoom_level
    })


class CytominePimsTile(CytomineDownloadableTile):
  """Tile fetch using the Zoomify protocol (for older Cytomine versions)"""
  def download_tile_image(self):
    slide = self.base_image
    if not isinstance(slide, CytomineSlide):
      raise TypeError(f"CytominePims tile should be used in conjunction with CytomineSlide only (as base image), found `{type(slide)}`")
    iip_topology = TileTopology(slide, None, max_width=256, max_height=256, overlap=0)
    col_tile = self.abs_offset_x // 256
    row_tile = self.abs_offset_y // 256
    tile_index = col_tile + row_tile * iip_topology.tile_horizontal_count
    _slice = slide.slice_instance
    zoom = self.base_image.image_instance.zoom - slide.zoom_level

    return Cytomine.get_instance().download_file(f"{_slice.imageServerUrl}/image/{_slice.path}/normalized-tile/zoom/{zoom}/ti/{tile_index}.jpg", self.cache_filepath, False, payload={
      "z_slices": "0",
      "timepoints": "0",
      "channels": "0,1,2",
      "colormaps": "#f00,#0f0,#00f",
    })

class CytominePims20241Tile(CytomineDownloadableTile):
  """Tile fetch using the PIMS CE/EET2024.1 """
  def download_tile_image(self):
    slide = self.base_image
    if not isinstance(slide, CytomineSlide):
      raise TypeError(f"CytominePims tile should be used in conjunction with CytomineSlide only (as base image), found `{type(slide)}`")
    col_tile = self.abs_offset_x // 256
    row_tile = self.abs_offset_y // 256
    _slice = slide.slice_instance
    zoom = self.base_image.image_instance.zoom - slide.zoom_level

    return Cytomine.get_instance().download_file(f"sliceinstance/{_slice.id}/normalized-tile/zoom/{zoom}/tx/{col_tile}/ty/{row_tile}.jpg", self.cache_filepath, False, payload={
    "z_slices": "0",
    "timepoints": "0",
    "channels": "0,1,2",
    "colormaps": "#f00,#0f0,#00f",
    })
    # Not compatible with 2024.1, rather use


class CytomineWindowTile(CytomineDownloadableTile):
  """Tiling using the window service"""
  def download_tile_image(self):
    return self.base_image.image_instance.window(
      x=self.abs_offset_x,
      y=self.abs_offset_y,
      w=self.width,
      h=self.height,
      dest_pattern=self.cache_filepath
    )


class CytomineTile(Tile):
  def __init__(self, working_path, parent, offset, width, height, tile_class=CytomineIIPTile, tile_identifier=None, polygon_mask=None, n_jobs=1):
    """A abritrarily sized cytomine tile. Will be re-constructed by fetching smaller tiles using the specified
    protocol.

    Parameters
    ----------
    parent: Image
        The image from which is extracted the tile
    offset: (int, int)
        The x and y coordinates of the pixel at the origin point of the slide in the parent image.
        Coordinates order is the following : (x, y).
    width: int
        The width of the tile
    height: int
        The height of the tile
    tile_class: class
        A subclass of 'CytomineDownloadableTile', specifies the download protocol for underlying tiles.
        The current tile will be built by downloading non-overlapping 256x256 tiles using the given protocol and
        assembling them into the expected tile. By default, uses IIP through Cytomine API (class `CytomineIIPTile`).
    tile_identifier: int, optional (default: None)
        A integer identifier that identifies uniquely the tile among a set of tiles
    polygon_mask: Polygon (optional, default: None)
        The polygon representing the alpha mask to apply to the tile window


    Notes
    -----
    The coordinates origin is the leftmost pixel at the top of the slide
    """
    Tile.__init__(self, parent, offset, width, height, tile_identifier=tile_identifier, polygon_mask=polygon_mask)
    self._working_path = working_path
    self._n_jobs = n_jobs
    self._tile_class = tile_class
    os.makedirs(working_path, exist_ok=True)

  def _pad_iip_tile(self, img):
    padding = [(0, 256 - img.shape[0]), (0, 256 - img.shape[1])]
    if img.ndim == 3:
      padding += [(0, 0)]
    return np.pad(img, padding, mode='constant', constant_values=0)

  def _iip_window(self):
    left_margin = self.abs_offset_x % 256
    top_margin = self.abs_offset_y % 256
    right_margin = 256 - (self.abs_offset_x + self.width) % 256
    bottom_margin = 256 - (self.abs_offset_y + self.height) % 256
    margins = [top_margin, left_margin, bottom_margin, right_margin]

    offset = self.abs_offset_x - left_margin, self.abs_offset_y - top_margin
    width = self.width + left_margin + right_margin
    height = self.height + top_margin + bottom_margin
    window = self.base_image.window(offset=offset, max_width=width, max_height=height)

    return window, offset, (width, height), margins

  @property
  def np_image(self):
    from sldc_cytomine.tile_builder import CytomineGenericTileBuilder
    window, _, (width, height), margins = self._iip_window()
    builder = CytomineGenericTileBuilder(self._tile_class, self._working_path)
    topology = TileTopology(window, builder, max_width=256, max_height=256, overlap=0)

    def download_tile(tile):
      return tile.np_image

    rebuilt = np.zeros([height, width, self.base_image.channels], dtype=np.uint8)
    for tile, tile_image in parallel.generic_download(list(topology), download_tile, n_workers=self._n_jobs):
      y_start, x_start = tile.offset_y, tile.offset_x
      y_end, x_end = y_start + 256, x_start + 256
      rebuilt[y_start:y_end, x_start:x_end] = self._pad_iip_tile(tile_image)

    return rebuilt[margins[0]:-margins[2], margins[1]:-margins[3]]

  def fetch_subtiles(self):
    """fetch underlying tiles without loading them into memory"""
    from sldc_cytomine.tile_builder import CytomineGenericTileBuilder
    window, _, _, _ = self._iip_window()
    builder = CytomineGenericTileBuilder(self._tile_class, self._working_path)
    topology = TileTopology(window, builder, max_width=256, max_height=256, overlap=0)

    def download_tile(tile: CytomineDownloadableTile):
      return tile.download_tile_image()

    _ = parallel.generic_download(list(topology), download_tile, n_workers=self._n_jobs)
