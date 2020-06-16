# -*- coding: utf-8 -*-
import os
import warnings

import numpy as np
import PIL
from PIL.Image import fromarray
from cytomine import Cytomine
from cytomine.models import ImageInstance
from shapely.geometry import Polygon, box
from sldc import TileExtractionException, alpha_rasterize, Image, Tile, TileBuilder, TileTopology

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "1.0"


class CytomineSlide(Image):
    """
    A slide from a cytomine project
    """

    def __init__(self, id_img_instance, zoom_level=0):
        """Construct CytomineSlide objects

        Parameters
        ----------
        id_img_instance: int
            The id of the image instance
        zoom_level: int
            The zoom level at which the slide must be read. The maximum zoom level is 0 (most zoomed in). The greater
            the value, the lower the zoom.
        """
        if zoom_level > 0:
            warnings.warn("when using zoom_level > 0, tile width, height and "
                          "overlap should be repesctively set to 256, 256 and 0")
        self._img_instance = ImageInstance().fetch(id_img_instance)
        self._slice_instance = self._img_instance.reference_slice()
        if zoom_level > self._img_instance.zoom:
            raise ValueError("invalid number of zoom levels selected ({}, max={})".format(
                zoom_level, self._img_instance.zoom))
        self._zoom_level = zoom_level

    @property
    def image_instance(self):
        return self._img_instance

    @property
    def slice_instance(self):
        return self._slice_instance

    @property
    def np_image(self):
        raise NotImplementedError("Disabled due to the too heavy size of the images")

    @property
    def width(self):
        return self._img_instance.width // (2 ** self.zoom_level)

    @property
    def height(self):
        return self._img_instance.height // (2 ** self.zoom_level)

    @property
    def channels(self):
        return 3

    @property
    def zoom_level(self):
        return self._zoom_level

    @property
    def api_zoom_level(self):
        """The zoom level used by cytomine api uses 0 as lower level of zoom (most zoomed out). This property
        returns a zoom value that can be used to communicate with the backend."""
        return self._img_instance.zoom - self.zoom_level

    def __str__(self):
        return "CytomineSlide (#{}) ({} x {}) (zoom: {})".format(self._img_instance.id, self.width, self.height, self.zoom_level)


class CytomineTile(Tile):
    """
    A tile from a cytomine slide
    """
    def __init__(self, working_path, parent, offset, width, height, tile_identifier=None, polygon_mask=None):
        """Constructor for CytomineTile objects

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

    @property
    def np_image(self):
        try:
            image_instance = self.base_image.image_instance
            x, y, width, height = self.abs_offset_x, self.abs_offset_y, self.width, self.height
            zoom = self.parent.zoom_level

            # check if the tile was cached
            cache_filename_format = "{id}-{zoom}-{x}-{y}-{w}-{h}.png"
            cache_filename = cache_filename_format.format(id=image_instance.id, x=x, y=y, w=width, h=height, zoom=zoom)
            cache_path = os.path.join(self._working_path, cache_filename)
            success = True
            if not os.path.exists(cache_path):
                if zoom == 0:
                    success = self._get_tile_no_zoom(cache_path)
                else:
                    success = self._get_tile_with_zoom(cache_path)

            if not success:
                raise TileExtractionException("Cannot fetch tile at for "
                                              "'{}'.".format(cache_filename_format.split(".", 1)[0]))

            # load image
            np_array = np.asarray(PIL.Image.open(cache_path))
            if np_array.shape[1] != width or np_array.shape[0] != height \
                    or np_array.shape[2] < self._underlying_image_channels:
                raise TileExtractionException("Fetched image has invalid size : {} instead "
                                              "of {}".format(np_array.shape, (width, height, self.channels)))

            # drop alpha channel if there is one
            if np_array.shape[2] >= 4:
                np_array = np_array[:, :, 0:3]
            np_array = np_array.astype("uint8")
            if self.polygon_mask is None:
                return np_array
            else:
                try:
                    return alpha_rasterize(np_array, self.polygon_mask)
                except ValueError:
                    return np_array
        except IOError as e:
            raise TileExtractionException(str(e))

    def _get_tile_with_zoom(self, path):
        parent = self.parent
        iip_topology = TileTopology(parent, None, max_width=256, max_height=256, overlap=0)
        col_tile = self.abs_offset_x // 256
        row_tile = self.abs_offset_y // 256
        iip_tile_index = col_tile + row_tile * iip_topology.tile_horizontal_count
        _slice = parent.slice_instance
        return Cytomine.get_instance().download_file(_slice.imageServerUrl + "/slice/tile", path, False, payload={
            "fif": _slice.path,
            "mimeType": _slice.mime,
            "tileIndex": iip_tile_index,
            "z": parent.api_zoom_level
        })

    def _get_tile_no_zoom(self, path):
        return self.parent.image_instance.window(
            x=self.abs_offset_x,
            y=self.abs_offset_y,
            w=self.width,
            h=self.height,
            dest_pattern=path
        )

    @property
    def channels(self):
        return 3 if self.polygon_mask is None else 4

    def _tile_box(self):
        offset_x, offset_y = self.abs_offset
        return box(offset_x, offset_y, offset_x + self.width, offset_y + self.height)


class CytomineTileBuilder(TileBuilder):
    """
    A builder for CytomineTile objects
    """

    def __init__(self, working_path):
        """Construct CytomineTileBuilder objects

        Parameters
        ----------
        working_path:
            A writable working path for the tile builder
        """
        self._working_path = working_path

    def build(self, image, offset, width, height, polygon_mask=None):
        return CytomineTile(self._working_path, image, offset, width, height, polygon_mask=polygon_mask)


class TileCache(object):
    """A class to use jointly with tiles to avoid fetching them several time
    """
    def __init__(self, tile_builder, working_path):
        self._tile_builder = tile_builder
        self._working_path = working_path

    def fetch_and_cache(self, tile):
        """Fetch the np_image for the passed tile and cache it in the working path. If the np_image was already
        cached nothing is fetched from the server.
        Parameters
        ----------
        tile: Tile
            The tile of which the np_image must be fetched and cached
        Returns
        -------
        path: string
            The full path to which was cached the np_image
        """
        if not self._cache_has(tile, alpha=False):
            self._save(tile, tile.np_image.astype("uint8"), alpha=False)
        return self._tile_path(tile, alpha=False)

    def polygon_fetch_and_cache(self, image, polygon, alpha=True):
        """Fetch the np_image for the tile boxing the passed polygon and cache it in the working path. If the np_image
        was already cached nothing is fetched from the server.
        Parameters
        ----------
        image: Image
            The image from which the tile must be extracted
        polygon: Polygon
            The polygon that should be boxed by the tile
        alpha: bool
            True of applying an alpha mask structured like the polygon

        Returns
        -------
        path: string
            The full path to which was cached the np_image
        """
        tile = image.tile_from_polygon(self._tile_builder, polygon, mask=alpha)
        if not self._cache_has(tile, alpha=alpha):
            np_image = tile.np_image
            self._save(tile, np_image.astype("uint8"), alpha=alpha)
        return self._tile_path(tile, alpha)

    def tile_np_image(self, tile):
        """Get the np_image of the given tile from the cache. If it was not cached, fetch it from the
        server and cache it before returning it.

        Parameters
        ----------
        tile: Tile
            The tile of which the np_image must be fetched and cached

        Returns
        -------
        np_image: array-like
            The image representation
        """
        path = self.fetch_and_cache(tile)
        return np.asarray(PIL.Image.open(path)).astype("uint8")

    def polygon_np_image(self, image, polygon, alpha=True):
        """Get the np_image of the tile that boxes the polygon in the image from the cache. If it was not cached,
        fetch it from the server and cache it before returning it.

        Parameters
        ----------
        image: Image
            The image from which the tile must be extracted
        polygon: Polygon
            The polygon that should be boxed by the tile
        alpha: bool
            True of applying an alpha mask structured like the polygon

        Returns
        -------
        np_image: array-like
            The image representation
        """
        return self.tile_np_image(image.tile_from_polygon(self._tile_builder, polygon, mask=alpha))

    def _save(self, tile, np_image, alpha=False):
        """Save the tile np_image at the path produced by _tile_path
        Parameters
        ----------
        tile: Tile
            The tile from which was generated np_image
        np_image: array-like
            The numpy image to save
        alpha: bool (optional, default: False)
            True if the np_image has an alpha channel
        """
        fromarray(np_image).save(self._tile_path(tile, alpha))

    def _cache_has(self, tile, alpha=False):
        """Check whether the given tile was already cached by the tile cache
        Parameters
        ----------
        tile: Tile
            The tile
        alpha: bool (optional, default: False)
            True if the alp
        :return:
        """
        return os.path.isfile(self._tile_path(tile, alpha))

    def _tile_path(self, tile, alpha=False):
        """Return the path where to store the tile

        Parameters
        ----------
        tile: Tile
            The tile object containing the image to store
        alpha: bool (optional, default: False)
            True if an alpha mask is applied

        Returns
        -------
        path: string
            The path in which to store the image
        """
        basename = "{}_{}_{}_{}_{}".format(tile.base_image.image_instance.id, tile.offset_x,
                                           tile.offset_y, tile.width, tile.height)
        if alpha:
            basename = "{}_alpha".format(basename)
        return os.path.join(self._working_path, "{}.png".format(basename))
