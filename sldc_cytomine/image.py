# -*- coding: utf-8 -*-
from sldc import Image
from abc import abstractmethod
from cytomine.models import ImageInstance


__author__ = "Mormont Romain <romain.mormont@gmail.com>"


class AbstractCytomineSlide(Image):
  def __init__(self, image_instance: ImageInstance, zoom_level: int=0):
    """
    Parameters
    ----------
    img_instance: ImageInstance
      The the image instance
    zoom_level: int
      The zoom level at which the slide must be read. The maximum zoom level is 0 (most zoomed in). The greater
      the value, the lower the zoom.
    """
    self._image_instance = image_instance
    self._zoom_level = zoom_level

  @property
  def image_instance(self):
    return self._image_instance

  @property
  def zoom_level(self):
    return self._zoom_level

  @property
  def channels(self):
    return 3

  @property
  def width(self):
    return self.image_instance.width // (2 ** self.zoom_level)

  @property
  def height(self):
    return self.image_instance.height // (2 ** self.zoom_level)
  
  @property
  def np_image(self):
    raise NotImplementedError("cannot dump a whole image through this method, please use the dump() function")

  @property
  @abstractmethod
  def api_zoom_level(self):
    """The zoom level used by cytomine api uses 0 as lower level of zoom (most zoomed out). 
    This property returns a zoom value that can be used to communicate with the backend."""
    pass

  @classmethod
  def from_id(cls, id_img_instance, zoom_level=0):
    return cls(ImageInstance.fetch(id_img_instance), zoom_level=zoom_level)


class CytomineSlide(AbstractCytomineSlide):
  """
  A cytomine ImageInstance, for cytomine/Cytomine-core version >= v3.2.0
  """
  def __init__(self, img_instance, zoom_level=0):
    super().__init__(img_instance, zoom_level=zoom_level)
    self._slice_instance = self.image_instance.reference_slice()
    if zoom_level > self.image_instance.zoom:
      raise ValueError("invalid number of zoom levels selected ({}, max={})".format(
        zoom_level, self.image_instance.zoom))

  @property
  def slice_instance(self):
    return self._slice_instance

  @property
  def api_zoom_level(self):
    return self.image_instance.zoom - self.zoom_level

  def __str__(self):
    return "Image instance (#{}) ({} x {}) (zoom: {})".format(self.image_instance.id, self.width, self.height, self.zoom_level)


class LegacyCytomineSlide(AbstractCytomineSlide):
  """
  For Cytomine instances working with cytomine/Cytomine-core version < v3.2.0
  """
  @property
  def api_zoom_level(self):
    return self.image_instance.depth - self.zoom_level
