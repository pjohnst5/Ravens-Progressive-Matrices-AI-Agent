from ImageHelpers import translate_to_find_best

class CompositeTransform():
  def __init__(self, score, base_transform, combo):
    self.score = score
    self.base_transform = base_transform
    self.combo = combo

  def execute(self, img1, img2=None):
    if img2 is not None:
      return self.base_transform(img1, img2)
    return self.base_transform(img1)

  def __str__(self):
    msg = str(self.combo)
    msg += str(self.base_transform)
    msg += str(self.score)
    return msg
