import cv2
import numpy as np

TWO_BY_TWO = '2x2'
THREE_BY_THREE = '3x3'

# Base transforms
def Identity(img):
  return img

def Rotate_90(img):
  M = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), -90, 1)
  img = cv2.warpAffine(img, M, img.shape)
  return img


def Rotate_180(img):
  M = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), 180, 1)
  img = cv2.warpAffine(img, M, img.shape)
  return img


def Rotate_270(img):
  M = cv2.getRotationMatrix2D((img.shape[0] / 2, img.shape[1] / 2), -270, 1)
  img = cv2.warpAffine(img, M, img.shape)
  return img


def Identity_Flip(img):
  return cv2.flip(img, 0)


def Rotate_90_Flip(img):
  return Identity_Flip(Rotate_90(img))


def Rotate_180_Flip(img):
  return Identity_Flip(Rotate_180(img))


def Rotate_270_Flip(img):
  return Identity_Flip(Rotate_270(img))

def AND(img1, img2):
  and_result = cv2.bitwise_and(img1, img2)
  return and_result

def OR(img1, img2):
  or_result = cv2.bitwise_or(img1, img2)
  return or_result

def XOR(img1, img2):
  xor_result = cv2.bitwise_xor(img1, img2)
  return xor_result

# Image operations


def Union(img_1, img_2):
  img_max = np.maximum(img_1, img_2)
  return img_max


def Intersection(img_1, img_2):
  img_min = np.minimum(img_1, img_2)
  return img_min

def Subtraction(img_1, img_2):
  diff_img = img_1 - img_2
  sum_diff = np.sum(diff_img)
  return sum_diff

 # Returns the image as a numpy array
def read_image(img, problem):
  img_path = problem.figures[img].visualFilename
  grayscale_cutoff = 127
  img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
  _, img = cv2.threshold(img, grayscale_cutoff, 1, cv2.THRESH_BINARY_INV)
  np_img = np.asarray(img, dtype=float)
  return np_img


def show_image(name, img):
  # 255 is white, 0 is white in paper
  # 0 is black, 1 is black in paper
  cv2.imshow(name, (1 - img) * 255)
  cv2.waitKey()

def translate_image(img, h, w):
  height, width = img.shape[:2]
  T = np.float32([[1, 0, w], [0, 1, h]])
  img_translation = cv2.warpAffine(img, T, (width, height))
  return img_translation

 # Returns best translated image, fills with white
def translate_to_find_best(img_to_translate, img_to_compare):
  best_similarity = 0.0
  window_to_explore = 5
  img_to_return = None

  for h in range(-(window_to_explore-1), window_to_explore):
    for w in range(-(window_to_explore-1), window_to_explore):
      img_translation = translate_image(img_to_translate, h, w)
      similarity = calculate_similarity(img_translation, img_to_compare)
      if similarity > best_similarity:
        best_similarity = similarity
        img_to_return = img_translation

  return img_to_return, best_similarity


def calculate_similarity(img_1, img_2):
  intersection_img = Intersection(img_1, img_2)
  union_img = Union(img_1, img_2,)
  similarity = np.sum(intersection_img) / np.sum(union_img)
  return similarity
