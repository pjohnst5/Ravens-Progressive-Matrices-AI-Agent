# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
from PIL import Image
import numpy as np
import cv2
import sys
from RavensProblem import RavensProblem
from CompositeTransform import CompositeTransform
from ImageHelpers import Identity, Rotate_90, Rotate_180, Rotate_270, Identity_Flip, Rotate_90_Flip, Rotate_180_Flip, Rotate_270_Flip, AND, OR, XOR, Union, Intersection, read_image, show_image, TWO_BY_TWO, THREE_BY_THREE, translate_image, Subtraction, translate_to_find_best, calculate_similarity
from DPR import DPR
from operator import add, sub

class Agent:
  BASE_TRANSFORMS = [Identity, Rotate_90, Rotate_180, Rotate_270, Identity_Flip, Rotate_90_Flip, Rotate_180_Flip, Rotate_270_Flip]
  BINARY_BASE_TRANSFORMS = [AND, OR, XOR]
  # The default constructor for your Agent. Make sure to execute any
  # processing necessary before your Agent starts solving problems here.
  #
  # Do not add any variables to this signature; they will not be used by
  # main().
  def __init__(self):
    self.dpr_right_all = []
    self.dpr_right_vote = []
    self.dpr_right_second_best_vote = []
    self.dpr_right_vote_ratio = []
    self.dpr_wrong_all = []
    self.dpr_wrong_vote = []
    self.dpr_wrong_second_best_vote = []
    self.dpr_wrong_vote_ratio = []
    self.two_by_two_combos = [['A', 'B'], ['A', 'C']]
    self.two_by_two_to_apply = {
      str(['A', 'B']): 'C',
      str(['A', 'C']): 'B'
    }

    self.three_by_three_combos = [['A', 'B'], ['B', 'C'],
                                  ['D', 'E'], ['E', 'F'],
                                  ['G', 'H'],
                                  ['A', 'D'], ['D', 'G'],
                                  ['B', 'E'], ['E', 'H'],
                                  ['C', 'F'],
                                  ['A', 'C'], ['D', 'F'],
                                  ['A', 'G'], ['B', 'H'],
                                  ['A', 'E']]
    self.three_by_three_to_apply = {
        str(['A', 'B']): 'H',
        str(['B', 'C']): 'H',
        str(['D', 'E']): 'H',
        str(['E', 'F']): 'H',
        str(['G', 'H']): 'H',

        str(['A', 'D']): 'F',
        str(['D', 'G']): 'F',
        str(['B', 'E']): 'F',
        str(['E', 'H']): 'F',
        str(['C', 'F']): 'F',

        str(['A', 'C']): 'G',
        str(['D', 'F']): 'G',
        
        str(['A', 'G']): 'C',
        str(['B', 'H']): 'C',
        str(['A', 'E']): 'E',
    }

    self.binary_transform_combos = [['A', 'B', 'C'], ['D', 'E', 'F'], ['A', 'D', 'G'], ['B', 'E', 'H']]
    self.binary_transforms_to_apply = {
      str(['A', 'B', 'C']): ['G', 'H'],
      str(['D', 'E', 'F']): ['G', 'H'],
      
      str(['A', 'D', 'G']): ['C', 'F'],
      str(['B', 'E', 'H']): ['C', 'F']
    }

  # The primary method for solving incoming Raven's Progressive Matrices.
  # For each problem, your Agent's Solve() method will be called. At the
  # conclusion of Solve(), your Agent should return an int representing its
  # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
  # are also the Names of the individual RavensFigures, obtained through
  # RavensFigure.getName(). Return a negative number to skip a problem.
  #
  # Make sure to return your answer *as an integer* at the end of Solve().
  # Returning your answer as a string may cause your program to crash.
  def Solve(self, problem):
    print()
    print(problem.name)

    best_composite_transformation = self.Inspect(problem)
    prediction = self.Predict(problem, best_composite_transformation)
    affine_selection, similarity = self.Compare(problem, prediction, best_composite_transformation)
    affine_diff = abs(similarity - best_composite_transformation.score)

    DPRAgent = DPR()
    dpr_selection, vote, second_best_vote, second_best_selection = DPRAgent.Solve(problem)
    if vote == 0:
      vote_ratio = 1
    else:
      vote_ratio = second_best_vote / vote

    all_black_selection = self.get_most_black(problem)
    
    print("\t similarity={:2f}".format(similarity))
    print("\t score={:2f}".format(best_composite_transformation.score))
    print("\t affine_diff={:2f}".format(affine_diff))
    print("\t\t transform pair=" + str(best_composite_transformation.combo))
    print("\t\t transform=" + str(best_composite_transformation.base_transform))
    print("\t\t affine selection=" + str(affine_selection))
    print("\t vote={:2f}".format(vote))
    print("\t second_best_vote={:2f}".format(second_best_vote))
    print("\t vote_ratio={:2f}".format(vote_ratio))
    print("\t\t dpr selection=" + str(dpr_selection))
    print("\t\t dpr 2nd selection=" + str(second_best_selection))

    if len(best_composite_transformation.combo) == 3 and best_composite_transformation.base_transform == XOR and similarity > 0.90 and best_composite_transformation.score > 0.93:
      selection = affine_selection
      selection_agent = "affine"
    elif similarity > 0.94 and best_composite_transformation.score > 0.94:
      selection = affine_selection
      selection_agent = "affine"
    #elif vote >= something and diff >= something
    #elif vote >= something and second_best_vote < something and vote_ratio?
    #elif vote >= 3.95 and second_best_vote < 9.1
    #elif second_best_vote < something
    #elif diff >= something
    #elif vote >= 3.95 and second_best_vote < 9.1:
    elif vote_ratio < 0.92:
      selection = dpr_selection
      selection_agent = "dpr"
      if hasattr(problem, "answer"):
        if selection == problem.answer:
          self.dpr_right_all.append(
              {"vote": vote, "second_best_vote": second_best_vote, "vote_ratio": vote_ratio})
          self.dpr_right_vote.append({"vote": vote})
          self.dpr_right_second_best_vote.append({"second_best_vote": second_best_vote})
          self.dpr_right_vote_ratio.append({"vote_ratio": vote_ratio})
        else:
          self.dpr_wrong_all.append(
              {"vote": vote, "second_best_vote": second_best_vote, "vote_ratio": vote_ratio})
          self.dpr_wrong_vote.append({"vote": vote})
          self.dpr_wrong_second_best_vote.append({"second_best_vote": second_best_vote})
          self.dpr_wrong_vote_ratio.append({"vote_ratio": vote_ratio})
    else:
      selection = 5
      selection_agent = "guess"
        

    print("\t " + selection_agent)
    if hasattr(problem, "answer"):
      print("\t Correct answer: " + str(problem.answer))
      if problem.answer == selection:
        print("\t Correct!")
      else:
        print("\t Wrong :/")
    return selection

  # Returns best composite transformation
  def Inspect(self, problem: RavensProblem):
    best_composite_transformation = self.inspection(problem)

    return best_composite_transformation

  def Predict(self, problem: RavensProblem, transform: CompositeTransform):
    if problem.problemType == TWO_BY_TWO:
      to_apply = self.two_by_two_to_apply
    else:
      to_apply = self.three_by_three_to_apply
    
    if len(transform.combo) == 3:
      img_names = self.binary_transforms_to_apply[str(transform.combo)]
      img1_name = img_names[0]
      img2_name = img_names[1]
      img1_to_operate_on = read_image(img1_name, problem)
      img2_to_operate_on = read_image(img2_name, problem)
      prediction_without_translate = transform.execute(img1_to_operate_on, img2_to_operate_on)
      return prediction_without_translate


    img_to_operate_on_name = to_apply[str(transform.combo)]
    img_to_operate_on = read_image(img_to_operate_on_name, problem)

    prediction_without_translate = transform.execute(img_to_operate_on)

    return prediction_without_translate

  def Compare(self, problem: RavensProblem, prediction, transform):
    best_similarity = 0.0
    answer = -1

    if problem.problemType == TWO_BY_TWO:
      ints = range(1, 7)
    else:
      ints = range(1,9)

    for i in ints:
      answer_choice_img = read_image(str(i), problem)

      # Create predictions tailored to each answer choice if it's a binary
      if len(transform.combo) == 3:
        img_names = self.binary_transforms_to_apply[str(transform.combo)]
        img1_name = img_names[0]
        img2_name = img_names[1]
        img1 = read_image(img1_name, problem)
        img2 = read_image(img2_name, problem)
        window_to_explore = 2
        best_binary_similarity_single_pair = 0

        for h1 in range(-(window_to_explore-1), window_to_explore):
          for w1 in range(-(window_to_explore-1), window_to_explore):
            img1_translation = translate_image(img1, h1, w1)
            for h2 in range(-(window_to_explore-1), window_to_explore):
              for w2 in range(-(window_to_explore-1), window_to_explore):
                img2_translation = translate_image(img2, h2, w2)
                binary_result = transform.execute(img1_translation, img2_translation)
                iterative_similarity = calculate_similarity(binary_result, answer_choice_img)
                if iterative_similarity > best_binary_similarity_single_pair:
                  best_binary_similarity_single_pair = iterative_similarity
        similarity = best_binary_similarity_single_pair
      else:
        translated_prediction, similarity = translate_to_find_best(prediction, answer_choice_img)

      if similarity > best_similarity:
        best_similarity = similarity
        answer = i

    return answer, best_similarity
    
  # Inspects the problem and returns the best composite transformation
  def inspection(self, problem: RavensProblem):
    composite_transformations = []
    binary_transformations = []
    if problem.problemType == TWO_BY_TWO:
      combos = self.two_by_two_combos
    else:
      combos = self.three_by_three_combos

    for combo in combos:
      src_val = combo[0]
      dst_val = combo[1]
      src_img = read_image(src_val, problem)
      dst_img = read_image(dst_val, problem)

      composite_transformations.extend(self.create_all_composite_transformations(src_img, dst_img, combo))

    best_transform = composite_transformations[0]
    for transform in composite_transformations:
      if transform.score > best_transform.score:
        best_transform = transform

    # binary transforms
    if problem.problemType == THREE_BY_THREE:
      for combo in self.binary_transform_combos:
        img1 = read_image(combo[0], problem)
        img2 = read_image(combo[1], problem)
        target_img = read_image(combo[2], problem)
        binary_transform_results = self.create_all_binary_transforms(
            img1, img2, target_img, combo)
        binary_transformations.extend(binary_transform_results)

      best_binary_transform = binary_transformations[0]
      for binary_transform in binary_transformations:
        if binary_transform.score > best_binary_transform.score:
          best_binary_transform = binary_transform

      # Favor binary transforms if it's high
      if best_binary_transform.score > 0.93:
        return best_binary_transform

    return best_transform

  def create_all_composite_transformations(self, src_img, dst_img, combo):
    transformations = []

    for base_transform in Agent.BASE_TRANSFORMS:
      base_transformed_img = base_transform(src_img.copy())
      translated_img, similarity = translate_to_find_best(base_transformed_img, dst_img)
      #similarity = calculate_similarity(base_transformed_img, dst_img)
      transformations.append(CompositeTransform(similarity, base_transform, combo))
    return transformations

  def create_all_binary_transforms(self, img1, img2, target_image, combo):
    binary_results = []
    or_similarity = -1
    for binary_transform in Agent.BINARY_BASE_TRANSFORMS:
      window_to_explore = 2
      best_binary_similarity_single_pair = 0

      for h1 in range(-(window_to_explore-1), window_to_explore):
        for w1 in range(-(window_to_explore-1), window_to_explore):
          img1_translation = translate_image(img1, h1, w1)
          for h2 in range(-(window_to_explore-1), window_to_explore):
            for w2 in range(-(window_to_explore-1), window_to_explore):
              img2_translation = translate_image(img2, h2, w2)
              binary_result = binary_transform(img1_translation, img2_translation)
              similarity = calculate_similarity(binary_result, target_image)
              if similarity > best_binary_similarity_single_pair:
                best_binary_similarity_single_pair = similarity
      # Basicall, if Or and Xor return same similarity, favor OR
      if binary_transform == OR:
        or_similarity = best_binary_similarity_single_pair
      if binary_transform == XOR:
        if best_binary_similarity_single_pair == or_similarity:
          best_binary_similarity_single_pair = 0
      binary_results.append(CompositeTransform(best_binary_similarity_single_pair, binary_transform, combo))
    return binary_results

  def get_most_black(self, problem: RavensProblem):
    highest = 0
    answer_choice = -1

    if problem.problemType == TWO_BY_TWO:
      ints = range(1, 7)
    else:
      ints = range(1, 9)

    for i in ints:
      img = read_image(str(i), problem)
      sum = np.sum(img)
      if sum >= highest:
        highest = sum
        answer_choice = i
    return answer_choice

  def print_results(self):
    print("Correct dpr ones ordered by vote")
    print (sorted(self.dpr_right_all, key=lambda i: i['vote'], reverse=True))

    print("\nCorrect dpr ones ordered by second_best_vote")
    print(sorted(self.dpr_right_all, key=lambda i: i['second_best_vote'], reverse=True))

    print("\nCorrect dpr ones ordered by vote_ratio")
    print(sorted(self.dpr_right_all, key=lambda i: i['vote_ratio'], reverse=True))

    print("\n Correct dpr by ratio")
    print(sorted(self.dpr_right_vote_ratio, key=lambda i: i['vote_ratio'], reverse=True))


    print("\nINcorrect dpr ones ordered by vote")
    print (sorted(self.dpr_wrong_all, key=lambda i: i['vote'], reverse=True))

    print("\nINcorrect dpr ones ordered by second_best_vote")
    print(sorted(self.dpr_wrong_all, key=lambda i: i['second_best_vote'], reverse=True))

    print("\nINcorrect dpr ones ordered by vote_ratio")
    print(sorted(self.dpr_wrong_all, key=lambda i: i['vote_ratio'], reverse=True))

    print("\n INcorrect dpr by ratio")
    print(sorted(self.dpr_wrong_vote_ratio, key=lambda i: i['vote_ratio'], reverse=True))


