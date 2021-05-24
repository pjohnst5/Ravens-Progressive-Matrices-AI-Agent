from PIL import Image
import numpy as np
import cv2
import sys
from RavensProblem import RavensProblem
from ImageHelpers import Intersection, read_image, TWO_BY_TWO, THREE_BY_THREE

training_pair_to_test_square_2X2 = {
    str(['A', 'B']): 'C',
    str(['A', 'C']): 'B'
}

training_pair_to_test_square_3X3 = {
    str(['A', 'B']): 'H',
    str(['B', 'C']): 'H',
    str(['D', 'E']): 'H',
    str(['E', 'F']): 'H',
    str(['G', 'H']): 'H',
    str(['A', 'C']): 'H',
    str(['D', 'F']): 'H',

    str(['A', 'D']): 'F',
    str(['D', 'G']): 'F',
    str(['B', 'E']): 'F',
    str(['E', 'H']): 'F',
    str(['C', 'F']): 'F',
    str(['A', 'G']): 'F',
    str(['B', 'H']): 'F',

    str(['A', 'E']): 'E',
    str(['B', 'F']): 'E',
    str(['F', 'G']): 'E',
    str(['D', 'H']): 'E',
    str(['C', 'D']): 'E',

    str(['G', 'E']): 'B',
    str(['F', 'A']): 'B',

    str(['C', 'E']): 'D',
    str(['H', 'A']): 'D'

}

class DPR:
    def __init__(self):
        pass

    def Solve(self, problem):
        dpr_votes = self.Get_Votes(problem, self.calculate_dpr, self.dpr_vote)
        ipr_votes = self.Get_Votes(problem, self.calculate_ipr, self.ipr_vote)
        all_votes = self.merge_votes([dpr_votes, ipr_votes])
        best_answer, vote, second_best_vote, second_answer = self.get_best_vote(all_votes)
        return best_answer, vote, second_best_vote, second_answer

    def Get_Votes(self, problem, ratio_method, vote_method):
        votes = {}
        test_answer_map = self.get_test_answers(problem, ratio_method)

        if problem.problemType == TWO_BY_TWO:
            training_pair_to_test_square = training_pair_to_test_square_2X2
        else:
            training_pair_to_test_square = training_pair_to_test_square_3X3

        for training_pair, test_image_name in training_pair_to_test_square.items():
            img_1_name = training_pair[2]
            img_2_name = training_pair[7]
            img_1 = read_image(img_1_name, problem)
            img_2 = read_image(img_2_name, problem)
            training_pair_ratio = ratio_method(img_1, img_2)

            test_answer_ints = test_answer_map[test_image_name]
            for i, test_answer_ratio in test_answer_ints.items():
                vote, difference = vote_method(training_pair_ratio, test_answer_ratio)
                if i not in votes.keys():
                    votes[i] = 0
                votes[i] += vote
        
        return votes

    def get_test_answers(self, problem, ratio_method):
        test_answer_map = {}

        if problem.problemType == TWO_BY_TWO:
            test_squares = ['C', 'B']
            answer_range = range(1,7)
        else:
            test_squares = ['H', 'E', 'F', 'B', 'D']
            answer_range = range(1,9)
        
        for test_square in test_squares:
            number_map = {}
            for i in answer_range:
                test_square_img = read_image(test_square, problem)
                answer_img = read_image(str(i), problem)
                test_answer_ratio = ratio_method(test_square_img, answer_img)
                number_map[i] = test_answer_ratio
            
            test_answer_map[test_square] = number_map
        
        return test_answer_map

    def calculate_dpr(self, img1, img2):
        ratio_1 = np.sum(img1) / img1.size
        ratio_2 = np.sum(img2) / img2.size
        return ratio_1 - ratio_2

    def dpr_vote(self, dpr_1, dpr_2):
        dpr_voting_closeness_threshold = 0.0005
        abs_difference = abs(dpr_1 - dpr_2)
        vote = 0
        if abs_difference < dpr_voting_closeness_threshold:
            vote = 1 - abs_difference
        return vote, abs_difference

    def calculate_ipr(self, img1, img2):
        np.seterr(all='raise')
        intersection_img = Intersection(img1, img2)
        img1_sum = np.sum(img1)
        img2_sum = np.sum(img2)
        if img1_sum == 0 or img2_sum == 0:
            return -2
        ratio_1 = np.sum(intersection_img) / img1_sum
        ratio_2 = np.sum(intersection_img) / img2_sum
        return ratio_1 - ratio_2

    def ipr_vote(self, ipr_1, ipr_2):
        ipr_voting_closeness_threshold = 0.04
        abs_difference = abs(ipr_1 - ipr_2)
        vote = 0
        if abs_difference < ipr_voting_closeness_threshold:
            vote = 1 - abs_difference
        return vote, abs_difference

    def merge_votes(self, array_of_vote_maps):
        all_votes = {}
        first_map = array_of_vote_maps[0]

        for key, value in first_map.items():
            all_votes[key] = value

        rest_of_maps = array_of_vote_maps[1:]

        for remaining_map in rest_of_maps:
            for key in all_votes.keys():
                all_votes[key] += remaining_map[key]
        return all_votes

    def normalize_votes(self, all_votes):
        min_vote = 100
        max_vote = -100
        for answer, value in all_votes.items():
            if value > max_vote:
                max_vote = value
            if value < min_vote:
                min_vote = value
        
        normalized_votes = {}

        for answer, value in all_votes.items():
            normalized_vote = (value - min_vote) / (max_vote - min_vote)
            normalized_votes[answer] = normalized_vote
        return normalized_votes


    def get_best_vote(self, all_votes):
        best_answer = 1
        best_value = all_votes[best_answer]
        for answer, value in all_votes.items():
            if value > best_value:
                best_value = value
                best_answer = answer
        sorted_votes = sorted(all_votes.items(), key=lambda item: item[1], reverse=True)
        second_best_value = sorted_votes[1]

        return best_answer, best_value, second_best_value[1], second_best_value[0]
