import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import numpy as np
from modules.benchmarking.metrics_and_benchmarking import *

class TestAccumulateTopN(unittest.TestCase):
    """
    Class to test the the accumulate_top_n function
    """

    def test_standard_array(self):
        """Test functionality on a standard array"""
        array = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])
        n = 3
        expected = [3, (3+1)/2, (3+1+4)/3, (3+1+4)/3, (3+4+5)/3, (4+5+9)/3, 
                    (4+5+9)/3, (6+5+9)/3, (6+5+9)/3, (6+5+9)/3, (6+5+9)/3]
        self.assertAlmostEqual(accumulate_top_n(array, n), expected)

    def test_empty_array(self):
        """Test functionality on an empty array"""
        self.assertEqual(accumulate_top_n(np.array([]), 3), [])

    def test_fewer_elements_than_n(self):
        """Test functionality when length of array smaller than n"""
        array = np.array([1, 2])
        n = 5
        expected = [1.0, 1.5]
        self.assertAlmostEqual(accumulate_top_n(array, n), expected)

    def test_array_length_equals_n(self):
        """Test functionality when length of array equals n"""
        array = np.array([1, 2, 3])
        n = 3
        expected = [1.0, 1.5, 2.0]
        self.assertAlmostEqual(accumulate_top_n(array, n), expected)

    def test_large_n(self):
        """Test functionality when n is larger than length of array"""
        array = np.array([1, 2, 3])
        n = 10
        expected = [1.0, 1.5, 2.0]
        self.assertAlmostEqual(accumulate_top_n(array, n), expected)

    def test_all_elements_same(self):
        """Test functionality when all elements are the same"""
        array = np.array([2, 2, 2])
        n = 2
        expected = [2.0, 2.0, 2.0]
        self.assertAlmostEqual(accumulate_top_n(array, n), expected)

    def test_negative_numbers_and_zero(self):
        """Test functionality when there are negative numbers and zero"""
        array = np.array([-1, 0, -2, -3])
        n = 2
        expected = [-1, (-1+0)/2, (-1+0)/2, (-1+0)/2]
        self.assertAlmostEqual(accumulate_top_n(array, n), expected)

    def test_floating_point_numbers(self):
        """Test functionality on floating point numbers"""
        array = np.array([1.5, 2.3, 3.7])
        n = 2
        expected = [1.5, (1.5+2.3)/2, (2.3+3.7)/2]
        self.assertAlmostEqual(accumulate_top_n(array, n), expected)

class TestTopAUC(unittest.TestCase):
    """Class to test the top_auc function"""

    def test_basic_functionality(self):
        """Test basic functionlaity of top_auc function"""

        buffer = {'mol1': (0.9, 1), 'mol2': (0.8, 2), 
                  'mol3': (0.9, 3), 'mol4': (0.7, 4)}
        scores = [item[0] for item in buffer.values()]
        top_n_values = accumulate_top_n(scores, 2)

        sum = 0
        prev = 0
        for value in top_n_values:
            sum += (prev+value)/2
            prev = value

        expected = sum/len(top_n_values)
        result = top_auc(buffer, top_n=2, max_oracle_calls=4)
        self.assertAlmostEqual(result, expected)

    def test_basic_functionality_large(self):
        """Test functionality on a large set of molecules"""

        buffer = {'mol1': (1, 1), 'mol2': (1, 2), 
                  'mol3': (1, 3), 'mol4': (1, 4), 
                  'mol5': (0.30, 5), 'mol6': (0.39, 6), 
                  'mol7': (0.61, 7)}
        scores = [item[0] for item in buffer.values()]
        top_n_values = accumulate_top_n(scores, 3)

        sum = 0
        prev = 0
        max_oracle_calls = 7

        for value in top_n_values[:max_oracle_calls]:
            sum += (prev+value)/2
            prev = value

        expected = sum/max_oracle_calls 

        result = top_auc(buffer, top_n=3, max_oracle_calls=max_oracle_calls)
        self.assertAlmostEqual(result, expected)

    def test_larger_max_oracle_call(self):
        """Test when max_oracle_calls is larger than the length of the array"""

        buffer = {'mol1': (0.3, 1), 'mol2': (0.9, 2), 'mol3': (0.2, 3), 
                  'mol4': (0.7, 4), 'mol5': (0.3, 5)}
        scores = [item[0] for item in buffer.values()]
        top_n_values = accumulate_top_n(scores, 3)

        sum = 0
        prev = 0
        max_oracle_calls = 10

        for value in top_n_values:
            sum += (prev+value)/2
            prev = value

        sum += ((max_oracle_calls - len(top_n_values))*prev)

        expected = sum/max_oracle_calls 
        result = top_auc(buffer, top_n=3, max_oracle_calls=max_oracle_calls)
        self.assertAlmostEqual(result, expected)


class TestMeanOfTopN(unittest.TestCase):
    def test_mean_of_top_n_basic(self):
        array = [10, 20, 30, 40, 50]
        n = 3
        expected_mean = (30 + 40 + 50) / 3
        result = mean_of_top_n(array, n)
        self.assertEqual(result, expected_mean, "The mean of the top n \
                         elements should be calculated correctly.")

    def test_mean_of_top_n_all_elements(self):
        array = [15, 25, 35, 45]
        n = 4
        expected_mean = (15 + 25 + 35 + 45) / 4
        result = mean_of_top_n(array, n)
        self.assertEqual(result, expected_mean, "The mean should be equal to \
                         the mean of all elements when n equals array length.")

    def test_mean_of_top_n_one_element(self):
        array = [1, 2, 3, 4, 100]
        n = 1
        expected_mean = 100
        result = mean_of_top_n(array, n)
        self.assertEqual(result, expected_mean, "The mean should be equal to \
                         the highest element when n is 1.")

    def test_mean_of_top_n_with_n_greater_than_length(self):
        array = [10, 20, 30]
        n = 5
        with self.assertRaises(ValueError):
            mean_of_top_n(array, n)

    def test_mean_of_top_n_with_negative_numbers(self):
        array = [-10, -20, -30, -40]
        n = 2
        expected_mean = (-10 + -20) / 2
        result = mean_of_top_n(array, n)
        self.assertEqual(result, expected_mean, "The mean should correctly \
                         handle negative numbers.")

    def test_mean_of_top_n_floats(self):
        array = [1.5, 2.3, 3.7, 4.6, 5.9]
        n = 3
        expected_mean = (3.7 + 4.6 + 5.9) / 3
        result = mean_of_top_n(array, n)
        self.assertAlmostEqual(result, expected_mean, "The mean of floating \
                               point numbers should be calculated correctly.")


if __name__ == '__main__':
    unittest.main()
