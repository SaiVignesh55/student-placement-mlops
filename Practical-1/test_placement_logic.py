import unittest

from placement_logic import predict_placement


class TestPlacementLogic(unittest.TestCase):

    def test_placed_student(self):
        result = predict_placement(8.2, 2, 3)
        self.assertEqual(result, "PLACED")

    def test_not_placed_due_to_low_cgpa(self):
        result = predict_placement(6.2, 2, 3)
        self.assertEqual(result, "NOT PLACED")

    def test_not_placed_due_to_no_internship(self):
        result = predict_placement(8.2, 0, 3)
        self.assertEqual(result, "NOT PLACED")

    def test_not_placed_due_to_few_projects(self):
        result = predict_placement(8.2, 2, 1)
        self.assertEqual(result, "NOT PLACED")


if __name__ == "__main__":
    unittest.main()
