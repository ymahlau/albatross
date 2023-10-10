import itertools
import random
import time
import unittest

import numpy as np

from src.equilibria.nash import calculate_nash_equilibrium


class TestNashSolverCPP(unittest.TestCase):
    def test_simple(self):
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, -1], [-1, 1], [-1, 1], [1, -1]], dtype=np.float32)
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values)
        print(values)
        print(action_probs)
        self.assertAlmostEqual(0, values[0], places=3)
        self.assertAlmostEqual(0, values[1], places=3)
        for player in range(2):
            for action in range(2):
                self.assertAlmostEqual(0.5, action_probs[player][action], places=4)

    def test_simple_general_interface(self):
        available_actions = [[3, 2], [5, 1]]
        joint_action_list = [(2, 1), (3, 5), (3, 1), (2, 5)]
        joint_action_values = np.asarray([[1, -1], [1, -1], [-1, 1], [-1, 1]], dtype=float)
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values,
                                                          use_cpp=True)
        print(values)
        print(action_probs)
        self.assertAlmostEqual(0, values[0], places=3)
        self.assertAlmostEqual(0, values[1], places=3)
        for player in range(2):
            for action in range(2):
                self.assertAlmostEqual(0.5, action_probs[player][action], places=4)

    def test_asymmetric(self):
        """
          EE  1  P1:  (1)  0.800000  0.200000  0.000000  EP=  3.0  P2:  (1)  0.666667  0.333333  EP=                 2.8
          EE  2  P1:  (2)  0.000000  0.333333  0.666667  EP=  4.0  P2:  (2)  0.333333  0.666667  EP=  2.6666666666666665
          EE  3  P1:  (3)  1.000000  0.000000  0.000000  EP=  3.0  P2:  (3)  1.000000  0.000000  EP=                 3.0
        """
        available_actions = [[0, 1, 2], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        joint_action_values = np.asarray([[3, 3], [3, 2], [2, 2], [5, 6], [0, 3], [6, 1]], dtype=float)
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values,
                                                          use_cpp=True)
        print(values)
        print(action_probs)

    def test_degenerate(self):
        """
          EE  1  P1:  (1)  1.000000  0.000000  0.000000  EP=  3.0  P2:  (1)  1.000000  0.000000  EP=                 3.0
          EE  2  P1:  (1)  1.000000  0.000000  0.000000  EP=  3.0  P2:  (2)  0.666667  0.333333  EP=                 3.0
          EE  3  P1:  (2)  0.000000  0.333333  0.666667  EP=  4.0  P2:  (3)  0.333333  0.666667  EP=  2.6666666666666665
        """
        available_actions = [[0, 1, 2], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        joint_action_values = np.asarray([[3, 3], [3, 3], [2, 2], [5, 6], [0, 3], [6, 1]], dtype=float)
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values,
                                                          use_cpp=False)
        print(values)
        print(action_probs)

    def test_floating_point(self):
        """
          Divide values by 10 !
          EE  1  P1:  (1)  0.800000  0.200000  0.000000  EP=  3.0  P2:  (1)  0.666667  0.333333  EP=                 2.8
          EE  2  P1:  (2)  0.000000  0.333333  0.666667  EP=  4.0  P2:  (2)  0.333333  0.666667  EP=  2.6666666666666665
          EE  3  P1:  (3)  1.000000  0.000000  0.000000  EP=  3.0  P2:  (3)  1.000000  0.000000  EP=                 3.0
        """
        available_actions = [[0, 1, 2], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        joint_action_values = np.asarray([[3, 3], [3, 2], [2, 2], [5, 6], [0, 3], [6, 1]], dtype=float) / 10.0
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values,
                                                          use_cpp=False)
        print(values)
        print(action_probs)

    def test_temp(self):
        """
          EE  1  P1:  (1)  1.000000  0.000000  EP=  1.0  P2:  (1)  1.000000  0.000000  EP=  1.0
          EE  2  P1:  (2)  0.000000  1.000000  EP=  5.0  P2:  (2)  0.000000  1.000000  EP=  5.0
        """
        available_actions = [[0, 1], [0, 1]]
        joint_action_list = [(0, 0), (0, 1), (1, 0), (1, 1)]
        joint_action_values = np.asarray([[1, 1], [1, 1], [1, 1], [5, 5]], dtype=np.float32)
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values)
        print(values)
        print(action_probs)

    def test_big(self):
        dim = 8
        available_actions = [list(range(dim)) for _ in range(2)]
        joint_action_list = list(itertools.product(*available_actions))
        joint_action_values = np.random.rand(len(joint_action_list), 2)
        start = time.time()
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values)
        print(time.time() - start)
        print(values)
        print(action_probs)

    def test_3_player(self):
        available_actions = [[0, 1], [0, 1], [0, 1]]
        joint_actions = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        joint_action_values = np.asarray([
            [7, 7, 7],
            [7, 7, 6],
            [1, 1, 2.3],
            [1, 2, 0],
            [6, 0, 0],
            [8, 5, 3],
            [6, 6.5, 1],
            [6, 5.5, 5],
        ], dtype=np.float32)
        start = time.time()
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_actions, joint_action_values)
        print(time.time() - start)
        print(values)
        print(action_probs)

    def test_book_example(self):
        # https://link.springer.com/content/pdf/10.1007/978-93-86279-17-0_5.pdf
        available_actions = [[0, 1], [0, 1], [0, 1]]
        joint_actions = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        joint_action_values = np.asarray([
            [0, 0, 0],
            [0, 0, 2],
            [0, 2, 0],
            [1, 0, 0],
            [2, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ], dtype=np.float32)
        start = time.time()
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_actions, joint_action_values)
        print(time.time() - start)
        print(values)
        print(action_probs)

    def test_n_player(self):
        num_player = 4
        num_actions = 3
        max_action_idx = 10
        num_iter = 100
        available_actions = [random.choices(list(range(max_action_idx)), k=num_actions) for _ in range(num_player)]
        joint_action_list = list(itertools.product(*available_actions))
        time_sum = 0
        values, action_probs = None, None
        for _ in range(num_iter):
            joint_action_values = np.random.rand(len(joint_action_list), num_player)
            start = time.time()
            values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_values)
            time_sum += time.time() - start
        print(time_sum / num_iter)
        print(values)
        print(action_probs)

    def test_cooperative_multiplayer(self):
        # https://link.springer.com/content/pdf/10.1007/978-93-86279-17-0_5.pdf
        available_actions = [[0, 1], [0, 1], [0, 1]]
        joint_actions = [
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        ]
        joint_action_values = np.asarray([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 2, 2],
            [1, 0, 0],
            [2, 0, 2],
            [2, 2, 0],
            [3, 3, 3],
        ], dtype=np.float32)
        start = time.time()
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_actions, joint_action_values)
        print(time.time() - start)
        print(values)
        print(action_probs)
        
    def test_special_error(self):
        available_actions = [[7, 7, 5], [8, 9, 1], [1, 1, 1], [9, 9, 6]]
        joint_action_list = [(7, 8, 1, 9), (7, 8, 1, 9), (7, 8, 1, 6), (7, 8, 1, 9), (7, 8, 1, 9), (7, 8, 1, 6),
                             (7, 8, 1, 9), (7, 8, 1, 9), (7, 8, 1, 6), (7, 9, 1, 9), (7, 9, 1, 9), (7, 9, 1, 6),
                             (7, 9, 1, 9), (7, 9, 1, 9), (7, 9, 1, 6), (7, 9, 1, 9), (7, 9, 1, 9), (7, 9, 1, 6),
                             (7, 1, 1, 9), (7, 1, 1, 9), (7, 1, 1, 6), (7, 1, 1, 9), (7, 1, 1, 9), (7, 1, 1, 6),
                             (7, 1, 1, 9), (7, 1, 1, 9), (7, 1, 1, 6), (7, 8, 1, 9), (7, 8, 1, 9), (7, 8, 1, 6),
                             (7, 8, 1, 9), (7, 8, 1, 9), (7, 8, 1, 6), (7, 8, 1, 9), (7, 8, 1, 9), (7, 8, 1, 6),
                             (7, 9, 1, 9), (7, 9, 1, 9), (7, 9, 1, 6), (7, 9, 1, 9), (7, 9, 1, 9), (7, 9, 1, 6),
                             (7, 9, 1, 9), (7, 9, 1, 9), (7, 9, 1, 6), (7, 1, 1, 9), (7, 1, 1, 9), (7, 1, 1, 6),
                             (7, 1, 1, 9), (7, 1, 1, 9), (7, 1, 1, 6), (7, 1, 1, 9), (7, 1, 1, 9), (7, 1, 1, 6),
                             (5, 8, 1, 9), (5, 8, 1, 9), (5, 8, 1, 6), (5, 8, 1, 9), (5, 8, 1, 9), (5, 8, 1, 6),
                             (5, 8, 1, 9), (5, 8, 1, 9), (5, 8, 1, 6), (5, 9, 1, 9), (5, 9, 1, 9), (5, 9, 1, 6),
                             (5, 9, 1, 9), (5, 9, 1, 9), (5, 9, 1, 6), (5, 9, 1, 9), (5, 9, 1, 9), (5, 9, 1, 6),
                             (5, 1, 1, 9), (5, 1, 1, 9), (5, 1, 1, 6), (5, 1, 1, 9), (5, 1, 1, 9), (5, 1, 1, 6),
                             (5, 1, 1, 9), (5, 1, 1, 9), (5, 1, 1, 6)]
        joint_action_value_arr = np.asarray(
            [
                [0.39553605, 0.26308985, 0.96972347, 0.63437033],
                [0.87202009, 0.72306532, 0.89998366, 0.52292568],
                [0.14561845, 0.2509899, 0.27477058, 0.36729797],
                [0.35159174, 0.83109897, 0.56497305, 0.38785596],
                [0.88704532, 0.36354572, 0.90965297, 0.52963601],
                [0.9745789, 0.02561551, 0.72853037, 0.3087391],
                [0.55582947, 0.23276069, 0.09974263, 0.70557482],
                [0.63654461, 0.27405585, 0.44613936, 0.55117804],
                [0.96074583, 0.03654894, 0.23990841, 0.05741734],
                [0.40828456, 0.95562312, 0.48926004, 0.43996475],
                [0.29546605, 0.66040321, 0.85368922, 0.33496075],
                [0.03625471, 0.7198145, 0.35572331, 0.0969355],
                [0.94193491, 0.22180286, 0.42711048, 0.93725159],
                [0.28042501, 0.09829993, 0.01873863, 0.457887],
                [0.72134492, 0.88889129, 0.52625772, 0.3714609],
                [0.02718456, 0.79156456, 0.52148626, 0.97462595],
                [0.89252366, 0.80085313, 0.80580095, 0.64516301],
                [0.42214522, 0.47262787, 0.07000166, 0.49194568],
                [0.49848032, 0.59312213, 0.68278203, 0.46880793],
                [0.14645703, 0.19119043, 0.25687641, 0.32374439],
                [0.29235378, 0.61391599, 0.94703789, 0.59188874],
                [0.43234474, 0.74979447, 0.82776207, 0.94420898],
                [0.31428168, 0.02319889, 0.92243655, 0.07669855],
                [0.82561533, 0.56837543, 0.12768766, 0.25402158],
                [0.2024879, 0.94057967, 0.96544233, 0.20186098],
                [0.60479134, 0.2770253, 0.45335971, 0.43818029],
                [0.92815488, 0.9705755, 0.34486758, 0.26528805],
                [0.60636845, 0.22028752, 0.79517342, 0.25228118],
                [0.24093641, 0.91493328, 0.16027618, 0.8710931],
                [0.34801933, 0.58015186, 0.7265555, 0.0626591],
                [0.60064662, 0.79877934, 0.47650621, 0.26454901],
                [0.97671445, 0.67313455, 0.62069142, 0.43455165],
                [0.87648896, 0.11461223, 0.39654576, 0.73227092],
                [0.14395879, 0.84851757, 0.31077306, 0.56419599],
                [0.51390873, 0.87660479, 0.07939991, 0.58885081],
                [0.7527991, 0.84476281, 0.58018992, 0.29701674],
                [0.63113023, 0.29230595, 0.14376838, 0.20531412],
                [0.6478823, 0.1003738, 0.76018153, 0.33330044],
                [0.35921806, 0.85012295, 0.1230535, 0.21648103],
                [0.13400189, 0.04229872, 0.77086528, 0.95887791],
                [0.41857592, 0.07729187, 0.91205631, 0.60971121],
                [0.9079916, 0.32610635, 0.52858158, 0.42995458],
                [0.38640664, 0.40144045, 0.55043318, 0.3704066],
                [0.94059481, 0.85276656, 0.58142147, 0.48881781],
                [0.45462811, 0.94018489, 0.92418261, 0.25336619],
                [0.094351, 0.47898928, 0.19867966, 0.4855243],
                [0.04115445, 0.75538038, 0.97571898, 0.35091393],
                [0.01533523, 0.44550683, 0.49052263, 0.51153754],
                [0.47177658, 0.2594259, 0.95025305, 0.11434969],
                [0.97629583, 0.27402873, 0.190349, 0.93417775],
                [0.65765279, 0.88686867, 0.83497384, 0.81457681],
                [0.73076722, 0.9978446, 0.1334464, 0.90932611],
                [0.4459188, 0.52126516, 0.30893413, 0.10086215],
                [0.26378705, 0.06962337, 0.24151619, 0.97857977],
                [0.39130483, 0.78086002, 0.36820083, 0.63197082],
                [0.80748203, 0.03363463, 0.77104611, 0.9461493],
                [0.01405928, 0.90323628, 0.70855935, 0.88582386],
                [0.76637876, 0.14222474, 0.04982145, 0.51236456],
                [0.14667909, 0.32915669, 0.39395715, 0.35528617],
                [0.41832428, 0.11033062, 0.31816797, 0.84483859],
                [0.02940566, 0.55810543, 0.29185097, 0.21925097],
                [0.13855261, 0.89118101, 0.55413587, 0.40260247],
                [0.66542985, 0.78808352, 0.21272259, 0.37618521],
                [0.13055318, 0.64391405, 0.07903432, 0.18626174],
                [0.59164525, 0.12204693, 0.44366442, 0.26309885],
                [0.52021622, 0.26331586, 0.22764727, 0.35130552],
                [0.54586325, 0.15880609, 0.3025836, 0.79208157],
                [0.09729315, 0.94633175, 0.62413365, 0.57627927],
                [0.73547524, 0.59575448, 0.30054649, 0.29680542],
                [0.74762231, 0.63357017, 0.49819526, 0.52183986],
                [0.59472518, 0.82221344, 0.85633981, 0.64058988],
                [0.37618606, 0.78468238, 0.75975534, 0.9625028],
                [0.61669899, 0.5688187, 0.38029372, 0.60335219],
                [0.45009175, 0.69182649, 0.57427001, 0.40647297],
                [0.01472821, 0.58016655, 0.8294275, 0.16442718],
                [0.09788249, 0.0252093, 0.53242227, 0.50588806],
                [0.76250857, 0.34791807, 0.72933834, 0.71217234],
                [0.6277213, 0.49668262, 0.3449152, 0.71553869],
                [0.35677325, 0.67039844, 0.96629777, 0.47731114],
                [0.98973166, 0.35805004, 0.84892369, 0.42605258],
                [0.20247228, 0.20981338, 0.63591637, 0.67194483]
            ],
            dtype=np.float32
        )
        start = time.time()
        values, action_probs = calculate_nash_equilibrium(available_actions, joint_action_list, joint_action_value_arr)
        print(time.time() - start)
        print(values)
        print(action_probs)

