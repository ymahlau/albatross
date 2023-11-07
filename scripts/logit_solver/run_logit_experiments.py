import pickle
from pathlib import Path

from scripts.logit_solver.solve_logit_parallel import compute_equilibrium_data_parallel, LogitSolverExperimentConfig
from src.equilibria.logit import SbrMode
from src.game.normal_form.random_matrix import NFGType


def create_gt_logit_data():
    save_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'temp.pkl'

    cfg = LogitSolverExperimentConfig(
        num_player=2,
        num_actions=6,
        num_iterations=int(1e6),
        nfg_type=NFGType.ZERO_SUM,
        num_games=10,
        sbr_mode=SbrMode.NAGURNEY,
        temperature_range=(0, 5),
    )

    data = compute_equilibrium_data_parallel(
        cfg=cfg,
        restrict_cpu=False,
        num_procs=4,
    )
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'Done!', flush=True)

def temp():
    save_path = Path(__file__).parent.parent.parent.parent / 'a_data' / 'logit_solver' / 'temp.pkl'
    with open(save_path, 'rb') as f:
        data = pickle.load(f)
    a = 1

if __name__ == '__main__':
    # create_gt_logit_data()
    temp()
