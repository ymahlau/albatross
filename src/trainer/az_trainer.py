import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import multiprocessing as mp

from src.game.initialization import get_game_from_config
from src.game.values import UtilityNorm
from src.misc.utils import set_seed
from src.network.initialization import get_network_from_config
from src.search.config import ResponseInferenceServerEvalConfig
from src.search.initialization import get_search_from_config
from src.trainer.az_collector import run_collector
from src.trainer.az_distributor import run_distributor
from src.trainer.az_evaluator import run_evaluator
from src.trainer.az_inference_server import run_inference_server
from src.trainer.az_logger import run_logger
from src.trainer.az_saver import run_saver
from src.trainer.az_updater import run_updater
from src.trainer.az_worker import run_worker
from src.trainer.config import AlphaZeroTrainerConfig


class AlphaZeroTrainer:
    def __init__(
            self,
            cfg: AlphaZeroTrainerConfig,
    ):
        self.cfg = cfg
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        os.environ["OMP_NUM_THREADS"] = "1"
        if mp.get_start_method() != 'spawn':
            try:
                mp.set_start_method('spawn')  # this is important for using CUDA
            except RuntimeError:
                pass
        # the network cfg needs the game_cfg
        if self.cfg.net_cfg is None:
            if not self.cfg.only_generate_buffer:
                raise ValueError(f"Cannot start trainer without network")
        else:
            if self.cfg.net_cfg.game_cfg is None:
                self.cfg.net_cfg.game_cfg = self.cfg.game_cfg
        self._validate()

    def _validate(self):
        if self.cfg.worker_cfg.anneal_cfgs is not None:
            if self.cfg.single_sbr_temperature:
                if len(self.cfg.worker_cfg.anneal_cfgs) != 1:
                    raise ValueError(f"Invalid number of annealing configs, expected 1")
            else:
                if len(self.cfg.worker_cfg.anneal_cfgs) != self.cfg.game_cfg.num_players:
                    raise ValueError(f"Invalid number of annealing configs")
        if self.cfg.num_worker % self.cfg.num_inference_server != 0:
            raise ValueError(f"Number of workers needs to be divisible by number of inference server")
        if self.cfg.merge_inference_update_gpu and self.cfg.num_inference_server > 1:
            raise ValueError("Can only merge inference and updater gpu with a single inference server")
        if self.cfg.temperature_input and not self.cfg.single_sbr_temperature:
            if not isinstance(self.cfg.worker_cfg.search_cfg.eval_func_cfg, ResponseInferenceServerEvalConfig):
                raise ValueError("Response model can only be trained with response inference server eval function")
            if self.cfg.proxy_net_path is None:
                raise ValueError("Need proxy network to train response network")
        # test if network and search config can be initialized
        if self.cfg.net_cfg is not None:
            _ = get_network_from_config(self.cfg.net_cfg)
        _ = get_search_from_config(self.cfg.worker_cfg.search_cfg)

    def _train(self) -> None:
        if self.cfg.logger_cfg.id is not None:
            set_seed(self.cfg.logger_cfg.id)
        print(f'Started main process with pid {os.getpid()}')
        # initialization
        tmp_dir = Path(os.getcwd()) / 'tmp'
        if not Path.exists(tmp_dir):
            tmp_dir.mkdir(parents=True, exist_ok=True)
        process_list = []
        stop_flag = mp.Value('b', False, lock=False)
        collector_in_queue = mp.Queue(maxsize=self.cfg.data_qsize)
        updater_in_queue = mp.Queue(maxsize=self.cfg.updater_in_qsize)
        update_out_queue = mp.Queue(maxsize=self.cfg.updater_out_qsize)
        eval_in_queue = mp.Queue(maxsize=self.cfg.distributor_out_qsize)
        saver_in_queue = mp.Queue(maxsize=self.cfg.distributor_out_qsize)
        info_queue = mp.Queue(maxsize=self.cfg.info_qsize)
        update_counter = mp.Value('i', 0, lock=False)
        step_counter = mp.Value('i', 0, lock=True)  # the lock is important if multiple workers are running
        error_counter = mp.Value('i', 0, lock=True)
        episode_counter = mp.Value('i', 0, lock=True)
        queue_list = [collector_in_queue, update_out_queue, eval_in_queue, info_queue, saver_in_queue,
                      updater_in_queue]
        dist_out_queue_list: list[mp.Queue] = [eval_in_queue, saver_in_queue]
        cpu_counter, available_cpu_list = 0, None
        if self.cfg.restrict_cpu:
            available_cpu_list = list(os.sched_getaffinity(0))
            print(f"Available CPUs in Main: {available_cpu_list}", flush=True)
        # inference server arrays
        game = get_game_from_config(self.cfg.game_cfg)
        obs_shape = game.get_obs_shape()
        out_shape = 1 + game.num_actions if self.cfg.net_cfg.predict_policy else 1
        n = int(self.cfg.max_eval_per_worker * self.cfg.num_worker / self.cfg.num_inference_server)
        input_arr_size = np.prod(np.asarray([n, *obs_shape])).item()
        output_arr_size = np.prod(np.asarray((n, out_shape))).item()
        input_list, input_rdy_list, output_list, output_rdy_list = [], [], [], []
        num_inference_server = self.cfg.num_inference_server
        if self.cfg.temperature_input and not self.cfg.single_sbr_temperature:
            # arrays for response inference server
            num_inference_server *= 2
        for inference_id in range(num_inference_server):
            input_rdy_list.append(mp.Array('i', n, lock=False))
            output_rdy_list.append(mp.Array('i', n, lock=False))
            input_list.append(mp.Array('f', input_arr_size, lock=False))
            output_list.append(mp.Array('f', output_arr_size, lock=False))
        # start updater
        if not self.cfg.only_generate_buffer:
            gpu_idx = 0 if self.cfg.updater_cfg.use_gpu else None
            cpu_list_updater = None
            if self.cfg.restrict_cpu and self.cfg.max_cpu_updater is not None:
                if available_cpu_list is None:
                    raise Exception("This should never happen")
                cpu_list_updater = available_cpu_list[cpu_counter:cpu_counter + self.cfg.max_cpu_updater]
                cpu_counter += self.cfg.max_cpu_updater
            seed_updater = random.randint(0, int(1e6))
            kwargs_updater = {
                'trainer_cfg': self.cfg,
                'data_in_queue': updater_in_queue,
                'net_queue_out': update_out_queue,
                'stop_flag': stop_flag,
                'info_queue': info_queue,
                'update_counter': update_counter,
                'save_state_after_seconds': self.cfg.save_state_after_seconds,
                'cpu_list': cpu_list_updater,
                'gpu_idx': gpu_idx,
                'prev_run_dir': Path(self.cfg.prev_run_dir) if self.cfg.prev_run_dir is not None else None,
                'prev_run_idx': self.cfg.prev_run_idx,
                'seed': seed_updater,
            }
            p = mp.Process(target=run_updater, kwargs=kwargs_updater)
            p.start()
            process_list.append(p)
        # start evaluator
        if not self.cfg.only_generate_buffer:
            cpu_list_eval = None
            if self.cfg.restrict_cpu and self.cfg.max_cpu_evaluator is not None:
                if available_cpu_list is None:
                    raise Exception("This should never happen")
                cpu_list_eval = available_cpu_list[cpu_counter:cpu_counter+self.cfg.max_cpu_evaluator]
                cpu_counter += self.cfg.max_cpu_evaluator
            seed_evaluator = random.randint(0, 2 ** 32 - 1)
            kwargs_evaluator = {
                'trainer_cfg': self.cfg,
                'net_queue': eval_in_queue,
                'stop_flag': stop_flag,
                'info_queue': info_queue,
                'prev_run_dir': Path(self.cfg.prev_run_dir) if self.cfg.prev_run_dir is not None else None,
                'prev_run_idx': self.cfg.prev_run_idx,
                'cpu_list': cpu_list_eval,
                'seed': seed_evaluator,
            }
            p = mp.Process(target=run_evaluator, kwargs=kwargs_evaluator)
            p.start()
            process_list.append(p)
        # start worker
        for worker_id in range(self.cfg.num_worker):
            cpu_list_worker = None
            if self.cfg.restrict_cpu and self.cfg.max_cpu_worker is not None:
                if available_cpu_list is None:
                    raise Exception("This should never happen")
                cpu_list_worker = available_cpu_list[cpu_counter:cpu_counter + self.cfg.max_cpu_worker]
            seed_worker = random.randint(0, 2 ** 32 - 1)
            worker_per_server = int(self.cfg.num_worker / self.cfg.num_inference_server)
            inference_id = math.floor(worker_id / worker_per_server)
            arr_input = False
            if self.cfg.temperature_input and not self.cfg.single_sbr_temperature:
                inference_id *= 2
                arr_input = True
            kwargs_worker = {
                'worker_id': worker_id,
                'trainer_cfg': self.cfg,
                'data_queue': collector_in_queue,
                'stop_flag': stop_flag,
                'info_queue': info_queue,
                'step_counter': step_counter,
                'episode_counter': episode_counter,
                'error_counter': error_counter,
                'input_arr': input_list[inference_id],
                'input_rdy_arr': input_rdy_list[inference_id],
                'output_arr': output_list[inference_id],
                'output_rdy_arr': output_rdy_list[inference_id],
                'cpu_list': cpu_list_worker,
                'seed': seed_worker,
                'debug': False,
                'input_arr2': input_list[inference_id + 1] if arr_input else None,
                'input_rdy_arr2': input_rdy_list[inference_id + 1] if arr_input else None,
                'output_arr2': output_list[inference_id + 1] if arr_input else None,
                'output_rdy_arr2': output_rdy_list[inference_id + 1] if arr_input else None,
            }
            p = mp.Process(target=run_worker, kwargs=kwargs_worker)
            p.start()
            process_list.append(p)
        if cpu_counter is not None and self.cfg.max_cpu_worker is not None:
            cpu_counter += self.cfg.max_cpu_worker
        # inference server
        cpu_list_inference = None
        if self.cfg.restrict_cpu and self.cfg.max_cpu_inference_server is not None:
            if available_cpu_list is None:
                raise Exception("This should never happen")
            cpu_list_inference = available_cpu_list[cpu_counter:cpu_counter + self.cfg.max_cpu_inference_server]
        arr_idx, gpu_idx = 0, int(self.cfg.updater_cfg.use_gpu)
        for inference_id in range(self.cfg.num_inference_server):
            inference_net_queue = mp.Queue(maxsize=self.cfg.distributor_out_qsize)
            seed_server = random.randint(0, 2 ** 32 - 1)
            kwargs_inference = {
                'trainer_cfg': self.cfg,
                'const_net_path': None,
                'net_queue': inference_net_queue,
                'stop_flag': stop_flag,
                'info_queue': info_queue,
                'seed': seed_server,
                'input_rdy_arr': input_rdy_list[arr_idx],
                'output_rdy_arr': output_rdy_list[arr_idx],
                'input_arr': input_list[arr_idx],
                'output_arr': output_list[arr_idx],
                'cpu_list': cpu_list_inference,
                'gpu_idx': gpu_idx if not self.cfg.merge_inference_update_gpu else 0,
                'prev_run_dir': Path(self.cfg.prev_run_dir) if self.cfg.prev_run_dir is not None else None,
                'prev_run_idx': self.cfg.prev_run_idx,
            }
            p = mp.Process(target=run_inference_server, kwargs=kwargs_inference)
            p.start()
            process_list.append(p)
            dist_out_queue_list.append(inference_net_queue)
            queue_list.append(inference_net_queue)
            arr_idx += 1
            gpu_idx += 1
            if self.cfg.temperature_input and not self.cfg.single_sbr_temperature:
                # proxy
                seed_server = random.randint(0, 2 ** 32 - 1)
                kwargs_inference = {
                    'trainer_cfg': self.cfg,
                    'const_net_path': self.cfg.proxy_net_path,
                    'net_queue': None,
                    'stop_flag': stop_flag,
                    'info_queue': info_queue,
                    'seed': seed_server,
                    'input_rdy_arr': input_rdy_list[arr_idx],
                    'output_rdy_arr': output_rdy_list[arr_idx],
                    'input_arr': input_list[arr_idx],
                    'output_arr': output_list[arr_idx],
                    'cpu_list': cpu_list_inference,
                    'gpu_idx': gpu_idx if not self.cfg.merge_inference_update_gpu else 0,
                    'prev_run_dir': Path(self.cfg.prev_run_dir) if self.cfg.prev_run_dir is not None else None,
                    'prev_run_idx': self.cfg.prev_run_idx,
                }
                p = mp.Process(target=run_inference_server, kwargs=kwargs_inference)
                p.start()
                process_list.append(p)
                arr_idx += 1
                gpu_idx += 1
        if self.cfg.max_cpu_inference_server is not None:
            cpu_counter += self.cfg.max_cpu_inference_server
        # cpu list for distributor, collector, saver and logger
        cpu_list_ldsc = None
        if self.cfg.restrict_cpu and self.cfg.max_cpu_log_dist_save_collect is not None:
            if available_cpu_list is None:
                    raise Exception("This should never happen")
            cpu_list_ldsc = available_cpu_list[cpu_counter:cpu_counter + self.cfg.max_cpu_log_dist_save_collect]
            cpu_counter += self.cfg.max_cpu_log_dist_save_collect
        # start distributor
        if not self.cfg.only_generate_buffer:
            bucket_size = int(self.cfg.logger_cfg.updater_bucket_size / self.cfg.updater_cfg.updates_until_distribution)
            kwargs_distributor = {
                'in_queue': update_out_queue,
                'net_queue_list': dist_out_queue_list,
                'stop_flag': stop_flag,
                'info_queue': info_queue,
                'info_bucket_size': bucket_size,
                'cpu_list': cpu_list_ldsc,
                'max_queue_size': self.cfg.distributor_out_qsize,
            }
            p = mp.Process(target=run_distributor, kwargs=kwargs_distributor)
            p.start()
            process_list.append(p)
        # start logger
        kwargs_logger = {
            'trainer_cfg': self.cfg,
            'info_queue': info_queue,
            'data_queue': collector_in_queue,
            'stop_flag': stop_flag,
            'step_counter': step_counter,
            'episode_counter': episode_counter,
            'update_counter': update_counter,
            'cpu_list': cpu_list_ldsc,
            'error_counter': error_counter,
        }
        p = mp.Process(target=run_logger, kwargs=kwargs_logger)
        p.start()
        process_list.append(p)
        # start model saver
        if not self.cfg.only_generate_buffer:
            kwargs_saver = {
                'trainer_cfg': self.cfg,
                'net_queue': saver_in_queue,
                'stop_flag': stop_flag,
                'cpu_list': cpu_list_ldsc,
                'info_queue': info_queue,
            }
            p = mp.Process(target=run_saver, kwargs=kwargs_saver)
            p.start()
            process_list.append(p)
        # start collector
        seed_collector = random.randint(0, 2 ** 32 - 1)
        kwargs_collector = {
            'trainer_cfg': self.cfg,
            'data_queue': collector_in_queue,
            'updater_queue': updater_in_queue,
            'updater_queue_maxsize': self.cfg.updater_in_qsize,
            'stop_flag': stop_flag,
            'info_queue': info_queue,
            'save_state': self.cfg.save_state,
            'save_state_after_seconds': self.cfg.save_state_after_seconds,
            'cpu_list': cpu_list_ldsc,  # only works on Linux
            'only_generate_buffer': self.cfg.only_generate_buffer,
            'prev_run_dir': Path(self.cfg.prev_run_dir) if self.cfg.prev_run_dir is not None else None,
            'seed': seed_collector,
            'grouped_sampling': self.cfg.updater_cfg.utility_loss != UtilityNorm.NONE,
        }
        p = mp.Process(target=run_collector, kwargs=kwargs_collector)
        p.start()
        process_list.append(p)
        # wait for keyboard interrupt to end training
        try:
            while not stop_flag.value:
                time.sleep(5)
        except KeyboardInterrupt:
            print('Detected Keyboard interrupt in main process', flush=True)
        # close children
        stop_flag.value = True
        time.sleep(3)
        # all queues need to be empty for children to close
        for q in queue_list:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    pass
        # join children
        for p in process_list:
            p.join(timeout=5)
        sys.exit(0)

    def start_training(self):
        if self.cfg.prev_run_dir is not None:
            raise ValueError("Cannot start new training if previous run directory is specified. Call Continue instead")
        if self.cfg.prev_run_idx is not None:
            raise ValueError("Cannot start new training if previous run index is specified. Call Continue instead")
        self._train()

    def continue_training(self):
        if self.cfg.prev_run_dir is None:
            raise ValueError("Cannot continue training without information about the previous run directory")
        if self.cfg.prev_run_idx is None:
            raise ValueError("Cannot continue training without information about the previous run index")
        if isinstance(self.cfg.prev_run_dir, str):
            self.cfg.prev_run_dir = self.cfg.prev_run_dir
        self._train()
