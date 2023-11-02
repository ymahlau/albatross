import unittest

import numpy as np
from matplotlib import pyplot as plt

from src.supervised.annealer import TemperatureAnnealingConfig, AnnealingType, TemperatureAnnealer


class TestAnnealer(unittest.TestCase):
    # LINEAR ##########################################################################
    def test_linear(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.LINEAR],
            end_times_min=[4],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_linear_offset(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[2, 5],
            anneal_types=[AnnealingType.CONST, AnnealingType.LINEAR],
            end_times_min=[4, 8],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_linear_cyclic(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.LINEAR],
            end_times_min=[3],
            cyclic=True,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    # COSINE ##########################################################################
    def test_cosine(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.COSINE],
            end_times_min=[4],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_cosine_offset(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[2, 5],
            anneal_types=[AnnealingType.CONST, AnnealingType.COSINE],
            end_times_min=[4, 8],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_cosine_cyclic(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.COSINE],
            end_times_min=[3],
            cyclic=True,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    # LOG ##########################################################################
    def test_log(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.LOG],
            end_times_min=[4],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_log_offset(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[2, 5],
            anneal_types=[AnnealingType.CONST, AnnealingType.LOG],
            end_times_min=[4, 8],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_log_cyclic(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.LOG],
            end_times_min=[3],
            cyclic=True,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    # DOUBLE COSINE ##########################################################################
    def test_double_cosine(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.DOUBLE_COS],
            end_times_min=[4],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_double_cosine_offset(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[2, 5],
            anneal_types=[AnnealingType.CONST, AnnealingType.DOUBLE_COS],
            end_times_min=[4, 8],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_double_cosine_cyclic(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.DOUBLE_COS],
            end_times_min=[3],
            cyclic=True,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

# ENHANCED DOUBLE COSINE ##########################################################################
    def test_enhanced_double_cosine(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.ENHANCED_DCOS],
            end_times_min=[4],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_enhanced_double_cosine_offset(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[2, 5],
            anneal_types=[AnnealingType.CONST, AnnealingType.ENHANCED_DCOS],
            end_times_min=[4, 8],
            cyclic=False,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

    def test_enhanced_double_cosine_cyclic(self):
        resolution = 100
        start = 0
        end = 10
        times = np.linspace(start, end, resolution)
        cfg = TemperatureAnnealingConfig(
            init_temp=1,
            anneal_temps=[5],
            anneal_types=[AnnealingType.ENHANCED_DCOS],
            end_times_min=[3],
            cyclic=True,
        )
        annealer = TemperatureAnnealer(cfg)
        y_list = []
        for i in range(resolution):
            y_list.append(annealer(times[i]))
        plt.clf()
        plt.plot(times, y_list)
        plt.show()

