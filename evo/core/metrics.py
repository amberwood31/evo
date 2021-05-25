# -*- coding: UTF8 -*-
"""
Provides metrics for the evaluation of SLAM algorithms.
author: Michael Grupp

This file is part of evo (github.com/MichaelGrupp/evo).

evo is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

evo is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with evo.  If not, see <http://www.gnu.org/licenses/>.
"""

import abc
import logging
import math
import sys
import typing
from enum import Enum

import numpy as np

from evo import EvoException
from evo.core import filters, trajectory
from evo.core.result import Result
from evo.core import lie_algebra as lie

import matplotlib.pyplot as plt
import evo.tools.plot as plot
from evo.tools.settings import SETTINGS
import evo.core.transformations as tr



if sys.version_info[0] >= 3 and sys.version_info[1] >= 4:
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})

logger = logging.getLogger(__name__)

PathPair = typing.Tuple[trajectory.PosePath3D, trajectory.PosePath3D]


class MetricsException(EvoException):
    pass


class StatisticsType(Enum):
    rmse = "rmse"
    mean = "mean"
    median = "median"
    std = "std"
    min = "min"
    max = "max"
    sse = "sse"
    segment_mean_std = "segment_mean_std"

class PoseRelation(Enum):
    full_transformation = "full transformation"
    translation_part = "translation part"
    rotation_part = "rotation part"
    rotation_angle_rad = "rotation angle in radians"
    rotation_angle_deg = "rotation angle in degrees"
    z = "z"
    xy = "xy"


class Unit(Enum):
    none = "unit-less"
    meters = "m"
    seconds = "s"
    degrees = "deg"
    radians = "rad"
    frames = "frames"


class VelUnit(Enum):
    meters_per_sec = "m/s"
    rad_per_sec = "rad/s"
    degrees_per_sec = "deg/s"


class Metric(ABC):
    @abc.abstractmethod
    def process_data(self, data):
        return

    @abc.abstractmethod
    def get_statistic(self, statistics_type):
        return

    @abc.abstractmethod
    def get_all_statistics(self):
        return

    @abc.abstractmethod
    def get_result(self):
        return


class PE(Metric):
    """
    Abstract base class of pose error metrics.
    """
    def __init__(self):
        self.unit = Unit.none
        self.error = np.array([])

    def __str__(self) -> str:
        return "PE metric base class"

    @abc.abstractmethod
    def process_data(self, data):
        return

    def get_statistic(self, statistics_type: StatisticsType):

        if statistics_type == StatisticsType.rmse:
            squared_errors = np.power(self.error, 2)
            return math.sqrt(np.mean(squared_errors))
        elif statistics_type == StatisticsType.sse:
            squared_errors = np.power(self.error, 2)
            return np.sum(squared_errors)
        elif statistics_type == StatisticsType.mean:
            return float(np.mean(self.error))
        elif statistics_type == StatisticsType.median:
            return np.median(self.error)
        elif statistics_type == StatisticsType.max:
            return np.max(self.error)
        elif statistics_type == StatisticsType.min:
            return np.min(self.error)
        elif statistics_type == StatisticsType.std:
            return float(np.std(self.error))
        else:
            raise MetricsException("unsupported statistics_type")

    def get_all_statistics(self) -> typing.Dict[str, float]:
        """
        :return: a dictionary {StatisticsType.value : float}
        """
        statistics = {}

        for s in StatisticsType:
            print(s.value)
            try:
                statistics[s.value] = self.get_statistic(s)
                print(statistics[s.value])
            except MetricsException as e:
                if "unsupported statistics_type" not in str(e):
                    raise
        return statistics

    def get_result(self, ref_name: str = "reference",
                   est_name: str = "estimate") -> Result:
        """
        Wrap the result in Result object.
        :param ref_name: optional, label of the reference data
        :param est_name: optional, label of the estimated data
        :return:
        """
        result = Result()
        metric_name = self.__class__.__name__
        result.add_info({
            "title": str(self),
            "ref_name": ref_name,
            "est_name": est_name,
            "label": "{} {}".format(metric_name,
                                    "({})".format(self.unit.value))
        })
        result.add_stats(self.get_all_statistics())
        if hasattr(self, "error"):
            result.add_np_array("error_array", self.error)
        return result


class RPE(PE):
    """
    RPE: relative pose error
    metric for investigating the odometry drift
    """
    def __init__(self,
                 pose_relation: PoseRelation = PoseRelation.translation_part,
                 delta: float = 1.0, delta_unit: Unit = Unit.frames,
                 rel_delta_tol: float = 0.1, all_pairs: bool = False):
        if delta < 0:
            raise MetricsException("delta must be a positive number")
        if delta_unit == Unit.frames and not isinstance(delta, int) \
                and not delta.is_integer():
            raise MetricsException(
                "delta must be integer for delta unit {}".format(delta_unit))
        self.delta = int(delta) if delta_unit == Unit.frames else delta
        self.delta_unit = delta_unit
        self.rel_delta_tol = rel_delta_tol
        self.pose_relation = pose_relation
        self.all_pairs = all_pairs
        self.E: typing.List[np.ndarray] = []
        self.error = np.array([])
        self.delta_ids: typing.List[int] = []
        if pose_relation == PoseRelation.translation_part:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.rotation_angle_deg:
            self.unit = Unit.degrees
        elif pose_relation == PoseRelation.rotation_angle_rad:
            self.unit = Unit.radians
        else:
            # dimension-less
            self.unit = Unit.none

    def __str__(self) -> str:
        title = "RPE w.r.t. {} ({})\nfor delta = {} ({})".format(
            self.pose_relation.value, self.unit.value, self.delta,
            self.delta_unit.value)
        if self.all_pairs:
            title += " using all pairs"
        else:
            title += " using consecutive pairs"
        return title

    @staticmethod
    def rpe_base(Q_i: np.ndarray, Q_i_delta: np.ndarray, P_i: np.ndarray,
                 P_i_delta: np.ndarray) -> np.ndarray:
        """
        Computes the relative SE(3) error pose for a single pose pair
        following the notation of the TUM RGB-D paper.
        :param Q_i: reference SE(3) pose at i
        :param Q_i_delta: reference SE(3) pose at i+delta
        :param P_i: estimated SE(3) pose at i
        :param P_i_delta: estimated SE(3) pose at i+delta
        :return: the RPE matrix E_i in SE(3)
        """
        Q_rel = lie.relative_se3(Q_i, Q_i_delta)
        P_rel = lie.relative_se3(P_i, P_i_delta)
        E_i = lie.relative_se3(Q_rel, P_rel)
        return E_i

    def process_data(self, data: PathPair) -> None:
        """
        Calculates the RPE on a batch of SE(3) poses from trajectories.
        :param data: tuple (traj_ref, traj_est) with:
        traj_ref: reference evo.trajectory.PosePath or derived
        traj_est: estimated evo.trajectory.PosePath or derived
        """
        if len(data) != 2:
            raise MetricsException(
                "please provide data tuple as: (traj_ref, traj_est)")
        traj_ref, traj_est = data
        if traj_ref.num_poses != traj_est.num_poses:
            raise MetricsException(
                "trajectories must have same number of poses")

        id_pairs = id_pairs_from_delta(traj_est.poses_se3, self.delta,
                                       self.delta_unit, self.rel_delta_tol,
                                       all_pairs=self.all_pairs)

        # Store flat id list e.g. for plotting.
        self.delta_ids = [j for i, j in id_pairs]

        self.E = [
            self.rpe_base(traj_ref.poses_se3[i], traj_ref.poses_se3[j],
                          traj_est.poses_se3[i], traj_est.poses_se3[j])
            for i, j in id_pairs
        ]

        logger.debug(
            "Compared {} relative pose pairs, delta = {} ({}) {}".format(
                len(self.E), self.delta, self.delta_unit.value,
                ("with all pairs." if self.all_pairs \
                else "with consecutive pairs.")))

        logger.debug("Calculating RPE for {} pose relation...".format(
            self.pose_relation.value))

        if self.pose_relation == PoseRelation.translation_part:
            self.error = [np.linalg.norm(E_i[:3, 3]) for E_i in self.E]
        elif self.pose_relation == PoseRelation.rotation_part:
            # ideal: rot(E_i) = 3x3 identity
            self.error = np.array([
                np.linalg.norm(lie.so3_from_se3(E_i) - np.eye(3))
                for E_i in self.E
            ])
        elif self.pose_relation == PoseRelation.full_transformation:
            # ideal: E_i = 4x4 identity
            self.error = np.array(
                [np.linalg.norm(E_i - np.eye(4)) for E_i in self.E])
        elif self.pose_relation == PoseRelation.rotation_angle_rad:
            self.error = np.array(
                [abs(lie.so3_log(E_i[:3, :3])) for E_i in self.E])
        elif self.pose_relation == PoseRelation.rotation_angle_deg:
            self.error = np.array([
                abs(lie.so3_log(E_i[:3, :3])) * 180 / np.pi for E_i in self.E
            ])
        else:
            raise MetricsException("unsupported pose_relation: ",
                                   self.pose_relation)


class APE(PE):
    """
    APE: absolute pose error
    metric for investigating the global consistency of a SLAM trajectory
    """
    def __init__(self,
                 pose_relation: PoseRelation = PoseRelation.translation_part):
        self.pose_relation = pose_relation
        self.E: typing.List[np.ndarray] = []
        self.error = np.array([])
        if pose_relation == PoseRelation.translation_part:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.z:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.rotation_angle_deg:
            self.unit = Unit.degrees
        elif pose_relation == PoseRelation.rotation_angle_rad:
            self.unit = Unit.radians
        else:
            self.unit = Unit.none  # dimension-less

    def __str__(self) -> str:
        title = "APE w.r.t. "
        title += (str(self.pose_relation.value) + " " +
                  ("(" + self.unit.value + ")" if self.unit else ""))
        return title

    @staticmethod
    def ape_base(x_t: np.ndarray, x_t_star: np.ndarray) -> np.ndarray:
        """
        Computes the absolute error pose for a single SE(3) pose pair
        following the notation of the Kümmerle paper.
        :param x_t: estimated absolute pose at t
        :param x_t_star: reference absolute pose at t
        .:return: the delta pose
        """
        return lie.relative_se3(x_t, x_t_star)

    def process_data(self, data: PathPair) -> None:
        """
        Calculates the APE on a batch of SE(3) poses from trajectories.
        :param data: tuple (traj_ref, traj_est) with:
        traj_ref: reference evo.trajectory.PosePath or derived
        traj_est: estimated evo.trajectory.PosePath or derived
        """
        if len(data) != 2:
            raise MetricsException(
                "please provide data tuple as: (traj_ref, traj_est)")
        traj_ref, traj_est = data
        if traj_ref.num_poses != traj_est.num_poses:
            raise MetricsException(
                "trajectories must have same number of poses")

        if self.pose_relation == PoseRelation.translation_part:
            # don't require full SE(3) matrices for faster computation
            self.E = traj_est.positions_xyz - traj_ref.positions_xyz
        elif self.pose_relation == PoseRelation.z:
            self.E = traj_est.positions_xyz - traj_ref.positions_xyz
        elif self.pose_relation == PoseRelation.xy:
            self.E = traj_est.positions_xyz - traj_ref.positions_xyz
           
        else:
            self.E = [
                self.ape_base(x_t, x_t_star) for x_t, x_t_star in zip(
                    traj_est.poses_se3, traj_ref.poses_se3)
            ]
        logger.debug("Compared {} absolute pose pairs.".format(len(self.E)))
        logger.debug("Calculating APE for {} pose relation...".format(
            (self.pose_relation.value)))

        if self.pose_relation == PoseRelation.translation_part:
            # E is an array of position vectors only in this case
            self.error = np.array([np.linalg.norm(E_i) for E_i in self.E])
        elif self.pose_relation == PoseRelation.z:
            self.error = np.array([E_i[2] for E_i in self.E])
        elif self.pose_relation == PoseRelation.xy:
            self.error = np.array([np.linalg.norm(E_i[0:2]) for E_i in self.E])
        
        elif self.pose_relation == PoseRelation.rotation_part:
            self.error = np.array([
                np.linalg.norm(lie.so3_from_se3(E_i) - np.eye(3))
                for E_i in self.E
            ])
        elif self.pose_relation == PoseRelation.full_transformation:
            self.error = np.array(
                [np.linalg.norm(E_i - np.eye(4)) for E_i in self.E])
        elif self.pose_relation == PoseRelation.rotation_angle_rad:
            self.error = np.array(
                [abs(lie.so3_log(E_i[:3, :3])) for E_i in self.E])
        elif self.pose_relation == PoseRelation.rotation_angle_deg:
            self.error = np.array([
                abs(lie.so3_log(E_i[:3, :3])) * 180 / np.pi for E_i in self.E
            ])
        else:
            raise MetricsException("unsupported pose_relation")

class SEG_APE(PE):
    """
    SEG_APE: divide the trajectory into segment, and then align origin 
            then compute APE
            
    """

    def __init__(self, pose_relation: PoseRelation = PoseRelation.translation_part, segment_length = 50, plot = False):
        self.segment_length = segment_length
        self.pose_relation = pose_relation
        self.error = []
        self.seg_error: typing.List[np.ndarray] = []
        self.segment_index = []

        self.plot_show = plot


        self.E: typing.List[List[np.ndarray]] = []
        if pose_relation == PoseRelation.translation_part:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.z:
            self.unit = Unit.meters
        elif pose_relation == PoseRelation.rotation_angle_deg:
            self.unit = Unit.degrees
        elif pose_relation == PoseRelation.rotation_angle_rad:
            self.unit = Unit.radians
        else:
            self.unit = Unit.none 

    def __str__(self) -> str:
        title = "segment APE w.r.t. "
        title += (str(self.pose_relation.value) + " " + 
        ("(" + self.unit.value + ")" if self.unit else "") + " "
        + "segment " + str(self.segment_length) + "m")
        return title

    def get_statistic(self, statistics_type: StatisticsType):

        if statistics_type == StatisticsType.rmse:
            return -1
        elif statistics_type == StatisticsType.sse:
            return -1
        elif statistics_type == StatisticsType.mean:
            return -1
        elif statistics_type == StatisticsType.median:
            return -1
        elif statistics_type == StatisticsType.max:
            return -1
        elif statistics_type == StatisticsType.min:
            return -1
        elif statistics_type == StatisticsType.std:
            return -1
        elif statistics_type == StatisticsType.segment_mean_std:
            max_segment_errors = np.array([np.max(x) for x in self.seg_error])
            print('max_segment_error:', max_segment_errors)
            plot_seg = True
            if plot_seg == True:
                fig = plt.figure()
                ax = fig.add_subplot()

                seg_index = np.array(self.segment_index)
                starting_index = seg_index[:,0]

                ax.plot(starting_index, max_segment_errors)                
                plt.xlabel('Starting pose index')
                plt.ylabel('Max error')
                plt.savefig('segment_max_error')
                if self.plot_show:
                    plt.show()

                max_segment_index = max_segment_errors.argmax()
                start = self.segment_index[max_segment_index][0]
                end = self.segment_index[max_segment_index][1]
                print('max segment index: ', start, end)


            return (np.mean(max_segment_errors), np.std(max_segment_errors))

           

        else:
            raise MetricsException("unsupported statistics_type")


    def last_pose_from_segment_length(self, gt_poses: trajectory.PosePath3D, first_pose):
        dist = gt_poses.distances
        # print(dist)

        for i in range(1,gt_poses.num_poses):  #todo check whether to use shape[0] or [1]
            # print('segment length: ', self.segment_length)
            if dist[i] > dist[first_pose] + self.segment_length:
                return i
        
        return -1

    def process_data(self, data: PathPair, align = False, align_origin = False, 
    align_odom = False) -> None:
        """
        Calculate the APE of segments from a batch of SE(3) poses from trajectories
        :param data: tuple (traj_ref, traj_est) with:
        traj_ref: reference evo.trajectory.PosePath or derived
        traj_est: estimated evo.trajectory.PosePath or derived

        """
        if len(data) != 2:
            raise MetricsException(
                "please provide data tuple as: (traj_ref, traj_est)")
        traj_ref, traj_est = data
        if traj_ref.num_poses != traj_est.num_poses:
            raise MetricsException(
                "trajectories must have same number of poses")

        step_size = 10
        i = 0

        for first_pose in range(0,traj_ref.num_poses, step_size):

            print('processing segment ', i)
            i = i+1
            last_pose = self.last_pose_from_segment_length(traj_ref, first_pose)

            if (last_pose == -1):
                break

            traj_ref_segment = trajectory.PosePath3D(poses_se3=traj_ref.poses_se3[first_pose:last_pose])
            traj_est_segment = trajectory.PosePath3D(poses_se3=traj_est.poses_se3[first_pose:last_pose])
            # if align or correct_scale:            
            if align:
                alignment_transformation = lie.sim3(
            *traj_est_segment.align(traj_ref_segment, False, False, -1))
            elif align_origin:
                print('align origin: ', align_origin)
                alignment_transformation = traj_est_segment.align_origin(traj_ref_segment)
            elif align_odom:
                alignment_transformation = traj_est_segment.align_odom(traj_ref_segment)
            
            self.segment_index.append([first_pose, last_pose])
            if i==1:

                print('starting idx: ', first_pose)
                print('ending idx: ', last_pose)
                # print('origin 0 position: ', traj_est_segment.poses_se3[0])
                # print('origin 1 position: ', traj_est_segment.poses_se3[1])
                # print('ref 0 position: ', traj_ref_segment.poses_se3[0])
                # print('results 1 position: ', traj_ref_segment.poses_se3[1])


                fig = plt.figure()
                

                plot_option = '3D'
                if plot_option == '2D':
                    ax = fig.add_subplot()#projection="3d"
                elif plot_option == '3D':
                    ax = fig.add_subplot(projection="3d")


                x_ref = traj_ref_segment.positions_xyz[:, 0]
                y_ref = traj_ref_segment.positions_xyz[:, 1]
                z_ref = traj_ref_segment.positions_xyz[:, 2]
                front_vector_origin = np.array([1,0,0,1])
                f_vecs = np.array([(np.dot(pose, front_vector_origin)-pose[:,3])*10 for pose in traj_ref_segment.poses_se3])
                if plot_option == '2D':
                    ax.plot(x_ref, y_ref)#, z_ref)

                    # plt.quiver(x_ref, y_ref, f_vecs[:,0], f_vecs[:,1], color='g')#, arrow_length_ratio=0.05)
                elif plot_option == '3D':
                    ax.plot(x_ref, y_ref, z_ref)

                    plt.quiver(x_ref, y_ref, z_ref, f_vecs[:,0], f_vecs[:,1], f_vecs[:,2], color='g', arrow_length_ratio=0.05)



                x_est = traj_est_segment.positions_xyz[:, 0]
                y_est = traj_est_segment.positions_xyz[:, 1]
                z_est = traj_est_segment.positions_xyz[:, 2]
                f_vecs = np.array([(np.dot(pose, front_vector_origin)-pose[:,3])*10 for pose in traj_est_segment.poses_se3])
                if plot_option =='2D':
                    ax.plot(x_est, y_est, color = 'r')

                    # plt.quiver(x_est, y_est, f_vecs[:,0], f_vecs[:,1], color='b')
                elif plot_option == '3D':

                    ax.plot(x_est, y_est, z_est, color = 'r')
                    plt.quiver(x_est, y_est, z_est, f_vecs[:,0], f_vecs[:,1], f_vecs[:,2], color='b', arrow_length_ratio=0.05)


                plt.title('segment_length: '+str(self.segment_length) + 'm')
                plt.savefig('first_segment_aligned.png')
                if self.plot_show:
                    plt.show()
                

            if self.pose_relation == PoseRelation.translation_part:
                segment_E = traj_est_segment.positions_xyz - traj_ref_segment.positions_xyz
            elif self.pose_relation == PoseRelation.z:
                segment_E = traj_est_segment.positions_xyz - traj_ref_segment.positions_xyz
            elif self.pose_relation == PoseRelation.xy:
                segment_E = traj_est_segment.positions_xyz - traj_ref_segment.positions_xyz
            else:
                segment_E = [
                    self.ape_base(x_t, x_t_star) for x_t, x_t_star in zip(
                        traj_est_segment.poses_se3, traj_ref_segment.poses_se3
                    )
                ]
            logger.debug("Compared {} absolute pose pairs.".format(len(self.E)))
            logger.debug("Calculating APE for {} pose relation...".format(
            (self.pose_relation.value)))  

            if self.pose_relation == PoseRelation.translation_part:
                segment_error = np.array([np.linalg.norm(E_i) for E_i in segment_E])
            
            elif self.pose_relation == PoseRelation.z:
                segment_error = np.array([E_i[2] for E_i in segment_E])
            elif self.pose_relation == PoseRelation.xy:
                segment_error = np.array([np.linalg.norm(E_i[0:2]) for E_i in segment_E])
            elif self.pose_relation == PoseRelation.rotation_part:
                segment_error = np.array([
                    np.linalg.norm(lie.so3_from_se3(E_i) - np.eye(3))
                    for E_i in segment_E
                ])
            elif self.pose_relation == PoseRelation.full_transformation:
                segment_error = np.array(
                    [np.linalg.norm(E_i - np.eye(4)) for E_i in segment_E]
                )
            elif self.pose_relation == PoseRelation.rotation_angle_rad:
                segment_error = np.array(
                [abs(lie.so3_log(E_i[:3, :3])) for E_i in segment_E])
            elif self.pose_relation == PoseRelation.rotation_angle_deg:
                segment_error = np.array([
                abs(lie.so3_log(E_i[:3, :3])) * 180 / np.pi for E_i in segment_E
                ])
            else:
                raise MetricsException("unsupported pose_relation")


            self.E.append(segment_E)
            self.seg_error.append(segment_error)       









def id_pairs_from_delta(poses: typing.Sequence[np.ndarray], delta: float,
                        delta_unit: Unit, rel_tol: float = 0.1,
                        all_pairs: bool = False) -> filters.IdPairs:
    """
    high-level function - get index tuples of pairs with distance==delta
    from a pose list
    :param poses: list of SE(3) poses
    :param delta: the interval step for indices
    :param delta_unit: unit of delta (metrics.Unit enum member)
    :param rel_tol: relative tolerance to accept or reject deltas
    :param all_pairs: use all pairs instead of consecutive pairs
    :return: list of index tuples (pairs)
    """
    if delta_unit == Unit.frames:
        id_pairs = filters.filter_pairs_by_index(poses, int(delta), all_pairs)
    elif delta_unit == Unit.meters:
        id_pairs = filters.filter_pairs_by_path(poses, delta, delta * rel_tol,
                                                all_pairs)
    elif delta_unit in {Unit.degrees, Unit.radians}:
        use_degrees = (delta_unit == Unit.degrees)
        id_pairs = filters.filter_pairs_by_angle(poses, delta, delta * rel_tol,
                                                 use_degrees, all_pairs)
    else:
        raise filters.FilterException(
            "unsupported delta unit: {}".format(delta_unit))

    if len(id_pairs) == 0:
        raise filters.FilterException(
            "delta = {} ({}) produced an empty index list - try lower values "
            "or a less strict tolerance".format(delta, delta_unit.value))

    logger.debug(
        "Found {} pairs with delta {} ({}) "
        "among {} poses ".format(len(id_pairs), delta, delta_unit.value,
                                 len(poses)) +
        ("using consecutive pairs." if not all_pairs else "using all pairs."))

    return id_pairs
