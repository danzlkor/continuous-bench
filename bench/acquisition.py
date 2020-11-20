"""
This module contains functions to simulate diffusion signals using diffusion_models.py and compute summary measures
using summary_measures.py
"""
from typing import List
import numpy as np
from argparse import Namespace
from dataclasses import dataclass, fields
from typing import Optional
import argparse
from typing import Sequence

b0_tresh = 0.1


@dataclass
class ShellParameters:
    """
    Acquisition parameters for a single shell of diffusion MRI data
    """
    bval: Optional[float] = None
    qval: Optional[float] = None
    diffusion_time: Optional[float] = None
    gradient_duration: Optional[float] = None
    TE: Optional[float] = None
    TR: Optional[float] = None
    TI: Optional[float] = None
    b_delta: float = 1.
    b_eta: float = 0.
    lmax: Optional[int] = 8

    def __post_init__(self):
        if self.bval is None:
            self.bval = self.qval ** 2 * self.diffusion_time
        if self.no_anisotropy:
            self.lmax = 0

    @property
    def no_anisotropy(self):
        return (self.bval < b0_tresh) or (self.b_delta == 0)

    @classmethod
    def add_to_parser(cls, parser: argparse.ArgumentParser):
        """
        Adds the acquisition parameters to the given parsers

        :param parser: parser that will be updated with a new group of acquisition parameter arguments
        """
        group = parser.add_argument_group(
            "Acquisition parameters",
            "Text files or floats defining the acquisition parameters of each volume. " +
            "Text files should have a single row or single column and contain a single value per volume.",
        )
        for var in fields(cls):
            group.add_argument('--' + var.name)

    @classmethod
    def from_parser_args(cls, args):
        """
        Creates shells based on command line arguments.

        Extracts the command line arguments added by :meth:`add_to_parser`

        :param args: Namespace with acquisition parameter arguments from the command line
        :return: Tuple with:

            - 1D array with the index of the shell for each volume
            - List with the individual shells
        """
        to_shell = {}
        for var in fields(cls):
            if getattr(args, var.name, None) is None:
                continue
            try:
                to_shell[var.name] = float(getattr(args, var.name))
            except ValueError:
                to_shell[var.name] = np.genfromtxt(getattr(args, var.name))
        return cls.create_shells(**to_shell)

    @classmethod
    def create_shells(cls, **parameters):
        """
        Creates multiple shells from a sequence of parameters

        :param parameters: floats or sequence with values of the acquisition parameters for every volume
        :return: Tuple with:

            - 1D array with the index of the shell for each volume
            - List with the individual shells
        """
        nparams = None
        for name, params in parameters.items():
            if np.array(params).size == 1:
                continue
            if nparams is None:
                nparams = len(params)
                ref = name
            elif nparams != len(params):
                raise ValueError("Found inconsistent number of volumes: " +
                                 f"{nparams} for {ref} and {len(params)} for {name}")

        if nparams is None:
            raise ValueError("None of the parameters vary between volumes, so can not define shells")

        if parameters['bval'].max() > 100:
            parameters['bval'] /= 1e3
            parameters['bval'] = np.round(parameters['bval'], 1)

        parameters['bval'][parameters['bval'] < b0_tresh] = 0

        within_range = np.ones((nparams, nparams), dtype='bool')
        for name, params in parameters.items():
            arr = np.array(params)
            if arr.size == 1:
                continue
            if name == 'bval':
                within_range &= abs(arr[:, None] - arr[None, :]) < b0_tresh
            else:
                within_range &= arr[:, None] == arr[None, :]

        indices = -np.ones(nparams, dtype='int')
        while (indices == -1).any():
            nshell = 0
            new_shell = np.zeros(within_range.shape[0], dtype='bool')
            new_shell[np.where(indices == -1)[0][0]] = True
            while nshell != new_shell.sum():
                nshell = new_shell.sum()
                new_shell = within_range[new_shell, :].any(0)
            indices[new_shell] = max(indices) + 1

        shells = []
        for idx in np.unique(indices):
            use = idx == indices
            shell_params = {}
            for name, params in parameters.items():
                if np.array(params).size == 1:
                    shell_params[name] = params
                else:
                    shell_params[name] = np.median(np.array(params)[use])
            shells.append(cls(**shell_params))
        return indices, shells


def to_string(shells: Sequence[ShellParameters]):
    """
    Writes the shells as a table

    :param shells: individual shells
    """
    used_fields = set()
    for field in fields(ShellParameters):
        for shell in shells:
            if getattr(shell, field.name) is not None:
                used_fields.add(field.name)
                break
    used_fields = sorted(used_fields)

    len_column = max(max(len(name) for name in used_fields) + 2, 8)

    fmt = '{{:{}s}}'.format(len_column)
    res = "|".join(fmt.format(name) for name in used_fields)
    res = res + '\n' + '-' * len(res)

    fmt = '{{:{}f}}'.format(len_column)
    for shell in shells:
        parts = []
        for name in used_fields:
            value = getattr(shell, name)
            if value is None:
                parts.append(' ' * len_column)
            else:
                parts.append(fmt.format(value))
        res = res + '\n' + '|'.join(parts)
    return res


@dataclass
class Acquisition:
    shells: List[ShellParameters]
    idx_shells: np.ndarray
    bvecs: np.ndarray
    name: str = ''

    @classmethod
    def load(cls, name='unnamed', acq_path='/Users/hossein/PycharmProjects/deltamicro_analysis/data/Acquisitions'):
        """
        reads acquisition protocol parameters from bval and bvec text files
        :param name: name of acq protocol
        :param acq_path: path to acquisition files
        :return acq: an acquisition object containing shells, shell indices of
        each measurement, and gradient directions
        """
        bvec_add = acq_path + '/' + name + '/bvecs'
        bval_add = acq_path + '/' + name + '/bvals'
        args = Namespace(bvec=bvec_add, bval=bval_add)
        idx_shells, shells = ShellParameters.from_parser_args(args)
        bvecs = np.genfromtxt(args.bvec).T
        print('loaded input shells:')
        print(to_string(shells))
        print('')
        return cls(shells, idx_shells, bvecs, name)


