import itertools
import random
import sys

import numpy as np
import pytest
import torch
from sklearn.preprocessing import scale
from sklearn.utils._testing import (
    _convert_container,
    assert_allclose,
    assert_array_equal,
)

from bofire.domain.desirability_functions import (
    DeltaIdentityDesirabilityFunction,
    DesirabilityFunction,
    MaxIdentityDesirabilityFunction,
    MaxSigmoidDesirabilityFunction,
    MinIdentityDesirabilityFunction,
    MinSigmoidDesirabilityFunction,
    TargetDesirabilityFunction,
)
from bofire.strategies.botorch.utils.objectives import (
    AdditiveObjective,
    MultiplicativeObjective,
    Objective,
)


@pytest.mark.parametrize(
    "objective, desFunc",
    [
        (objective, desFunc) 
        for objective in [MultiplicativeObjective, AdditiveObjective]
        for desFunc in [DeltaIdentityDesirabilityFunction(w=1., ref_point=1.), 
                                MaxIdentityDesirabilityFunction(w=1.),
                                MaxSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =1.), 
                                MinIdentityDesirabilityFunction(w=1.),
                                MinSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =1.), 
                                TargetDesirabilityFunction (target_value= 5., steepness = 1., tolerance = 1e-3, w = 1.)
                                ]
    ],
)
def test_Objective_not_implemented(objective, desFunc):
    one_objective = objective(desFunc)
    x = torch.rand(20,1)

    with pytest.raises(NotImplementedError):
        one_objective.reward(x,None)
        




@pytest.mark.parametrize(
    "desFunc",
    [
        (desFunc) for desFunc in [DeltaIdentityDesirabilityFunction(w=0.5, ref_point=1., scale=0.8), 
                                MaxIdentityDesirabilityFunction(w=0.5),
                                MaxSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =0.5), 
                                MinIdentityDesirabilityFunction(w=0.5),
                                MinSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =0.5), 
                                TargetDesirabilityFunction (target_value= 5., steepness = 1., tolerance = 1e-3, w = 0.5)
                                ]
    ],
)
def test_Objective_desirability_function(desFunc):
    samples = samples = torch.rand(20,1)

    objective = MultiplicativeObjective(desFunc)
    assert_allclose(objective.reward(samples, desFunc)[0].detach().numpy(), torch.sign(desFunc(samples))*torch.abs(desFunc(samples))**0.5, rtol=1e-06)

    objective = AdditiveObjective(desFunc)
    assert_allclose(objective.reward(samples, desFunc)[0].detach().numpy(), desFunc(samples)*0.5, rtol=1e-06)


@pytest.mark.parametrize(
    "batch_shape, m, dtype",
    [
        (batch_shape, m, dtype) 
        for batch_shape, m, dtype in itertools.product(
        ([], [3]), 
        (2, 3), 
        (torch.float, torch.double)
        )
    ],
)
def test_Objective_max_identity(batch_shape, m, dtype):
    samples = torch.rand(*batch_shape, 2, m, dtype=dtype)
    desFunc = MaxIdentityDesirabilityFunction(w=0.5)

    objective = MultiplicativeObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], samples**0.5)

    objective = AdditiveObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], samples*0.5)

@pytest.mark.parametrize(
    "batch_shape, m, dtype",
    [
        (batch_shape, m, dtype) 
        for batch_shape, m, dtype in itertools.product(
        ([], [3]), 
        (2, 3), 
        (torch.float, torch.double)
        )
    ],
)
def test_Objective_min_identity(batch_shape, m, dtype):
    samples = torch.rand(*batch_shape, 2, m, dtype=dtype)
    desFunc = MinIdentityDesirabilityFunction(w=0.5)

    objective = MultiplicativeObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], -1*samples**0.5)

    objective = AdditiveObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], -1*samples*0.5)

@pytest.mark.parametrize(
    "batch_shape, m, dtype",
    [
        (batch_shape, m, dtype) 
        for batch_shape, m, dtype in itertools.product(
        ([], [3]), 
        (2, 3), 
        (torch.float, torch.double)
        )
    ],
)
def test_Objective_delta_identity(batch_shape, m, dtype):
    samples = torch.rand(*batch_shape, 2, m, dtype=dtype)

    desFunc = DeltaIdentityDesirabilityFunction(w=0.5, ref_point=5., scale=0.8)

    objective = MultiplicativeObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], ((5-samples)*0.8)**0.5)

    objective = AdditiveObjective(desFunc)
    assert torch.equal(objective.reward(samples, desFunc)[0], ((5-samples)*0.8)*0.5)

def test_MultiplicativeObjective_forward():
    (desFunc, desFunc2) = random.choices(
                                [DeltaIdentityDesirabilityFunction(w=0.5, ref_point=1.), 
                                MaxIdentityDesirabilityFunction(w=0.5),
                                MaxSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =0.5), 
                                MinIdentityDesirabilityFunction(w=0.5),
                                MinSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =0.5), 
                                TargetDesirabilityFunction (target_value= 5., steepness = 1., tolerance = 1e-3, w = 0.5)
                                ], k=2)

    objective = MultiplicativeObjective([desFunc, desFunc2])

    samples = torch.rand(20, 2)
    reward, _ = objective.reward(samples[:,0], desFunc)
    reward2, _ = objective.reward(samples[:,1], desFunc2)

    exp_reward = reward.detach().numpy()*reward2.detach().numpy()

    forward_reward = objective.forward(samples)

    assert_allclose(exp_reward, forward_reward, rtol=1e-06)

def test_AdditiveObjective_forward():
    (desFunc, desFunc2) = random.choices(
                                [DeltaIdentityDesirabilityFunction(w=0.5, ref_point=1.), 
                                MaxIdentityDesirabilityFunction(w=0.5),
                                MaxSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =0.5), 
                                MinIdentityDesirabilityFunction(w=0.5),
                                MinSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =0.5), 
                                TargetDesirabilityFunction (target_value= 5., steepness = 1., tolerance = 1e-3, w = 0.5)
                                ], k=2)

    objective = AdditiveObjective([desFunc, desFunc2])

    samples = torch.rand(20, 2)
    reward, _ = objective.reward(samples[:,0], desFunc)
    reward2, _ = objective.reward(samples[:,1], desFunc2)

    exp_reward = reward.detach().numpy()+reward2.detach().numpy()

    forward_reward = objective.forward(samples)

    assert_allclose(exp_reward, forward_reward, rtol=1e-06)

# TODO: test sigmoid behaviour 
# @pytest.mark.parametrize(
#     "batch_shape, m, dtype, desFunc",
#     [
#         (batch_shape, m, dtype, desFunc) 
#         for batch_shape, m, dtype, desFunc in itertools.product(
#         ([], [3]), 
#         (2, 3), 
#         (torch.float, torch.double), 
#         (MinSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =1.), MaxSigmoidDesirabilityFunction(steepness= 1., tp = 1., w =1.))
#         )
#     ],
# )
# def test_MultiplicativeObjective_sigmoid(batch_shape, m, dtype, desFunc):
#     objective = MultiplicativeObjective([desFunc])

#     samples = torch.rand(*batch_shape, 20, m, dtype=dtype)
#     reward, _ = objective.reward(samples, desFunc)

    #assert torch.equal(torch.topk(reward, 1, dim=0).indices, torch.topk(samples, 1, dim=0).indices)
    #assert torch.equal(torch.topk(reward, 1, largest=False, dim=0).indices, torch.topk(samples, 1, largest=False, dim=0).indices)

    # sort_samples, indices = torch.sort(samples, dim=- 1, descending=False)
    # delta_middle_sample = sort_samples[9,...]-sort_samples[8,...]
    # delta_middle_reward = reward[...,indices[9,...]]-reward[...,indices[8,...]]

    # delta_high_sample = sort_samples[0,...]-sort_samples[1,...]
    # delta_high_reward = reward[...,indices[0,...]]-reward[...,indices[1,...]]

    # assert delta_high_reward<delta_middle_reward
    # assert delta_high_reward<delta_high_sample
    # assert_allclose(delta_middle_sample, delta_middle_reward)
