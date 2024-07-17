import pyomo.environ as pyo
import tensorflow.keras as keras
from omlt import OffsetScaling, OmltBlock
from omlt.io.keras import load_keras_sequential
from omlt.neuralnet import (
    FullSpaceNNFormulation,
    FullSpaceSmoothNNFormulation,
    NetworkDefinition,
    ReducedSpaceSmoothNNFormulation,
    ReluBigMFormulation,
    ReluComplementarityFormulation,
    ReluPartitionFormulation,
)


def create_model(
    x_offset,
    x_factor,
    y_offset,
    y_factor,
    scaled_lb,
    scaled_ub,
    inputs,
    outputs,
    file_name: str = 'cost_nn.keras',
):
    m = pyo.ConcreteModel()
    m.cost = OmltBlock()
    nn_cost = keras.models.load_model(file_name, compile=False)

    scaler = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))},
    )

    scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}

    net = load_keras_sequential(
        nn_cost, scaling_object=scaler, scaled_input_bounds=scaled_input_bounds
    )
    m.cost.build_formulation(ReluBigMFormulation(net))
    return m
