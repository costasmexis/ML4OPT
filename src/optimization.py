from omlt import OffsetScaling, OmltBlock
from omlt.io.keras import load_keras_sequential
from omlt.neuralnet import FullSpaceSmoothNNFormulation
import pyomo.environ as pyo
from tensorflow import keras


class Optimization:
    def __init__(self, model_file: str, inputs : list, outputs : list):
        self.m = pyo.ConcreteModel()
        self.m.reformer = OmltBlock()
        self.nn_reformer = keras.models.load_model(model_file, compile=False)
        self.inputs = inputs
        self.outputs = outputs
        
        self.scaler = None
        self.scaled_input_bounds = None
        self.net = None

    def set_scaler(
        self,
        x_offset,
        x_factor,
        y_offset,
        y_factor,
        scaled_lb,
        scaled_ub,
    ):
        self.scaler = OffsetScaling(
            offset_inputs={i: x_offset[self.inputs[i]] for i in range(len(self.inputs))},
            factor_inputs={i: x_factor[self.inputs[i]] for i in range(len(self.inputs))},
            offset_outputs={i: y_offset[self.outputs[i]] for i in range(len(self.outputs))},
            factor_outputs={i: y_factor[self.outputs[i]] for i in range(len(self.outputs))},
        )

        self.scaled_input_bounds = {
            i: (scaled_lb[i], scaled_ub[i]) for i in range(len(self.inputs))
        }

    def load_net(self):
        self.net = load_keras_sequential(
            self.nn_reformer, scaling_object=self.scaler, scaled_input_bounds=self.scaled_input_bounds
        )
        
        self.m.reformer.build_formulation(FullSpaceSmoothNNFormulation(self.net))

    def solve(self, output_idx : int = 0, direction : str = 'maximize'):
        if direction == 'maximize':
            sense = pyo.maximize
        elif direction == 'minimize':
            sense = pyo.minimize
        else:
            raise ValueError('Direction must be either maximize or minimize')
        
        self.m.obj = pyo.Objective(expr=self.m.reformer.outputs[output_idx], sense=sense)
        
        solver = pyo.SolverFactory('ipopt')
        solver.solve(self.m, tee=True)
        
        print(f"Objective value: {pyo.value(self.m.obj)}")
        for i in range(len(self.inputs)):
            print(f"{self.inputs[i]}: {pyo.value(self.m.reformer.inputs[i])}")
