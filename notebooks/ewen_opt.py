import sys

sys.path.append('../')
import pyomo.environ as pyo
from src.ewenpyomo import ewen_model

def main():
    ewen_pyomo_model = ewen_model()
    ewen_pyomo_model.obj = pyo.Objective(expr=(ewen_pyomo_model.EFTotal), sense=pyo.minimize)

    solver = pyo.SolverFactory('ipopt')
    solver.solve(ewen_pyomo_model, tee=False)

    print(f'EFTotal value: {ewen_pyomo_model.obj()}')
    print(f'ERTotal value: {ewen_pyomo_model.ERTotal()}')
    print(f'interc value: {ewen_pyomo_model.interc()}')

    print(f'FiR_1 = {ewen_pyomo_model.FiR[1].value}')
    print(f'FiR_2 = {ewen_pyomo_model.FiR[2].value}')
    print(f'FiR_3 = {ewen_pyomo_model.FiR[3].value}')

    print(f'CiR_1 = {ewen_pyomo_model.CiR[1].value}')
    print(f'CiR_2 = {ewen_pyomo_model.CiR[2].value}')
    print(f'CiR_3 = {ewen_pyomo_model.CiR[3].value}')
    
if __name__ == '__main__':
    main()
    
