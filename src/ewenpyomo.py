import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import numpy as np
import pandas as pd


def ewen_model():
    # Définition du modèle
    model = pyo.ConcreteModel()
    
    # Paramètres
    Nbprocede = 15
    Nbreg = 3
    GV = 1000000
    
    model.pro = pyo.RangeSet(1, Nbprocede)
    model.reg = pyo.RangeSet(1, Nbreg)
    model.IndA = pyo.RangeSet(1, 5)
    model.IndB = pyo.RangeSet(6, 10)
    model.IndC = pyo.RangeSet(11, 15)
    
    Cinmax = [0, 50, 50, 80, 800, 0, 50, 80, 100, 400, 0, 25, 25, 50, 100]
    Coutmax = [100, 80, 100, 800, 800, 100, 80, 400, 800, 1000, 100, 50, 125, 800, 150]
    Load = [2000, 2000, 5000, 30000, 4000, 2000, 2000, 5000, 30000, 4000, 2000, 2000, 5000, 30000, 15000]
    Coutreg = [20, 10, 10]
    
    model.Cinmax = pyo.Param(model.pro, initialize=lambda model, i: Cinmax[i-1])
    model.Coutmax = pyo.Param(model.pro, initialize=lambda model, i: Coutmax[i-1])
    model.Load = pyo.Param(model.pro, initialize=lambda model, i: Load[i-1])
    model.Coutreg = pyo.Param(model.reg, initialize=lambda model, i: Coutreg[i-1])
    
    # Variables
    model.EF = pyo.Var(model.pro, within=pyo.NonNegativeReals)
    model.yef = pyo.Var(model.pro, within=pyo.Binary)
    model.FRP = pyo.Var(model.reg, model.pro, within=pyo.NonNegativeReals)
    model.CRP = pyo.Var(model.reg, model.pro, within=pyo.NonNegativeReals)
    model.yrp = pyo.Var(model.reg, model.pro, within=pyo.Binary)
    model.FRR = pyo.Var(model.reg, model.reg, within=pyo.NonNegativeReals)
    model.CRR = pyo.Var(model.reg, model.reg, within=pyo.NonNegativeReals)
    model.yrr = pyo.Var(model.reg, model.reg, within=pyo.Binary)
    model.FPP = pyo.Var(model.pro, model.pro, within=pyo.NonNegativeReals)
    model.CPP = pyo.Var(model.pro, model.pro, within=pyo.NonNegativeReals)
    model.ypp = pyo.Var(model.pro, model.pro, within=pyo.Binary)
    model.FPE = pyo.Var(model.pro, within=pyo.NonNegativeReals)
    model.CPE = pyo.Var(model.pro, within=pyo.NonNegativeReals)
    model.ype = pyo.Var(model.pro, within=pyo.Binary)
    model.FPR = pyo.Var(model.pro, model.reg, within=pyo.NonNegativeReals)
    model.CPR = pyo.Var(model.pro, model.reg, within=pyo.NonNegativeReals)
    model.ypr = pyo.Var(model.pro, model.reg, within=pyo.Binary)
    model.FRE = pyo.Var(model.reg, within=pyo.NonNegativeReals)
    model.CRE = pyo.Var(model.reg, within=pyo.NonNegativeReals)
    model.Fin = pyo.Var(model.pro, within=pyo.NonNegativeReals)
    model.Fout = pyo.Var(model.pro, within=pyo.NonNegativeReals)
    model.Cin = pyo.Var(model.pro, within=pyo.NonNegativeReals)
    model.Cout = pyo.Var(model.pro, within=pyo.NonNegativeReals)
    model.FiR = pyo.Var(model.reg, within=pyo.NonNegativeReals)
    model.FoutR = pyo.Var(model.reg, within=pyo.NonNegativeReals)
    model.CiR = pyo.Var(model.reg, within=pyo.NonNegativeReals)
    model.CoR = pyo.Var(model.reg, within=pyo.NonNegativeReals)
    model.yre = pyo.Var(model.reg, within=pyo.Binary)
    model.interr = pyo.Var(model.reg, within=pyo.NonNegativeIntegers)
    model.interp = pyo.Var(model.pro, within=pyo.NonNegativeIntegers)
    model.interc = pyo.Var(within=pyo.NonNegativeIntegers)
    model.ERTotal = pyo.Var(within=pyo.NonNegativeReals)
    model.EFTotal = pyo.Var(within=pyo.NonNegativeReals)
    
    

    
    # Contraintes
    model.constraints = pyo.ConstraintList()
    
    model.constraints.add(model.interc==28)
    model.constraints.add(model.ERTotal>=200)
    model.constraints.add(model.ERTotal<=220)
    
    # Pollution process balance
    for j in model.pro:
        model.constraints.add(
            model.Cin[j] == sum(model.CPP[i, j] for i in model.pro) + sum(model.CRP[r, j] for r in model.reg)
        )
    
    # Pollution balance load
    for j in model.pro:
        model.constraints.add(
            model.Cin[j] + model.Load[j] == model.Cout[j]
        )
    
    # Pollution output balance
    for j in model.pro:
        model.constraints.add(
            model.Cout[j] == sum(model.CPP[j, i] for i in model.pro) + sum(model.CPR[j, r] for r in model.reg) + model.CPE[j]
        )
    
    # Water process balance in
    for j in model.pro:
        model.constraints.add(
            model.EF[j] + sum(model.FPP[i, j] for i in model.pro) + sum(model.FRP[r, j] for r in model.reg) == model.Fin[j]
        )
    
    # Water process balance equal
    for j in model.pro:
        model.constraints.add(
            model.Fin[j] == model.Fout[j]
        )
    
    # Water output balance
    for j in model.pro:
        model.constraints.add(
            model.Fout[j] == sum(model.FPR[j, r] for r in model.reg) + model.FPE[j] + sum(model.FPP[j, i] for i in model.pro)
        )
        
        
    # Mass balance for each regeneration unit
    for r in model.reg:
        model.constraints.add(
            model.CiR[r] == sum(model.CRR[m, r] for m in model.reg) + sum(model.CPR[j, r] for j in model.pro)
        )
        model.constraints.add(
            model.CoR[r] + model.CRE[r] == model.CiR[r]
        )
        model.constraints.add(
            model.CoR[r] == sum(model.CRP[r, j] for j in model.pro) + sum(model.CRR[r, m] for m in model.reg)
        )
    
    # Regeneration water balance
    for r in model.reg:
        model.constraints.add(
            model.FiR[r] == sum(model.FPR[j, r] for j in model.pro) + sum(model.FRR[m, r] for m in model.reg)
        )
        model.constraints.add(
            model.FiR[r] == model.FRE[r] + model.FoutR[r]
        )
        model.constraints.add(
            model.FoutR[r] == sum(model.FRP[r, j] for j in model.pro) + sum(model.FRR[r, m] for m in model.reg)
        )    
        
        
        
    # Total sewer balance
    for j in model.pro:
        for r in model.reg:
            model.constraints.add(
                sum(model.CPE[j] for j in model.pro) + sum(model.CRE[r] for r in model.reg) == sum(model.Load[j] for j in model.pro)
            )
    
    for j in model.pro:
        for r in model.reg:
            model.constraints.add(
                sum(model.FPE[j] for j in model.pro) + sum(model.FRE[r] for r in model.reg) == sum(model.EF[j] for j in model.pro)
            )
    
    # Constraints on process inputs and outputs
    for j in model.pro:
        model.constraints.add(
            model.Cin[j] - model.Cinmax[j] * model.Fin[j] <= 0
        )
    
    for j in model.pro:
        model.constraints.add(
            model.Cout[j] - model.Coutmax[j] * model.Fout[j] <= 0
        )
    
    for r in model.reg:
        model.constraints.add(
            model.CoR[r] - model.Coutreg[r] * model.FoutR[r] == 0
        )
    
    # Constraints on process outputs
    for r in model.reg:
        for j in model.pro:
            model.constraints.add(
                model.CRP[r, j] - model.Coutreg[r] * model.FRP[r, j] == model.CoR[r] - model.Coutreg[r] * model.FoutR[r]
            )
    
    for r in model.reg:
        for m in model.reg:
            model.constraints.add(
                model.CRR[r, m] - model.Coutreg[r] * model.FRR[r, m] == model.CoR[r] - model.Coutreg[r] * model.FoutR[r]
            )
    
    for j in model.pro:
        for r in model.reg:
            model.constraints.add(
                model.CPR[j, r] - model.Coutmax[j] * model.FPR[j, r] == model.Cout[j] - model.Coutmax[j] * model.Fout[j]
            )
    
    for j in model.pro:
        for i in model.pro:
            model.constraints.add(
                model.CPP[j, i] - model.Coutmax[j] * model.FPP[j, i] == model.Cout[j] - model.Coutmax[j] * model.Fout[j]
            )
    
    for j in model.pro:
        model.constraints.add(
            model.CPE[j] - model.Coutmax[j] * model.FPE[j] == model.Cout[j] - model.Coutmax[j] * model.Fout[j]
        )    
    
    
    #binaires    
    # Contraintes sur CPP et FPP
    for i in model.pro:
        for j in model.pro:
            model.constraints.add(model.CPP[i, j] <= GV * model.ypp[i, j])
            model.constraints.add(model.FPP[i, j] <= GV * model.ypp[i, j])
    
    # Contraintes sur CRP et FRP
    for r in model.reg:
        for j in model.pro:
            model.constraints.add(model.CRP[r, j] <= GV * model.yrp[r, j])
            model.constraints.add(model.FRP[r, j] <= GV * model.yrp[r, j])
    
    # Contraintes sur EF
    for j in model.pro:
        model.constraints.add(model.EF[j] <= GV * model.yef[j])
    
    # Contraintes sur CPR et FPR
    for j in model.pro:
        for r in model.reg:
            model.constraints.add(model.CPR[j, r] <= GV * model.ypr[j, r])
            model.constraints.add(model.FPR[j, r] <= GV * model.ypr[j, r])
    
    # Contraintes sur CRR et FRR
    for r in model.reg:
        for m in model.reg:
            model.constraints.add(model.CRR[r, m] <= GV * model.yrr[r, m])
            model.constraints.add(model.FRR[r, m] <= GV * model.yrr[r, m])
    
    # Contraintes sur CPE et FPE
    for j in model.pro:
        model.constraints.add(model.CPE[j] <= GV * model.ype[j])
        model.constraints.add(model.FPE[j] <= GV * model.ype[j])
    
    # Contraintes sur CRE et FRE
    for r in model.reg:
        model.constraints.add(model.CRE[r] <= GV * model.yre[r])
        model.constraints.add(model.FRE[r] <= GV * model.yre[r])    
        
        
        
    # Interdiction de l'auto-soutien
    for j in model.pro:
        model.constraints.add(model.FPP[j, j] == 0)
    
    for r in model.reg:
        model.constraints.add(model.FRR[r, r] == 0)
    
        
    # Interdiction de l'auto-soutien
    for j in model.pro:
        model.constraints.add(model.FPP[j, j] == 0)
    
    for r in model.reg:
        model.constraints.add(model.FRR[r, r] == 0)
    
        
    # Seulement 2 connexions inversées entre A et B
    model.constraints.add(
        sum(model.ypp[i, j] for i in model.IndA for j in model.IndB) +
        sum(model.yrp[1, j] for j in model.IndB) +
        sum(model.ypr[i, 2] for i in model.IndA) == 1
    )
    
    model.constraints.add(
        sum(model.ypp[j, i] for i in model.IndA for j in model.IndB) +
        sum(model.yrp[2, j] for j in model.IndA) +
        sum(model.ypr[i, 1] for i in model.IndB) == 1
    )
    
    # Entre A et C
    model.constraints.add(
        sum(model.ypp[i, j] for i in model.IndA for j in model.IndC) +
        sum(model.yrp[1, j] for j in model.IndC) +
        sum(model.ypr[i, 3] for i in model.IndA) == 1
    )
    
    model.constraints.add(
        sum(model.ypp[j, i] for i in model.IndA for j in model.IndC) +
        sum(model.yrp[3, j] for j in model.IndA) +
        sum(model.ypr[i, 1] for i in model.IndC) == 1
    )
    
    # Entre B et C
    model.constraints.add(
        sum(model.ypp[i, j] for i in model.IndB for j in model.IndC) +
        sum(model.yrp[2, j] for j in model.IndC) +
        sum(model.ypr[i, 3] for i in model.IndB) == 1
    )
    
    model.constraints.add(
        sum(model.ypp[j, i] for i in model.IndB for j in model.IndC) +
        sum(model.yrp[3, j] for j in model.IndB) +
        sum(model.ypr[i, 2] for i in model.IndC) == 1
    )
    
    
    # Existence d'au moins un flux d'eau douce
    model.constraints.add(
        sum(model.EF[j] for j in model.pro) >= 2
    )
    
    # Tous les processus doivent exister
    for j in model.pro:
        model.constraints.add(
            sum(model.ypp[i, j] for i in model.pro) +
            sum(model.yrp[r, j] for r in model.reg) +
            model.yef[j] >= 1
        )
    
    for j in model.pro:
        model.constraints.add(
            sum(model.ypp[j, i] for i in model.pro) +
            sum(model.ypr[j, r] for r in model.reg) >= 1
        )
    
    for r in model.reg:
        model.constraints.add(
            sum(model.ypr[j, r] for j in model.pro) +
            sum(model.yrr[m, r] for m in model.reg) >= 1
        )
    
    for r in model.reg:
        model.constraints.add(
            sum(model.yrp[r, j] for j in model.pro) +
            sum(model.yrr[r, m] for m in model.reg) >= 1
        )
    
    
    # Calcul de la somme des connexions dans le réseau
    for r in model.reg:
        model.constraints.add(
            sum(model.yrr[m, r] for m in model.reg) +
            sum(model.ypr[j, r] for j in model.pro) == model.interr[r]
        )
    
    for j in model.pro:
        model.constraints.add(
            sum(model.yrp[r, j] for r in model.reg) +
            model.yef[j] +
            sum(model.ypp[i, j] for i in model.pro) == model.interp[j]
        )
    
    model.constraints.add(
        sum(model.interr[r] for r in model.reg) +
        sum(model.interp[j] for j in model.pro) == model.interc
    )
    
    # Calcul de la somme des flux de régénération
    model.constraints.add(
        sum(model.FiR[r] for r in model.reg) == model.ERTotal
    )
    
    # Calcul de la somme des flux d'eau douce
    model.constraints.add(
        sum(model.EF[j] for j in model.pro) == model.EFTotal
    )
    
    return model