import pandas as pd

class DataLoader:
    def __init__(self, filename: str, target: str, index_col: int = 0):
        self.filename = filename
        self.df = self.load_data(index_col)
        self.target = target
        self.inputs = self.df.drop(target, axis=1).columns.values.tolist()
        self.outputs = [target]
            
    def load_data(self, index_col):
        """ Load data from a csv file """
        return pd.read_csv(self.filename, index_col=index_col)
    
    def normalization(self, df: pd.DataFrame):
        """ Normalize the data OMLT style """
        dfin = df[self.inputs]
        dfout = df[self.outputs]
        
        x_offset, x_factor = dfin.mean().to_dict(), dfin.std().to_dict()
        y_offset, y_factor = dfout.mean().to_dict(), dfout.std().to_dict()

        dfin = (dfin - dfin.mean()).divide(dfin.std())
        dfout = (dfout - dfout.mean()).divide(dfout.std())

        scaled_lb = dfin.min()[self.inputs].values
        scaled_ub = dfin.max()[self.inputs].values

        x = dfin[self.inputs].values
        y = dfout[self.outputs].values
        
        return x, y, x_offset, x_factor, y_offset, y_factor, scaled_lb, scaled_ub
                
                