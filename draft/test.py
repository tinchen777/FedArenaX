

class ABCModel:
    
    def __new__(cls, s):
        obj = super().__new__(cls, s)
        
        return 1
    
    def __init__(self):
        self.fc1 = ...
        self.fc2 = ...
        self.fc3 = ...
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
    
    def __repr__(self):
        pass
    

a = ABCModel()

dict

list

set

tuple

t = (1, 2, [])


str


