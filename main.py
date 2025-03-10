import calculations
import base_calculation as calc
import input_dto as dtos



NKT = dtos.Tube('NKT',1400, 0.062, 0.073)
EXP = dtos.Tube('EXP', 1500, 0.15, 0.163)

oil = dtos.Fluid('oil',
                 density=800,
                 viscosity=0.005,
                 permeability=5*10**-13)

jgs = dtos.Fluid('jgs',
                 density=1070,
                 viscosity=0.001,
                 permeability=2*10**-12)


config = dtos.Config(dt = 20)
params = dtos.ReservoirParameters(pressure = 10132500, plast_thickness=10, contour_radius=250, porosity=0.2)
well_geometry = dtos.WellGeometry(depth_of_well=1500,exp = EXP, nkt = NKT)
fluids = dtos.Fluids(oil=oil, jgs=jgs)
all_data = dtos.InputDTO(well_geometry,fluids,params, 0.01, 27, config = config)
Genius_Calculator = calculations.ClassicCalculatorDirect(input_dto=all_data)
Genius_Calculator.calculate(27)
print('q')
