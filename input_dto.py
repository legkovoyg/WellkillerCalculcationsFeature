from typing import Optional, List
from dataclasses import dataclass, field
import math


@dataclass(order=True)
class Tube:
    """
    Класс для описания труб (НКТ, эксплуатационная) с вычислением зависимых параметров.

    Атрибуты:
        name (str): Имя трубы.
        _length (float): Длина трубы (м). Должна быть положительной.
        _inner_diameter (float): Внутренний диаметр трубы (м). Должен быть положительным.
        _outer_diameter (float): Внешний диаметр трубы (м). Должен быть положительным и больше внутреннего.
    """
    name: str
    _length: float = field(metadata={'unit': 'm'})
    _inner_diameter: float = field(metadata={'unit': 'm'})
    _outer_diameter: float = field(metadata={'unit': 'm'})

    def __post_init__(self):
        if self._length <= 0:
            raise ValueError("Длина должна быть положительным числом!")
        if self._inner_diameter <= 0:
            raise ValueError("Внутренний диаметр должен быть положительным числом!")
        if self._outer_diameter <= 0:
            raise ValueError("Внешний диаметр должен быть положительным числом!")
        if self._inner_diameter >= self._outer_diameter:
            raise ValueError("Внутренний диаметр должен быть меньше внешнего диаметра!")

    # Свойства для основных параметров с проверкой корректности
    @property
    def length(self) -> float:
        return self._length

    @length.setter
    def length(self, value: float):
        if value <= 0:
            raise ValueError("Длина должна быть положительным числом!")
        self._length = value

    @property
    def inner_diameter(self) -> float:
        return self._inner_diameter

    @inner_diameter.setter
    def inner_diameter(self, value: float):
        if value <= 0:
            raise ValueError("Внутренний диаметр должен быть положительным числом!")
        if value >= self._outer_diameter:
            raise ValueError("Внутренний диаметр должен быть меньше внешнего диаметра!")
        self._inner_diameter = value

    @property
    def outer_diameter(self) -> float:
        return self._outer_diameter

    @outer_diameter.setter
    def outer_diameter(self, value: float):
        if value <= 0:
            raise ValueError("Внешний диаметр должен быть положительным числом!")
        if value <= self._inner_diameter:
            raise ValueError("Внешний диаметр должен быть больше внутреннего диаметра!")
        self._outer_diameter = value

    # Вычисляемые свойства (зависимые параметры)
    @property
    def inner_radius(self) -> float:
        return self._inner_diameter / 2

    @property
    def outer_radius(self) -> float:
        return self._outer_diameter / 2

    @property
    def thickness(self) -> float:
        return self.outer_radius - self.inner_radius

    @property
    def inner_perimeter(self) -> float:
        return 2 * math.pi * self.inner_radius

    @property
    def outer_perimeter(self) -> float:
        return 2 * math.pi * self.outer_radius

    @property
    def inner_cross_sectional_area(self) -> float:
        return math.pi * self.inner_radius ** 2

    @property
    def outer_cross_sectional_area(self) -> float:
        return math.pi * self.outer_radius ** 2

    @property
    def inner_surface_area(self) -> float:
        return self.inner_perimeter * self.length

    @property
    def outer_surface_area(self) -> float:
        return self.outer_perimeter * self.length

    @property
    def inner_volume(self) -> float:
        return self.inner_cross_sectional_area * self.length

    @property
    def outer_volume(self) -> float:
        return self.outer_cross_sectional_area * self.length

    @property
    def wall_volume(self) -> float:
        return self.outer_volume - self.inner_volume

    # Переопределение стандартных методов
    def __repr__(self) -> str:
        return (f"Tube(name={self.name!r}, length={self._length}, "
                f"inner_diameter={self._inner_diameter}, outer_diameter={self._outer_diameter})")

    def __str__(self) -> str:
        return (
            f"Труба -- {self.name}:\n"
            f"Ключевые параметры:\n"
            f"  Длина:                                  {self._length:.2f} м\n"
            f"  Внутренний диаметр:                     {self._inner_diameter:.2f} м\n"
            f"  Внешний диаметр:                        {self._outer_diameter:.2f} м\n"
            f"Зависимые параметры:\n"
            f"  Внутренний радиус:                      {self.inner_radius:.2f} м\n"
            f"  Внешний радиус:                         {self.outer_radius:.2f} м\n"
            f"  Внутренний периметр:                    {self.inner_perimeter:.2f} м\n"
            f"  Внешний периметр:                       {self.outer_perimeter:.2f} м\n"
            f"  Внутренняя площадь поперечного сечения: {self.inner_cross_sectional_area:.5f} м²\n"
            f"  Внешняя площадь поперечного сечения:    {self.outer_cross_sectional_area:.5f} м²\n"
            f"  Внутренняя площадь поверхности:         {self.inner_surface_area:.2f} м²\n"
            f"  Внешняя площадь поверхности:            {self.outer_surface_area:.2f} м²\n"
            f"  Объем по внутренней стенке:             {self.inner_volume:.2f} м³\n"
            f"  Объем по внешней стенке:                {self.outer_volume:.2f} м³\n"
            f"  Объем стенки:                           {self.wall_volume:.2f} м³\n"
            f"  Толщина стенки:                         {self.thickness:.2f} м"
            f"\n"
        )

@dataclass()
class Fluid:
    """
    Класс, представляющий физические свойства флюида.

    Атрибуты:
      Density (float): Плотность в кг/м³ (должна быть положительной).
      Viscosity (float): Вязкость в Па·с (должна быть положительной).
      Permeability (float): Проницаемость в м² (не может быть отрицательной).
    """
    name: str
    density: float = field(metadata={'unit': 'kg/m³'})
    viscosity: float = field(metadata={'unit': 'Pa·s'})
    permeability: float = field(metadata={'unit': 'm²'})

    def __post_init__(self):
        if self.density <= 0:
            raise ValueError("Плотность должна быть положительной")
        if self.viscosity <= 0:
            raise ValueError("Вязкость должна быть положительной")
        if self.permeability < 0:
            raise ValueError("Проницаемость не может быть отрицательной")

    def __str__(self) -> str:
        return (
            f"Флюид {self.name}:\n"
            f"  Плотность:     {self.density:>8.2f} кг/м³\n"
            f"  Вязкость:      {self.viscosity:>8.2f} Па·с\n"
            f"  Проницаемость: {self.permeability:>8.2e} м²"
            f"\n"
        )

class WellGeometry:
    def __init__(self,
                 depth_of_well: float,
                 exp: Tube,
                 nkt: Tube = None):
        self.depth_of_well = depth_of_well
        self.nkt = nkt
        self.exp = exp

class Fluids:
    def __init__(self,
                 oil: Fluid,
                 jgs: Fluid):
        self.oil = oil
        self.jgs = jgs

class ReservoirParameters:
    def __init__(self,
                 pressure: float,
                 plast_thickness: float,
                 contour_radius: float,
                 porosity: float):
        self.pressure = pressure  # Па
        self.plast_thickness = plast_thickness  # м
        self.contour_radius = contour_radius  # м
        self.porosity = porosity  # %

class Config:
    def __init__(self,
                 dt
                 ):
        self.dt = dt

class InputDTO:
    def __init__(self,
                 well_geometry: WellGeometry,
                 fluids: Fluids,
                 reservoir_parameters: ReservoirParameters,
                 debit_jg: Optional[float],
                 volume: float,
                 config: Config):
        self.well_geometry = well_geometry
        self.fluids = fluids
        self.reservoir_parameters = reservoir_parameters
        self.debit_jg = debit_jg
        self.volume = volume # м³/с
        self.config = config


