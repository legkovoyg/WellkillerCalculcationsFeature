import numpy as np
from base_calculation import Calculator
from abc import ABC, abstractmethod


class ClassicCalculator(Calculator):
    def __init__(self, input_dto):
        super().__init__()
        self.well_geometry = input_dto.well_geometry
        self.fluids = input_dto.fluids
        self.reservoir_parameters = input_dto.reservoir_parameters
        self.debit_jg = input_dto.debit_jg
        self.volume = input_dto.volume
        self.config = input_dto.config

        acceleration_of_free_fall = 9.81
        pressure_to_depth_ratio = self.reservoir_parameters.pressure / (
                acceleration_of_free_fall * self.fluids.oil.density)
        self.static_level = max(0, self.well_geometry.depth_of_well - pressure_to_depth_ratio)

        count_of_rows = int(self.volume/(self.debit_jg*self.config.dt) + 1)

        self.t = np.arange(0, self.volume/self.debit_jg + 1, self.config.dt)
        self.volume_jg = self.t * self.debit_jg
        self.volume_jg_led = self.volume_jg * (self.fluids.jgs.density/self.fluids.oil.density)
        self.height_jg = np.zeros(count_of_rows)
        self.height_jg_led = np.zeros(count_of_rows)
        self.dh_jg = np.zeros(1)
        self.dh_jg_led = np.zeros(1)
        self.speed_peresch = np.zeros(1)
        self.h_yr = np.zeros(count_of_rows)

        # Volumes
        self.nkt_oil_volume = np.zeros(count_of_rows)
        self.kp_oil_volume = np.zeros(count_of_rows)
        self.exp_oil_volume = np.zeros(count_of_rows)
        self.nkt_jg_volume = np.zeros(count_of_rows)
        self.kp_jg_volume = np.zeros(count_of_rows)
        self.exp_jg_volume = np.zeros(count_of_rows)

        # heights
        self.nkt_oil_height = np.zeros(count_of_rows)
        self.kp_oil_height = np.zeros(count_of_rows)
        self.exp_oil_height = np.zeros(count_of_rows)
        self.nkt_jg_height = np.zeros(count_of_rows)
        self.kp_jg_height = np.zeros(count_of_rows)
        self.exp_jg_height = np.zeros(count_of_rows)

        # Params
        self.q_pogl = np.zeros(count_of_rows)
        self.v_pogl = np.zeros(count_of_rows)
        self.v_pogl_sum= np.zeros(count_of_rows)
        self.r_oil_injection = np.zeros(count_of_rows)
        self.p_yst = np.zeros(count_of_rows)
        self.p_friction = np.zeros(count_of_rows)
        self.p_nkt = np.zeros(count_of_rows)
        self.p_kp= np.zeros(count_of_rows)
        self.p_exp = np.zeros(count_of_rows)
        self.p_pc = np.zeros(count_of_rows)

    @abstractmethod
    def first_step(self):
        pass

    @abstractmethod
    def second_step(self):
        pass

    @abstractmethod
    def third_step(self):
        pass

    @abstractmethod
    def iterative_step(self):
        pass

class ClassicCalculatorDirect(ClassicCalculator):

    def __init__(self, input_dto):
        self.g = 9.81
        super().__init__(input_dto)

        self.height_jg = self.volume_jg / self.well_geometry.nkt.inner_cross_sectional_area
        self.height_jg_led = self.height_jg * (self.fluids.jgs.density/self.fluids.oil.density)

        self.dh_jg = self.height_jg[1] - self.height_jg[0]
        self.dh_jg_led = self.dh_jg * (self.fluids.jgs.density/self.fluids.oil.density)

        self.speed_peresch = np.sqrt(2*self.g*self.dh_jg_led)
        self.h_yr = self.well_geometry.depth_of_well - self.static_level

    # HEIGHTS
    def h_nkt_jg(self): #TODO Грамотно зациклить все периодические параметры
        """
        Вычисляет значение по формуле:

          =IF(self.well_geometry.nkt.length=0; IF(self.nkt_jg_height[-1]+(self.dh_jg-self.V_pogl/$C$23)*$C$21/($C$21+$C$22)>$C$8; $C$8; self.nkt_jg_height[-1]+self.dh_jg*$C$21/($C$21+$C$22)); E32)

        Аргументы:
          self.well_geometry.nkt.length  - значение переменной self.well_geometry.nkt.length;
          self.nkt_jg_height[-1]   - значение ячейки self.nkt_jg_height[-1];
          self.dh_jg   - значение ячейки self.dh_jg;
          self.v_pogl[i]  - значение ячейки self.v_pogl[i];
          self.well_geometry.exp.inner_cross_sectional_area   - значение ячейки self.well_geometry.exp.inner_cross_sectional_area;
          self.well_geometry.nkt.inner_cross_sectional_area   - значение ячейки self.well_geometry.nkt.inner_cross_sectional_area;
          (self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area   - значение ячейки (self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area;
          self.well_geometry.nkt.length - значение ячейки self.well_geometry.nkt.length;
          self.height_jg[1]   - значение ячейки self.height_jg[1.

        Возвращает:
          Вычисленное по формуле значение.
        """
        i = 4 #TODO Грамотно зациклить все периодические параметры
        if self.nkt_oil_height[i] == 0:
            intermediate = self.nkt_jg_height[i-1] + (self.dh_jg - self.v_pogl[i] / self.well_geometry.exp.inner_cross_sectional_area) * self.well_geometry.nkt.inner_cross_sectional_area / (self.well_geometry.nkt.inner_cross_sectional_area + (self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area))
            if intermediate > self.well_geometry.nkt.length:
                return self.well_geometry.nkt.length
            else:
                return self.nkt_jg_height[i-1] + self.dh_jg * self.well_geometry.nkt.inner_cross_sectional_area / (self.well_geometry.nkt.inner_cross_sectional_area + (self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area))
        else:
            return self.height_jg[i]

    def h_kp_jg(self): #TODO Грамотно зациклить все периодические параметры
        """
        Вычисляет значение по формуле:

          =IF(R32>0; 0;
               IF(V31 + (F32 - AL32/$C$23)*$C$21/($C$21+$C$22) > $C$8;
                  $C$8;
                  IF(U32 = $C$8;
                     V31 + (F32 - AL32/$C$23)*$C$21/$C$22;
                     V31 + (F32 - AL32/$C$23)*$C$21/($C$21+$C$22)
                  )
               )
              )

        где соответствующие переменные представлены как:
          - R32 → self.nkt_oil_height[0
          - V31 → self.kp_jg_height[0]
          - F32 → self.dh_jg
          - AL32 → self.v_pogl_sum[0]
          - $C$23 → self.well_geometry.exp.inner_cross_sectional_area
          - $C$21 → self.well_geometry.nkt.inner_cross_sectional_area
          - $C$22 → self.(self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area)
          - $C$8  → self.well_geometry.nkt.length
          - U32  → self.kp_oil_height[0]
        """
        # Если self.nkt_oil_height[0] больше нуля, возвращаем 0
        i = 4 #TODO Грамотно зациклить все периодические параметры
        if self.nkt_oil_height[i] > 0:
            return 0

        # Вычисляем базовое значение по формуле
        # base_value = self.kp_jg_height[0] + (self.dh_jg - self.v_pogl_sum[0] / self.well_geometry.exp.inner_cross_sectional_area) * self.well_geometry.nkt.inner_cross_sectional_area / (self.well_geometry.nkt.inner_cross_sectional_area + self.(self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area))
        base_value = self.kp_jg_height[i-1] + (
                    self.dh_jg - self.v_pogl_sum[i] / self.well_geometry.exp.inner_cross_sectional_area) * self.well_geometry.nkt.inner_cross_sectional_area / (
                                 self.well_geometry.nkt.inner_cross_sectional_area + (
                                     self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area))

        # Если базовое значение превышает верхнее ограничение, возвращаем его
        if base_value > self.well_geometry.nkt.length:
            return self.well_geometry.nkt.length

        # Если self.kp_oil_height[0] равен верхнему ограничению, используем альтернативное вычисление
        # if self.kp_oil_height[0] == self.well_geometry.nkt.length:
        #     return self.kp_jg_height[0] + (self.dh_jg - self.v_pogl_sum[0] / self.well_geometry.exp.inner_cross_sectional_area) * self.well_geometry.nkt.inner_cross_sectional_area / self.(self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area)
        # Если self.kp_oil_height[0] равен верхнему ограничению, используем альтернативное вычисление
        if self.nkt_oil_height[i] == self.well_geometry.nkt.length:
            return self.kp_jg_height[i-1] + (
                        self.dh_jg - self.v_pogl_sum[i] / self.well_geometry.exp.inner_cross_sectional_area) * self.well_geometry.nkt.inner_cross_sectional_area / (
                        self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area)

        # В противном случае возвращаем базовое значение
        return base_value

    def h_kp_oil(self): #TODO Грамотно зациклить все периодические параметры
        i = 4 #TODO Грамотно зациклить все периодические параметры
        return self.h_yr[i] - self.kp_jg_height[i]

    def h_exp_jg(self): #TODO Грамотно зациклить все периодические параметры
        """
        Вычисляет значение по формуле:

          =IF(R32 > 0; 0; IF(W31 + AL32 / $C$23 > $F$23; $F$23; W31 + AL32 / $C$23))

        где:
          - R32  → self.nkt_oil_height[0]
          - W31  → self.exp_jg_height[i-1]
          - AL32 → self.v_pogl_sum[0]
          - $C$23 → self.well_geometry.exp.inner_cross_sectional_area
          - $F$23 → self.well_geometry.exp.length
        """
        i = 4 #TODO Грамотно зациклить все периодические параметры
        if self.nkt_oil_height[i] > 0:
            return 0
        candidate = self.exp_jg_height[i-1] + self.v_pogl_sum[0] / self.well_geometry.exp.inner_cross_sectional_area
        if candidate > self.well_geometry.exp.length:
            return self.well_geometry.exp.length
        return candidate

    def h_exp_oil(self): #TODO Грамотно зациклить все периодические параметры
        """
        Вычисляет значение по формуле:

          =IF(R32>0; $C$6-$C$8; IF($C$6-$C$8-W32<0; 0; $C$6-$C$8-W32))

        где:
          - R32   → self.nkt_oil_height[0]
          - $C$6  → self.well_geometry.depth_of_well
          - $C$8  → self.well_geometry.nkt.length
          - W32   → self.exp_jg_height[0]
        """
        i = 4 #TODO Грамотно зациклить все периодические параметры
        if self.nkt_oil_height[i] > 0:
            return self.well_geometry.depth_of_well - self.well_geometry.nkt.length
        result = self.well_geometry.depth_of_well - self.well_geometry.nkt.length - self.exp_jg_height[i]
        return 0 if result < 0 else result


    # PRESSURES
    def pressure_friction(self): #TODO: ОСТАНОВИЛСЯ ВОТ ТУТ
        """
        Вычисляет значение по формуле:

          =IF(R32+U32<$C$8; 0; 32*$L$23/$C$12^2 * ($K$10*R32 + $K$8*U32) + 32*$L$24/($C$14^2 - $C$13^2) * ($K$10*S32 + $K$8*V32))

        где соответствия переменных:
          - R32  → self.nkt_oil_height[0]
          - U32  → self.kp_oil_height[0]
          - $C$8 → self.well_geometry.nkt.length
          - $L$23 → self.L23
          - $C$12 → self.C12
          - $K$10 → self.K10
          - $K$8  → self.K8
          - $L$24 → self.L24
          - $C$14 → self.C14
          - $C$13 → self.C13
          - S32  → self.S32
          - V32  → self.V32
        """
        i = 4
        if self.nkt_oil_height[i] + self.kp_oil_height[i] < self.well_geometry.nkt.length:
            return 0

        term1 = 32 * self.h_yr / (self.well_geometry.nkt.inner_cross_sectional_area ** 2) * (self.K10 * self.nkt_oil_height[0] + self.K8 * self.kp_oil_height[0])
        term2 = 32 * self.h_yr / ((self.well_geometry.nkt.inner_cross_sectional_area ** 2) - (self.C13 ** 2)) * (self.K10 * self.S32 + self.K8 * self.V32)
        return term1 + term2

    def pressure_wellhead(self):
        """
        Расчет устьевого давления по формуле
        P устьевое = 101325+AP32
        где
        AP32 - Потери на трение
        :return: None
        """

        self.p_yst[1] = self.p_yst[0] + self.p_friction[0]

    def pressure_nkt(self): #TODO I - функция
        """
        Расчет давлния на низе НКТ по формуле
        Давление в НКТ=R32*$C$11*$C$9+U32*$C$11*$C$10
        где R32 - self.nkt_oil_height[i]
        C9 - self.fluids.oil.density
        U32 -self.nkt_jg_height[i]
        C11 - self.g
        C10 - self.fluids.jg.density
        :return: None
        """
        i = 4 #TODO Грамотно зациклить все периодические параметры
        self.p_nkt[i] = self.g * (self.nkt_oil_height[i] * self.fluids.oil.density + self.nkt_jg_height[i] * self.fluids.jg.density)

    def pressure_kp(self): #TODO Грамотно зациклить все периодические параметры
        """
        Расчет давления на низе кольцевого пространства
        =S33*$C$11*$C$9+V33*$C$11*$C$10
        где S33 - self.kp_oil_height[i]
        C9 -self.fluids.oil.density
        V33 - self.kp_jg_height[i]
        C11 - self.g
        C10 - self.fluids.jg.density
        :return:
        """
        i = 4 #TODO Грамотно зациклить все периодические параметры
        self.p_kp[i] = self.g * (self.kp_oil_height[i] * self.fluids.oil.density + self.kp_jg_height[i] * self.fluids.jg.density)

    def pressure_exp(self): #TODO Грамотно зациклить все периодические параметры
        """
        Расчет давления на участке эксплуатационной колонны
        =$C$11*(T33*$C$9+W33*$C$10)
        где
        С11 - self.g
        Т33 - self.
        С9 - self.fluids.oil.density
        W33 -
        C10 - self.fluids.jg.density
        :return:
        """
        i = 4 #TODO Грамотно зациклить все периодические параметры
        self.p_exp[i] = self.g * (self.exp_oil_height[i] * self.fluids.oil.density + self.exp_jg_height[i] * self.fluids.jg.density)

    def pressure_downhole(self):  #TODO Грамотно зациклить все периодические параметры
        """
        Расчет устьевого давления по формуле
        =SUM(AQ33;AS33;AO33)
        где
        AQ33 - self.p_nkt[i]
        AS33 - self.p_exp[i]
        AO33 - self.p_yst[i]

        :return:
        """

        i = 4
        self.p_pc[i] = self.p_nkt[i] + self.p_exp[i] + self.p_yst[i]

    def first_step(self):
        self.q_pogl[0] = 0
        self.v_pogl[0] = 0
        self.v_pogl_sum[0] = 0
        self.r_oil_injection[0] = 0.1  #TODO (вынести r_oil_injection в отдельный параметр)

        self.nkt_oil_height[0] = self.h_yr[0]
        self.nkt_jg_height[0] = 0
        self.kp_jg_height[0] = 0
        self.kp_oil_height[0] = self.nkt_oil_height[0]
        self.exp_jg_height[0] = 0
        self.exp_oil_height[0] = self.well_geometry.exp.length - self.well_geometry.nkt.length

        self.p_friction[0] = 0
        self.p_yst[0] = 101325
        self.p_nkt[0] = self.g * (self.nkt_jg_height[0] * self.fluids.jgs.density + self.nkt_oil_height[0] * self.fluids.oil.density)
        self.p_kp[0] = self.g * (self.kp_oil_height * self.fluids.oil.density + self.kp_oil_height * self.fluids.oil.density)
        self.p_exp[0] = self.g * (self.exp_oil_height * self.fluids.oil.density + self.exp_jg_height * self.fluids.oil.density)
        self.p_pc[0] = self.p_nkt[0] + self.p_kp[0] + self.p_exp[0]

        self.nkt_oil_volume[0] = self.nkt_oil_height[0] * self.well_geometry.nkt.inner_cross_sectional_area
        self.nkt_jg_volume[0] = self.nkt_jg_height[0] * self.well_geometry.nkt.inner_cross_sectional_area
        self.kp_oil_volume[0] = self.kp_oil_height[0] * (self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area)
        self.kp_jg_volume[0] = self.kp_jg_height[0] * (self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area)
        self.exp_oil_volume[0] = self.exp_oil_height[0] * self.well_geometry.exp.inner_cross_sectional_area
        self.exp_jg_volume[0] = self.exp_jg_height[0] * self.well_geometry.exp.inner_cross_sectional_area

    def second_step(self):
        self.q_pogl[1] = 0
        self.v_pogl[1] = self.q_pogl[1] * self.config.dt
        self.v_pogl_sum[1] = sum(self.v_pogl[0:1])
        self.r_oil_injection[1] = 0.1  # TODO (вынести r_oil_injection в отдельный параметр)

        self.nkt_oil_height[1] = self.h_yr[1]
        self.nkt_jg_height[1] = self.h_nkt_jg()
        self.kp_jg_height[1] = self.h_kp_jg()
        self.kp_oil_height[1] = self.h_kp_oil()
        self.exp_jg_height[1] = self.h_exp_jg()
        self.exp_oil_height[1] = self.h_exp_oil()

        self.p_friction[1] = self.pressure_friction()
        self.p_yst[1] = 101325 + self.p_friction[1]
        self.p_nkt[1] = self.g * (
                    self.nkt_jg_height[1] * self.fluids.jgs.density + self.nkt_oil_height[1] * self.fluids.oil.density)
        self.p_kp[1] = self.g * (
                    self.kp_oil_height[1] * self.fluids.oil.density + self.kp_oil_height[1] * self.fluids.oil.density)
        self.p_exp[1] = self.g * (
                    self.exp_oil_height[1] * self.fluids.oil.density + self.exp_jg_height[1] * self.fluids.oil.density)
        self.p_pc[1] = self.p_nkt[1] + self.p_kp[1] + self.p_exp[1]

        self.nkt_oil_volume[1] = self.nkt_oil_height[1] * self.well_geometry.nkt.inner_cross_sectional_area
        self.nkt_jg_volume[1] = self.nkt_jg_height[1] * self.well_geometry.nkt.inner_cross_sectional_area
        self.kp_oil_volume[1] = self.kp_oil_height[1] * (
                    self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area)
        self.kp_jg_volume[1] = self.kp_jg_height[1] * (
                    self.well_geometry.exp.inner_cross_sectional_area - self.well_geometry.nkt.outer_cross_sectional_area)
        self.exp_oil_volume[1] = self.exp_oil_height[1] * self.well_geometry.exp.inner_cross_sectional_area
        self.exp_jg_volume[1] = self.exp_jg_height[1] * self.well_geometry.exp.inner_cross_sectional_area

    def third_step(self):
        pass

    def iterative_step(self):
        pass

    def calculate(self, volume):
        self.first_step()
        self.second_step()
        self.third_step()
        while True:
            self.iterative_step()



class ClassicCalculatorReverse(ClassicCalculator):

    def __init__(self, input_dto):
        super().__init__(input_dto)
        self.height_jg = self.volume_jg / self.well_geometry.kp.inner_cross_sectional_area


    def first_step(self):
        pass

    def second_step(self):
        pass

    def third_step(self):
        pass

    def iterative_step(self):
        pass

    def calculate(self, volume):
        self.first_step()
        self.second_step()
        self.third_step()
        while True:
            self.iterative_step()


