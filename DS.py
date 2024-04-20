import os
import pandas as pd
import numpy as np
from scipy.signal import fftconvolve
import time
import gurobipy as gb
from gurobipy import GRB



class DualSouring():

    def __init__(self, path_init="", path_testbed="", testbed_name=None, size_m=5):
        """
        :param path_init: путь к папке, где будут хранится дата фреймы.
        :param path_testbed: путь к папке с данными о товарах
        :param testbed_name: имя файла с данными о товарах. если не указано, то файл будет сгенерирован
        :param size_m: количество товаров в генерируемом файле
        """

        self.path_init = path_init
        self.initialize_dataframes() # создаем дата фреймы

        self.path_testbed = path_testbed
        # читаем или генерируем файл с товарами
        if testbed_name:
            self.testbed_name = testbed_name
            self.input_parameters = pd.read_excel(os.path.join(path_testbed, testbed_name), index_col=0)
        else:
            self.testbed_name = "test_params"
            self.generate_input_parameters(size_m)

        # предварительная обработка файла
        self.input_parameters.dropna()
        self.input_parameters.drop_duplicates()
        self.input_parameters.sort_values(by=['item'])
        self.input_parameters.reset_index(drop=True)

        # количество товаров
        self.size = self.input_parameters.shape[0]

        # список товаров
        self.item_table = sorted(self.input_parameters['item'].unique())

        # словари где будут хранится генерируемый спрос и спрос в течении l_e+1 периода
        self.Demand_dict = {}
        self.Demand_lead_dict = {}

    # создание дата фреймов
    def initialize_dataframes(self):

        self.last_points_df = pd.DataFrame(columns=['heuristic', 'item', 'parameter', 'cost', 'opt_Se', 'E_qe', 'E_qr', 'E_OH', 'E_SO', 'E_emission'])
        self.first_points_df = pd.DataFrame(columns=['heuristic', 'item', 'parameter', 'cost', 'opt_Se', 'E_qe', 'E_qr', 'E_OH', 'E_SO', 'E_emission'])
        self.optimum_points_df = pd.DataFrame(columns=['heuristic', 'item', 'parameter', 'cost', 'opt_Se', 'E_qe', 'E_qr', 'E_OH', 'E_SO', 'E_emission'])
        self.highest_emission_df = pd.DataFrame(columns=['heuristic', 'item', 'parameter', 'cost', 'opt_Se', 'E_qe', 'E_qr', 'E_OH', 'E_SO', 'E_emission'])
        self.lowest_emission_df = pd.DataFrame(columns=['emission'])
        self.total_possible_emission_df = pd.DataFrame(columns=['total_emission'])

        self.cost_emission = pd.DataFrame(columns=['cost', 'emission', 'e_reduction_pct'])
        self.emission_target = pd.DataFrame(columns=['target'])

        self.policies_df = pd.DataFrame(columns=['heuristic', 'item', 'parameter', 'cost', 'opt_Se', 'E_qe', 'E_qr', 'E_OH', 'E_SO', 'E_emission'])
        self.all_policies_df = pd.DataFrame(columns=['heuristic', 'item', 'parameter', 'cost', 'opt_Se', 'E_qe', 'E_qr', 'E_OH', 'E_SO', 'E_emission'])

    # генерация списка товаров
    def generate_input_parameters(self, size):

        self.input_parameters = pd.DataFrame(columns=['item', 'heuristic', 'D_dist_type', 'average', 'cv', 'l_e', 'l_r', 'ce', 'cr', 'h', 'p', 'e_r','e_e', 'last_period', 'seed', 'warm_up', 'optimizer'])

        for i in range(1, size + 1):
            item = i
            heuristic = "DIP"
            D_dist_type = "neg_binomial"
            average = np.random.randint(10, 100)
            cv = np.random.uniform(0.1, 0.5)
            l_e = np.random.randint(1, 5)
            l_r = np.random.randint(5, 10)
            c_e = np.random.uniform(10, 20)
            c_r = np.random.uniform(5, 10)
            h = np.random.uniform(0.5, 5)
            p = np.random.uniform(5, 20)
            e_r = np.random.uniform(0.1, 3)
            e_e = np.random.uniform(3, 5)
            last_period = 10000
            seed = np.random.randint(100, 1000)
            warm_up = last_period // 5
            optimizer = "gold_sect"

            data = [item, heuristic, D_dist_type, average, cv, l_e, l_r, c_e, c_r, h, p, e_r, e_e, last_period, seed, warm_up, optimizer]

            self.input_parameters = pd.concat([self.input_parameters, pd.DataFrame(columns=self.input_parameters.columns, data=[data])], ignore_index=True)

        self.input_parameters.to_excel(os.path.join(self.path_testbed, self.testbed_name + '.xlsx'))

    # подготовка к алгоритму: генерация спроса и нахождение начальных параметров
    def prepare(self):
        start_t = time.time()
        print("generate demand start")
        self.generate_demand()
        print("generate demand finish")
        print('time is %f sec' % (time.time() - start_t))

        start_t = time.time()
        print("search initial parameters start")
        self.search_initial_parameters()
        print("search initial parameters finish")
        print('time is %f sec' % (time.time() - start_t))

    # запуск алгоритма
    def run(self, emission_percent=0.9):

        start_t = time.time()
        print("main algorithm start")
        output = self.dual_mode_multi_item_optimization(emission_percent)
        print("main algorithm finish")
        print('time is %f sec' % (time.time() - start_t))

        return output

    # создание словарей спроса (demand)
    def generate_demand(self):

        for item, group in self.input_parameters.groupby('item'):
            dist_type = group['D_dist_type'].iloc[0]
            mean = group['average'].iloc[0]
            cv = group['cv'].iloc[0]
            last_period = group['last_period'].iloc[0]
            seed = group['seed'].iloc[0]
            l_e = group['l_e'].iloc[0]

            self.Demand_dict[item] = self.demand_simulator(dist_type, mean, cv, last_period, seed)
            self.Demand_lead_dict[item] = self.demand_simulator(dist_type, mean * (l_e + 1), cv / np.sqrt(l_e + 1), last_period, seed)

    # генерация спроса для конкретного товара
    def demand_simulator(self, dist_type, mean, cv, last_period, seed=2 ** 10):

        np.random.seed(seed=seed)

        if dist_type == "neg_binomial":
            sd = mean * cv
            #n = mean ** 2 / (sd ** 2 - mean)
            n = mean ** 2 / (sd ** 2 - mean) if sd ** 2 > mean else 1
            p = mean / sd ** 2
            if p < 0 or p > 1:
                p = 0.5

            demand = np.random.negative_binomial(n, p, last_period+1)

            sim_mean = np.mean(demand)
            sim_cv = np.std(demand) / sim_mean

            return {
                "demand_series": demand,
                "simulated_mean": sim_mean,
                "simulated_cv": sim_cv,
                "time_interval": (0, int(len(demand)-1))
            }

    # поиск начального набора допустимых параметров
    # тут создаются 3 набора:
    # 1) набор, в котором delta = 0
    # 2) набор, в котором после поиска по всем delta выбирается тот, где emission минимальный
    # 3) набор, в котором после поиска по всем delta выбирается тот, где cost минимальный
    # в конце вычисляется lowest, highest, total emission для поиска target emission
    def search_initial_parameters(self):

        # функция которая ведет поиск по всем delta, чтобы параметр min_by был минимальным
        def helpful_func(input_parameters, min_by):
            mean = input_parameters['average'].values[0]
            cv = input_parameters['cv'].values[0]
            lr = input_parameters['l_r'].values[0]
            bracket = (0, np.ceil(mean) * (lr + 1) + 5 * np.ceil(mean) * cv * np.sqrt(lr + 1))
            f = lambda hue_param: abs(self.prepare_for_newsvendor(input_parameters=input_parameters, hue_param=hue_param)[min_by].values[0])
            optimize_ = self.golden_section_discret(f, bracket)
            opt_parameter = optimize_['x_min']
            return self.prepare_for_newsvendor(input_parameters=input_parameters, hue_param=opt_parameter)

        # решение задачи newsvendor с параметром delta=0 для каждого товара из списка
        # это будет first point
        for i in range(self.size):
            self.first_points_df = pd.concat([self.first_points_df, self.prepare_for_newsvendor(input_parameters=self.input_parameters.iloc[i:i + 1], hue_param=0)], ignore_index=True)
        self.first_points_df.to_excel(os.path.join(self.path_init, 'firstpoint.xlsx'))

        print("first point done")

        # поиск по всем возможным delta для каждого товара из списка
        # для каждого параметра решается newsvendor. выбирается тот вариант, где emission минимальный
        # это будет last point
        for i in range(self.size):
            self.last_points_df = pd.concat([self.last_points_df, helpful_func(input_parameters=self.input_parameters.iloc[i:i + 1], min_by="E_emission")], ignore_index=True)
        self.last_points_df.to_excel(os.path.join(self.path_init, 'lastpoint.xlsx'))

        print("last point done")

        # поиск по всем возможным delta для каждого товара из списка
        # для каждого параметра решается newsvendor. выбирается тот вариант, где cost минимальный
        # это будет omptimum point
        for i in range(self.size):
            self.optimum_points_df = pd.concat([self.optimum_points_df, helpful_func(input_parameters=self.input_parameters.iloc[i:i + 1], min_by="cost")], ignore_index=True)
        self.optimum_points_df.to_excel(os.path.join(self.path_init, 'optimumpoint.xlsx'))

        print("optimum point done")

        # расчет emission: lowest, highest, total_possible. все это нужно, чтобы потом найти target emission
        firstpoints_emission_index = self.last_points_df.groupby(['item']).idxmin().cost
        firstpoints_emission = self.last_points_df.iloc[firstpoints_emission_index]

        lastpoints_emission_index = self.last_points_df.groupby(['item']).idxmin().cost
        lastpoints_emission = self.last_points_df.iloc[lastpoints_emission_index]

        optimalpoints_emission_index = self.optimum_points_df.groupby(['item']).idxmin().cost
        optimalpoints_emission = self.optimum_points_df.iloc[optimalpoints_emission_index]

        self.highest_emission_df = optimalpoints_emission.copy()
        self.highest_emission_df.to_excel(os.path.join(self.path_init, 'highest_emission.xlsx'))

        lowest_emission_policies = np.minimum(firstpoints_emission['E_emission'].values, lastpoints_emission['E_emission'].values)
        lowest_emission_policies_df = pd.DataFrame(columns=['emission'], data=lowest_emission_policies)

        self.lowest_emission_df = lowest_emission_policies_df.copy()
        self.lowest_emission_df.to_excel(os.path.join(self.path_init, 'lowest_emission.xlsx'))

        self.total_possible_emission_df.loc[0] = [(optimalpoints_emission['E_emission'].values - lowest_emission_policies).sum()]
        self.total_possible_emission_df.to_excel(os.path.join(self.path_init, 'total_possible_emission.xlsx'))

        # сохранение всех policy, который были получены в процессе решения задачи newsvendor
        self.policies_df.to_excel(os.path.join(self.path_init, 'p_df.xlsx'))
        self.all_policies_df.to_csv(os.path.join(self.path_init, 'p_df_all.csv'))

    # по описанию из статьи: наша задача становится задачей integer programming (IP)
    # сначала формулируется master problem
    # затем формулируются и решаются подзадачи с их оптимальными параметрами по каждому товару
    # решение IP будет производится с помощью библиотеки gurobipy
    def dual_mode_multi_item_optimization(self, e_reduct_pct):

        final_df = self.last_points_df.copy()

        # составляем коэффициенты для целевой функции и для emission.
        # эти параметры берутся из тех наборов, которые были найдены в search_initial_parameters
        Cost_0 = self.last_points_df['cost'].values
        Cost_optimalp = self.optimum_points_df['cost'].values
        Cost_first = self.first_points_df['cost'].values

        Emission_0 = self.last_points_df['E_emission'].values
        Emission_optimalp = self.optimum_points_df['E_emission'].values
        Emission_first = self.first_points_df['E_emission'].values

        # составляем emission target используя данные полученные в search_initial_parameters
        Emission_Target = self.lowest_emission_df['emission'].sum() + self.total_possible_emission_df['total_emission'].iloc[0] * (1 - e_reduct_pct)

        # Начинаем формулировать master problem
        MP = gb.Model("Master_Problem")
        MP.Params.LogToConsole = 0 # отключаем вывод информации в консоль
        MP.Params.FeasibilityTol = 10 ** -9 # ставим допустимую погрешность в поиске решений

        # добавление коэффициентов x^k_m, для которых указываются целевые функции Cost0, cost_optimal, cost_first
        # они не могут быть меньше 0 и являются непрерывними
        x_mk = {}
        x_mk[0] = MP.addMVar(self.size, obj=Cost_0, lb=0, vtype=GRB.CONTINUOUS)
        x_mk['opt'] = MP.addMVar(self.size, obj=Cost_optimalp, lb=0, vtype=GRB.CONTINUOUS)
        x_mk['first'] = MP.addMVar(self.size, obj=Cost_first, lb=0, vtype=GRB.CONTINUOUS)

        # добавление ограничения на emission (неравенство (22b) из статьи)
        C_Emission = MP.addConstr((Emission_0 @ x_mk[0] + Emission_optimalp @ x_mk['opt'] + Emission_first @ x_mk['first']) <= Emission_Target, name='C_Emission')

        # добавление условия что sum(x^k_m)=1 для каждого товара
        # ISPC=Item Single Policy Constraint
        C_ISPC = {}
        for m in self.item_table:
            ISPC_vars = []
            for k in final_df[final_df['item'] == m].index:
                ISPC_vars.append(x_mk[0][k])
                ISPC_vars.append(x_mk['opt'][k])
                ISPC_vars.append(x_mk['first'][k])
            C_ISPC[m] = MP.addConstr(sum(ISPC_vars) == 1, name='C_ISPC' + str(m))

        # объеденим в один lastpoint, omptimumpoint, firstpoint
        final_df = pd.concat([final_df, self.optimum_points_df], ignore_index=True)
        final_df = pd.concat([final_df, self.first_points_df], ignore_index=True)

        # определяем reduced_cost, который будет условием остановки цикла
        reduced_cost = -np.ones(len(self.item_table))
        reduced_cost = pd.Series(index=self.item_table, data=reduced_cost)

        # решение подзадачи
        def SubP_min(input_parameters, lambda1, lambda2):
            _heuristic = input_parameters['heuristic'].values[0]
            mean = input_parameters['average'].values[0]
            cv = input_parameters['cv'].values[0]
            lr = input_parameters['l_r'].values[0]

            bracket = (0, np.ceil(mean) * (lr + 1) + 5 * np.ceil(mean) * cv * np.sqrt(lr + 1))

            f_m = lambda hue_param: self.prepare_for_newsvendor(input_parameters=input_parameters, hue_param=hue_param)

            def f(hue_param):
                ds_solution = f_m(hue_param)
                return ds_solution['cost'].values[0] - lambda1 * ds_solution['E_emission'].values[0] - lambda2

            optimize_ = self.golden_section_discret(f, bracket)

            return {'parameter': optimize_['x_min'], 'reduced_cost': optimize_['y_min'], 'lambda1': lambda1,
                    'lambda2': lambda2}


        while reduced_cost.min() <= 0:

            MP.update()
            MP.optimize()

            for m in self.item_table:

                reduced_cost_local = 1

                # берем конкретный товар со всеми его параметрами и решаем подзадачу
                input_parameter = self.input_parameters[(self.input_parameters['item'] == m)]
                current_point = SubP_min(input_parameter, C_Emission.Pi, C_ISPC[m].Pi)
                # предыдущий набор параметров для этого товара, который имеет максимальный delta
                last_point_parameter = self.last_points_df[(self.last_points_df['item'] == m)]['parameter'].max()

                # список всех найденных параметров delta для товара m
                parameters_list = final_df[(final_df['item'] == m)]['parameter'].values

                next_point_parameter = min(current_point['parameter'] + 1, last_point_parameter)
                previous_point_parameter = max(current_point['parameter'] - 1, 0)

                # проверка на то, чтобы не добавлять повторно те же наборы параметров
                repeat_control = not (current_point['parameter'] in parameters_list and next_point_parameter in parameters_list and previous_point_parameter in parameters_list)

                if current_point['reduced_cost'] <= 0 and current_point['parameter'] < last_point_parameter and repeat_control:
                    hue_param = current_point['parameter']
                    if current_point['parameter'] in parameters_list:
                        if next_point_parameter in parameters_list:
                            hue_param = previous_point_parameter
                        else:
                            hue_param = next_point_parameter

                    new_point = self.prepare_for_newsvendor(input_parameters=input_parameter, hue_param=hue_param)

                    # проверка на то, что набор параметров new_point еще не был добавлен
                    if not (final_df == new_point.iloc[0]).all(1).any():
                        # добавляем в общий data frame
                        final_df = pd.concat([final_df, new_point], ignore_index=True)
                        # добавляем новую колонку к master problem
                        index_ = final_df[-1:].index[0]
                        newcolumn = gb.Column(coeffs=[final_df['E_emission'].iloc[-1], 1], constrs=[MP.getConstrByName('C_Emission'), MP.getConstrByName('C_ISPC' + str(m))])
                        x_mk[index_] = MP.addVar(obj=final_df['cost'].iloc[-1], column=newcolumn)

                    if current_point['reduced_cost'] < reduced_cost_local:
                        reduced_cost_local = current_point['reduced_cost']

                reduced_cost[m] = reduced_cost_local

        for index_ in x_mk:
            x_mk[index_].vtype = GRB.BINARY

        MP.update()
        MP.optimize()

        # поиск ненулевых переменных. они будут оптимальными параметрами
        nzero_index = []
        for v in MP.getVars():
            if round(v.x, 0) != 0:
                var_name = v.varName[1:]
                nzero_index.append(int(var_name))
        solutions = final_df.iloc[nzero_index].sort_values('item').reset_index(drop=True)

        total_emission = solutions['E_emission'].sum()

        # сохранение всех дата фреймов
        self.policies_df.to_excel(os.path.join(self.path_init, 'p_df.xlsx'))
        self.all_policies_df.to_csv(os.path.join(self.path_init, 'p_df_all.csv'))

        self.cost_emission = pd.concat([self.cost_emission, pd.DataFrame(columns=self.cost_emission.columns, data=[[MP.ObjVal, total_emission, e_reduct_pct]])], ignore_index=True)
        self.cost_emission.to_excel(os.path.join(self.path_init, 'cost_emission.xlsx'))

        solutions.to_excel(os.path.join(self.path_init, 'solutions.xlsx'))

        final_df.to_excel(os.path.join(self.path_init, 'final_df.xlsx'))

        self.emission_target = pd.concat([self.emission_target, pd.DataFrame(columns=self.emission_target.columns, data=[[Emission_Target]])], ignore_index=True)
        self.emission_target.to_excel(os.path.join(self.path_init, 'emissionstargets.xlsx'))

        return {'Opt_Emission': total_emission, 'Emission_Target': Emission_Target,
                'Total_Cost': MP.ObjVal, 'Optimal_Sol': solutions, 'repository': final_df}

    # для заданного товара m и параметра delta решается задача newsvendor для поиска оптимального S_e
    # алгоритм действий:
    # 1. проверка на то, решалась ли задача с таким параметром delta
    # 2. подготовка данных: генерация вероятностей (pmf) для demand, net demand, overshoot, q_r, q_e
    # 3. решение задачи newsvendor и расчет emissions
    # 4. запись результатов в policies_df и all_policies_df (сюда идет запись, даже, если с таким параметром задача решалась)
    def prepare_for_newsvendor(self, input_parameters, hue_param):

        m = input_parameters.item.values[0]
        k = input_parameters.heuristic.values[0]
        warm_up = input_parameters.warm_up.values[0]

        last_period = self.Demand_dict[m]['time_interval'][1]

        keys_ = ((self.policies_df['item'] == m) & (self.policies_df['heuristic'] == k) & (self.policies_df['parameter'] == hue_param))
        if keys_.any():
            outcome_df = self.policies_df[keys_].reset_index(drop=True)
        else:
            TS = self.dual_sourcing_heuristic(parameter=hue_param, demand_array=self.Demand_dict[m]['demand_series'], last_period=last_period, le=input_parameters.l_e.values[0], lr=input_parameters.l_r.values[0])

            D_le1_pmf = self.dist_generator(self.Demand_lead_dict[m]['demand_series'][warm_up:])
            O_pmf = self.dist_generator(TS['overshoot_series'][warm_up:])
            D_net_pmf = self.convolve(D_le1_pmf, O_pmf, minus_operator=True)

            qr_pmf = self.dist_generator(TS['reg_order_series'][warm_up:])
            qe_pmf = self.dist_generator(TS['exp_order_series'][warm_up:])

            NV = self.newsvendor(ce=input_parameters.ce.values[0], cr=input_parameters.cr.values[0],
                                 h=input_parameters.h.values[0], p=input_parameters.p.values[0], d_net_pmf=D_net_pmf,
                                 qr_pmf=qr_pmf, qe_pmf=qe_pmf)

            E_emission = NV['E_qe'] * input_parameters['e_e'].values[0] + NV['E_qr'] * input_parameters['e_r'].values[0]

            outcome_df = pd.DataFrame({'heuristic': [k], 'item': [m], 'parameter': [hue_param],
                                       'cost': [NV['cost']], 'opt_Se': [NV['se']],
                                       'E_qe': [NV['E_qe']], 'E_qr': [NV['E_qr']],
                                       'E_OH': [NV['E_OH']], 'E_SO': [NV['E_SO']],
                                       'E_emission': [E_emission]})

            self.policies_df = pd.concat([self.policies_df, outcome_df], ignore_index=True)
            self.policies_df.drop_duplicates(inplace=True)
            self.policies_df.reset_index(drop=True, inplace=True)

        self.all_policies_df = pd.concat([self.all_policies_df, outcome_df], ignore_index=True)
        self.policies_df.reset_index(drop=True, inplace=True)

        return outcome_df

    # решение задачи newsvendor
    def newsvendor(self, ce, cr, h, p, d_net_pmf, qr_pmf, qe_pmf):

        # Вычисление Critical Ratio (CR)
        CR = p / (p + h)

        # Копирование PMF чистого спроса и создание его кумулятивной функции
        D = d_net_pmf.copy()
        d_net_cdf = self.cdf_generator(D)

        # Нахождение точки перерасхода (Se) на кумулятивной функции
        Se_total = self.cdf_inv(CR, d_net_cdf)
        Se = Se_total["x"]
        Se_index = Se_total["index"]

        # Расчет ожидаемых значений количества регулярных и экспедитированных заказов
        E_qr = (qr_pmf[0] * qr_pmf[1]).sum()
        E_qe = (qe_pmf[0] * qe_pmf[1]).sum()

        # Выделение доменов перерасхода и избыточного запаса
        SO_domain = d_net_pmf[0, Se_index + 1:] - Se
        OH_domain = Se - d_net_pmf[0, :Se_index]

        # Расчет ожидаемых затрат на перерасход и избыточный запас
        E_SO = (SO_domain * d_net_pmf[1, Se_index + 1:]).sum()
        E_OH = (OH_domain * d_net_pmf[1, :Se_index]).sum()

        # Вычисление общей стоимости
        Cost = cr * E_qr + ce * E_qe + h * E_OH + p * E_SO

        # Возвращение результатов в виде словаря
        return {
            "cost": round(Cost, 10),
            "se": round(Se, 10),
            "E_qe": round(E_qe, 10),
            "E_qr": round(E_qr, 10),
            "E_OH": round(E_OH, 10),
            "E_SO": round(E_SO, 10)
        }

    # поиск минимум функции f на промежутке методом золотого сечения
    def golden_section_discret(self, f, bracket=(0, 1), tolerance=1, max_iter=1000):

        iteration_count = 0
        lower_bound = int(min(bracket))
        upper_bound = int(max(bracket) + 1)
        tolerance_threshold = tolerance
        golden_ratio = 1 - 0.618033988749895
        diff = upper_bound - lower_bound
        section_length = diff * golden_ratio

        left = round(lower_bound + section_length)
        right = round(upper_bound - section_length)
        left_value = f(left)
        right_value = f(right)

        while diff > tolerance_threshold and iteration_count < max_iter:
            if left_value < right_value:
                upper_bound = right
                right = left
                right_value = left_value
                diff = upper_bound - lower_bound
                left = round(lower_bound + diff * golden_ratio)
                left_value = f(left)
            else:
                lower_bound = left
                left = right
                left_value = right_value
                diff = upper_bound - lower_bound
                right = round(upper_bound - diff * golden_ratio)
                right_value = f(right)
            iteration_count += 1

        # Дополнительные итерации для точного определения минимума
        for _ in range(2):
            left_plus_1 = f(left + 1)
            left_minus_1 = f(left - 1)
            if left_value > left_plus_1:
                left += 1
                left_value = left_plus_1
                iteration_count += 1
            elif left_value > left_minus_1:
                left -= 1
                left_value = left_minus_1
                iteration_count += 1

        # Возвращение результатов в виде словаря
        return {
            "x_min": left,
            "y_min": left_value,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "tolerance": diff,
            "iteration_number": iteration_count
        }

    # расчет overshoot, expedited order quantity, regular order quantity по формулам из статьи.
    # на вход должен приходить параметр delta = S_r - S_e, с помощью которого можна посчитать количество товара для заказа
    def dual_sourcing_heuristic(self, parameter, demand_array, le, lr, last_period, empirical_d_net=False, dist_type=None, mean=None, cv=None ):
        D = demand_array.astype(int)
        LP = last_period
        l = lr - le

        O = np.zeros(LP + 2, dtype=int)
        qe = np.zeros(LP + 2, dtype=int)
        qr = np.zeros(LP + 2, dtype=int)

        for i in range(l - 1, LP + 1):
            O[i + 1] = max(O[i] + qr[i - l + 1] - D[i], 0)
            qe[i + 1] = max(D[i] - O[i] - qr[i - l + 1], 0)
            qr[i + 1] = max(parameter - O[i + 1] - qr[i - (l - 2):i + 1].sum(), 0)

        return {"overshoot_series": O, 'exp_order_series': qe, 'reg_order_series': qr}


    # вспомогательные функции для работы с распределениями и функциями вероятностей
    def dist_generator(self, x):
        """
        Этот метод преобразует одномерный массив  в
        функцию вероятности (PMF)
        """
        x_max = int(x.max())
        x_min = int(x.min())
        pmf = np.histogram(x, bins=x_max + 1 - x_min)[0]
        pmf = np.array([np.arange(x_min, x_max + 1), pmf / pmf.sum()])
        return pmf

    def cdf_generator(self, pmf):
        """
        Этот метод принимает на вход двумерный массив PMF и возвращает его функцию
        распределения (CDF).
        """
        cdf = pmf.copy()
        cdf[1] = np.cumsum(pmf[1])
        return cdf

    def cdf_inv(self, y, cdf):
        """
        Этот метод принимает вероятность и дискретное двумерное CDF и возвращает
        переменную с наиболее близкой накопленной вероятностью к запрошенной.
        """
        index_ = np.where(cdf[1] >= y)[0][0]
        return {"x": cdf[0, index_], "c_probability": cdf[1, index_], "index": index_}

    def convolve(self, X, Y, minus_operator=False):
        """
        функция свертки. нужна для генерации net demand, который расчитывается как раз, как свертка спроса в течении
        периодa l_e+1 (demand_le1) и перерасхода (overshoot)
        """

        # Определение функции flip, которая инвертирует массив
        def flip(x):
            return np.array([x[-(i + 1)] for i in range(len(x))])

        # Если оператор минус между X и Y, Y должен быть инвертирован и отражен
        if minus_operator == True:
            Y[1] = flip(Y[1])

        # Определение максимальных и минимальных значений для Y
        if minus_operator == False:
            max_y = Y[0].max()
            min_y = Y[0].min()
        else:
            max_y = -Y[0].min()
            min_y = -Y[0].max()

        # Определение максимальных и минимальных значений для X и Y
        dists_max = np.array([X[0].max(), max_y])
        dists_min = np.array([X[0].min(), min_y])

        # Свертка X и Y с использованием FFT
        D_convolved = np.array([np.arange(dists_min.sum(), dists_max.sum() + 1),
                                fftconvolve(X[1], Y[1])])

        return D_convolved







