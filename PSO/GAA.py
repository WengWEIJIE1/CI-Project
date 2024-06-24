import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 从 CSV 文件读取投资选项并手动标准化
def load_investment_options(csv_file):
    df = pd.read_csv(csv_file)
    
    # 填充空值为 0
    df['deal_amount'] = df['deal_amount'].fillna(0)
    df['ask_equity'] = df['ask_equity'].fillna(0)
    
    # 提取投资相关数据
    investment_options = df[['brand_name', 'deal_amount', 'ask_equity']].copy()
    
    # 手动标准化：将 deal_amount 和 ask_equity 归一化到 [0, 1] 范围
    deal_amount_max = investment_options['deal_amount'].max()
    deal_amount_min = investment_options['deal_amount'].min()
    ask_equity_max = investment_options['ask_equity'].max()
    ask_equity_min = investment_options['ask_equity'].min()
    
    investment_options['deal_amount'] = (investment_options['deal_amount'] - deal_amount_min) / (deal_amount_max - deal_amount_min)
    investment_options['ask_equity'] = (investment_options['ask_equity'] - ask_equity_min) / (ask_equity_max - ask_equity_min)
    
    return investment_options.to_dict('records')

# 初始化种群
def create_population(size, num_investments):
    population = []
    for _ in range(size):
        individual = [random.uniform(0, 1) for _ in range(num_investments)]
        total = sum(individual)
        individual = [x / total for x in individual]  # 确保投资比例之和为 1
        population.append(individual)
    return population

# 计算适应度
def compute_fitness(individual):
    total_return = sum(individual[i] * investment_options[i]['deal_amount'] for i in range(len(individual)))
    total_risk = np.sqrt(sum((individual[i] ** 2) * investment_options[i]['ask_equity'] ** 2 for i in range(len(individual))))
    
    if total_risk > max_risk:
        return 0  # 超过风险阈值，适应度为 0
    return total_return

# 选择
def selection(population, fitness_values):
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        probabilities = [1 / len(fitness_values)] * len(fitness_values)
    else:
        probabilities = [f / total_fitness for f in fitness_values]
    selected_indices = np.random.choice(len(population), len(population), p=probabilities)
    selected_population = [population[i] for i in selected_indices]
    
    if len(selected_population) % 2 != 0:
        selected_population.append(random.choice(selected_population))
    
    return selected_population

# 交叉
def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
    else:
        child1, child2 = parent1, parent2
    return child1, child2

# 变异
def mutate(individual, mutation_rate=0.05):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(0, 1)
    total = sum(individual)
    individual = [i / total for i in individual]  # 归一化为投资比例
    return individual

# 主遗传算法
def genetic_algorithm(pop_size, generations, crossover_rate=0.8, mutation_rate=0.05):
    num_investments = len(investment_options)
    population = create_population(pop_size, num_investments)
    best_solution = max(population, key=compute_fitness)
    best_fitness = compute_fitness(best_solution)
    
    best_solutions = []
    avg_fitnesses = []
    best_individuals = []
    
    for generation in range(generations):
        fitness_values = [compute_fitness(ind) for ind in population]
        best_current_solution = max(population, key=compute_fitness)
        best_current_fitness = compute_fitness(best_current_solution)
        avg_fitness = np.mean(fitness_values)
        
        best_solutions.append(best_current_fitness)
        avg_fitnesses.append(avg_fitness)
        best_individuals.append(best_current_solution)
        
        if best_current_fitness > best_fitness:
            best_solution = best_current_solution
            best_fitness = best_current_fitness
        
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness:.4f}")
        
        selected_population = selection(population, fitness_values)
        
        next_generation = []
        for i in range(0, pop_size, 2):
            parent1, parent2 = selected_population[i], selected_population[i + 1]
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            next_generation.extend([child1, child2])
        
        population = [mutate(ind, mutation_rate) for ind in next_generation]
    
    return best_solution, best_solutions, avg_fitnesses, best_individuals

# 参数设置
population_size = 50  # 增加种群大小
generations = 100  # 增加代数
crossover_rate = 0.8
mutation_rate = 0.05
max_risk = 1.0  # 标准化后风险阈值

# CSV 文件路径
csv_file = 'C:/Weng-CI-Python/PSO/Shark Tank India Dataset.csv'  # 根据实际情况修改路径
investment_options = load_investment_options(csv_file)

# 运行遗传算法
best_solution, best_solutions, avg_fitnesses, best_individuals = genetic_algorithm(
    population_size, generations, crossover_rate, mutation_rate
)

# 绘制图表
generations_range = range(1, generations + 1)

plt.figure(figsize=(14, 7))

# 绘制最佳解的变化
plt.subplot(1, 2, 1)
plt.plot(generations_range, best_solutions, marker='o', linestyle='-', color='b')
plt.title('Best Fitness per Generation', fontsize=14)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Best Fitness', fontsize=12)
plt.grid(True)

# 绘制平均适应度的变化
plt.subplot(1, 2, 2)
plt.plot(generations_range, avg_fitnesses, marker='x', linestyle='-', color='g')
plt.title('Average Fitness per Generation', fontsize=14)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Average Fitness', fontsize=12)
plt.grid(True)

plt.tight_layout()

# 创建 Tkinter 窗口
root = Tk()
root.title('Genetic Algorithm Optimization')
root.geometry('1000x600')

# 添加 Matplotlib 图表到 Tkinter 窗口
canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

# 输出最佳解
best_portfolio_fig = plt.figure(figsize=(10, 10))
investment_names = [option['brand_name'] for option in investment_options]
proportions = [proportion * 100 for proportion in best_solution]
plt.barh(investment_names, proportions, color='blue')
plt.title('Best Investment Portfolio Proportions', fontsize=14)
plt.xlabel('Proportion (%)', fontsize=12)
plt.ylabel('Investment Options', fontsize=12)
plt.gca().invert_yaxis()  # 反转 y 轴，使得公司名称能够完整显示
plt.tight_layout()

# 添加最佳投资组合图表到 Tkinter 窗口
canvas_portfolio = FigureCanvasTkAgg(best_portfolio_fig, master=root)
canvas_portfolio.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=1)

plt.show()

# 主 Tkinter 循环
root.mainloop()
