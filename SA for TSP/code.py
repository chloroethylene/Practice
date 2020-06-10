import numpy as np
import random
import matplotlib.pyplot as plt

all_cities = {'Shenyang': [123.429092,41.796768],
            'Changchun': [125.324501,43.886841],
            'Harbin': [126.642464,45.756966],
            'Beijing': [116.405289,39.904987],
            'Tianjin': [117.190186,39.125595],
            'Hohhot': [111.751990,40.841490],
            'Yinchuan': [106.232480,38.486440],
            'Taiyuan': [112.549248,37.857014],
            'Shijiazhuang': [114.502464,38.045475],
            'Jinan': [117.000923,36.675808],
            'Zhengzhou': [113.665413,34.757977],
            'Xian': [108.948021,34.263161],
            'Wuhan': [114.298569,30.584354],
            'Nanjing': [118.76741,32.041546],
            'Hefei': [117.283043,31.861191],
            'Shanghai': [121.472641,31.231707],
            'Changsha': [112.982277,28.19409],
            'Nanchang': [115.892151,28.676493],
            'Hangzhou': [120.15358,30.287458],
            'Fuzhou': [119.306236,26.075302],
            'Guangzhou': [113.28064,23.125177],
            'Taipei': [121.5200760,25.0307240],
            'Haikou': [110.199890,20.044220],
            'Nanning': [108.320007,22.82402],
            'Chongqing': [106.504959,29.533155],
            'Kunming': [102.71225,25.040609],
            'Guizhou': [106.713478,26.578342],
            'Chengdu': [104.065735,30.659462],
            'Lanzhou': [103.834170,36.061380],
            'Xining': [101.777820,36.617290],
            'Lhasa': [91.11450,29.644150],
            'Urumqi': [87.616880,43.826630],
            'Hongkong': [114.165460,22.275340],
            'Macao': [113.549130,22.198750]}
dist = {}
for city_1 in all_cities:
    for city_2 in all_cities:
        if city_1 == city_2:
            continue
        longi_1 = all_cities[city_1][0]*np.pi/180
        lati_1 = all_cities[city_1][1]*np.pi/180
        longi_2 = all_cities[city_2][0]*np.pi/180
        lati_2 = all_cities[city_2][1]*np.pi/180
        a1 = np.array([np.cos(lati_1),0,np.sin(lati_1)])
        a2 = np.array([np.cos(lati_2)*np.cos(abs(longi_1-longi_2)),np.cos(lati_2)*np.sin(abs(longi_1-longi_2)),np.sin(lati_2)])
        dist[(city_1,city_2)] = np.arccos(np.dot(a1,a2))*6371

def generate_random_agent():
    """
        生成单个随机路线
    """
    new_random_agent = list(all_cities.keys())
    random.shuffle(new_random_agent)
    return tuple(new_random_agent)

def generate_random_population(pop_size):
    """
        生成pop_size个随机路线
    """
    
    random_population = []
    for agent in range(pop_size):
        random_population.append(generate_random_agent())
    return random_population

def compute_fitness(solution):
    """
        计算个体适应度
        返回当前路线的总距离
        遗传算法会更倾向于选择距离较短的路线
    """
    
    solution_fitness = 0.0
    
    for index in range(len(solution)):
        city_1 = solution[index - 1]
        city_2 = solution[index]
        solution_fitness += dist[(city_1, city_2)]
        
    return solution_fitness

def mutate_agent(agent_genome, max_mutations=3):
    """
        对路线产生1至max_mutations个点的突变
    """
    
    agent_genome = list(agent_genome)
    num_mutations = random.randint(1, max_mutations)
    
    for mutation in range(num_mutations):
        swap_index1 = random.randint(0, len(agent_genome) - 1)
        swap_index2 = swap_index1

        while swap_index1 == swap_index2:
            swap_index2 = random.randint(0, len(agent_genome) - 1)

        agent_genome[swap_index1], agent_genome[swap_index2] = agent_genome[swap_index2], agent_genome[swap_index1]
            
    return tuple(agent_genome)

def shuffle_mutation(agent_genome):
    """
        对路线进行一次打乱操作
    """
    
    agent_genome = list(agent_genome)
    
    start_index = random.randint(0, len(agent_genome) - 1)
    length = random.randint(2, 20)
    
    genome_subset = agent_genome[start_index:start_index + length]
    agent_genome = agent_genome[:start_index] + agent_genome[start_index + length:]
    
    insert_index = random.randint(0, len(agent_genome) + len(genome_subset) - 1)
    agent_genome = agent_genome[:insert_index] + genome_subset + agent_genome[insert_index:]
    
    return tuple(agent_genome)
    
def run_genetic_algorithm(generations=5000, population_size=100):
    """
        遗传算法
    """
    
    population_subset_size = int(population_size / 10.)
    generations_10pct = int(generations / 5.)
    
    # 生成大小为population_size的随机初始种群
    population = generate_random_population(population_size)

    # 根据generations参数开始迭代
    for generation in range(generations):
        
        # 计算种群适应度
        population_fitness = {}

        for agent_genome in population:
            if agent_genome in population_fitness:
                continue

            population_fitness[agent_genome] = compute_fitness(agent_genome)

        # 取路线距离最短的10% and produce offspring each from them
        new_population = []
        for rank, agent_genome in enumerate(sorted(population_fitness,
                                                   key=population_fitness.get)[:population_subset_size]):
            
            if (generation % generations_10pct == 0 or generation == generations - 1) and rank == 0:
                print("Generation %d best: %d | Unique genomes: %d" % (generation,
                                                                       population_fitness[agent_genome],
                                                                       len(population_fitness)))
                print(agent_genome)
                print("")
                
                if generation == generations - 1:
                    return agent_genome

            # 将最优的路线直接复制到下一代中，占10%
            new_population.append(agent_genome)

            # 采用1到3点突变产生20%的后代
            for offspring in range(2):
                new_population.append(mutate_agent(agent_genome, 3))
                
            # 采用打乱突变产生70%的后代
            for offspring in range(7):
                new_population.append(shuffle_mutation(agent_genome))

        # 用新产生的种群代替旧种群
        population = new_population


best_route = list(run_genetic_algorithm(generations=5000, population_size=200))
best_route.append(best_route[0])
longitude = []
latitude = []
for city in best_route:
    longitude.append(all_cities[city][0])
    latitude.append(all_cities[city][1])
plt.plot(longitude,latitude,'o-')
plt.show()