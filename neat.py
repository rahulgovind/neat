from pool import Pool
import logging

logging.basicConfig(level=logging.DEBUG)
pool = Pool(population=200)
pool.evolve(max_gen=100)

pool.draw_final_network()
#pool.plot_fitness(lambda x: x**0.5/16*100, ylabel='Accuracy (%)')