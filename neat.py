from pool import Pool
import logging

logging.basicConfig(level=logging.DEBUG)
pool = Pool(population=300)
pool.evolve(max_gen=100)
