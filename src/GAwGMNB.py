import time

from src.GeneticAlgorithm import * 

class GMNBwGA(GeneticAlgorithm):
    """Genetic Algorithm with Global Model Naive Bayes."""
    
    def __init__(self, dataset_path: str, population_size:int, num_generations:int, 
                 crossover_rate:float, mutation_rate:float, tournament_winner_rate:float) -> None:
        
        super().__init__(population_size, num_generations, crossover_rate, 
                         mutation_rate, tournament_winner_rate, dataset_path, 'GMNB')        
        
        self.utils.debug(f"CTRL + C to stop and plot early | Close the terminal to stop and save to resume later", "info")
        self.utils.debug(f"Starting Genetic Algorithm with Global Model Naive Bayes", "warning")
        
    def run(self):

        self.start_time = time.time()         
        population_list, generation = self.create_population(self.population_size, self.num_attributes, self.last_generation_path) 
        
        while generation < self.num_generations:            
            generation_time = time.time()
            
            try:
                population_fitness = self.cv.evaluate_population_parallel(population_list) 
                super().get_best_fitness(population_fitness, population_list, generation)

                population_list = self.tournament(population_list, population_fitness, k = self.tournament_winner_rate) 
                population_list = self.pmx_crossover(population_list, self.crossover_rate)
                population_list = self.swap_mutation(population_list, self.mutation_rate) 
                
                estimated_time = (time.time() - self.start_time) / (generation + 1) * (self.num_generations - generation)
                
                print(f"Gen: {generation + 1}/{self.num_generations} | Estimated time {estimated_time:.2f} secs", end= "\r")
                
            except KeyboardInterrupt:
                break
            
            generation_time = time.time() - generation_time
            total_time = time.time() - self.start_time
            
            super().save_results_on_dataframe(generation, population_fitness, population_list, generation_time, total_time)
            
            generation += 1

        super().show_results()
        super().save_report()
        







    