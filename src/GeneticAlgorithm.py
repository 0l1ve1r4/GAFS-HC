from src.cross_validation import CrossValidation
from src.GeneticOperators import GeneticOperators
from src.Dataframe import Dataframe
from src.dataset import Dataset

import os


class GeneticAlgorithm(GeneticOperators):
    
    def __init__(self, population_size:int, num_generations:int, crossover_rate:float, 
                 mutation_rate:float, tournament_winner_rate:float, dataset_path:str, algorithm: str) -> None:
        
        super().__init__()
        self.dataset = Dataset(dataset_path) 
    
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_winner_rate = tournament_winner_rate
        self.algorithm = algorithm
        self.best_chromosome = ([], 0)      
        self.num_objects = len(self.dataset.dataset_objects) 
        self.num_attributes = len(self.dataset.dataset_attributes) - 1 
                
        train_files, test_files = self.get_paths(dataset_path)
        
        self.cv = CrossValidation(self.dataset_path, train_files, test_files) 
        self.dataset_default_fitness = self.cv.cross_validation_5_fold(train_files, test_files)
        
        self.utils.debug(f"Att.: {self.num_attributes} Obj.: {self.num_objects} | Default HF: {self.dataset_default_fitness:.3f}", "info")
    
    def get_paths(self, dataset_path: str) -> None:
        self.last_generation_path = 'generated-files/last_generation.txt'
        self.dataset_path = dataset_path
        self.path_without_extension = dataset_path.split(".")[0]       
        self.file_name = self.path_without_extension.split("/")[-1]  
        self.results_dir = f'results/{self.file_name}_Results'
        self.dataframe_path = f'{self.results_dir}/{self.file_name}_{self.algorithm}.csv'
        self.plot_save_path = f'{self.results_dir}/{self.file_name}_{self.algorithm}.png'
        self.df = Dataframe(self.dataframe_path, self.plot_save_path)
        
        os.makedirs(self.results_dir, exist_ok=True)
        test_files = [f"{self.path_without_extension}_test_fold_{i}.arff" for i in range(5)]
        train_files = [f"{self.path_without_extension}_train_fold_{i}.arff" for i in range(5)]
        
        return train_files, test_files
        
    def get_best_fitness(self, population_fitness: list, population_list: list, generation = 0) -> None:
        """Get the best fitness of the population."""
        
        best_fitness = float(self.best_chromosome[1])
        best_population = float(max(population_fitness))
        best_population_index = population_fitness.index(best_population)
        
        if best_population > best_fitness:
            self.best_chromosome = (population_list[best_population_index], best_population)
            
        with open(self.last_generation_path, "w+") as file:
            file.write(f'{str(generation)}\n')
            for chromosome in population_list:
                chromosome_i = [str(i) for i in chromosome]
                chromosome_i = ",".join(chromosome_i)
                file.write(f"{chromosome_i}\n")
            
    def show_results(self) -> None:
        """Show the results of the Genetic Algorithm in the terminal."""
        
        with open(self.last_generation_path, "w+") as file:
            file.write(f'@FINISHED')
            
        best_fitness = self.cv.save_best_chromosome(self.best_chromosome[0])
        porcentage_better = ((self.dataset_default_fitness - best_fitness) / self.dataset_default_fitness * 100) * - 1    
        selected_attributes = self.best_chromosome[0].count(1)
        
        self.utils.debug(f"Founded a solution {porcentage_better:.2f}% better with {selected_attributes} attributes. HF: {best_fitness:.2f}", type="success")

    def save_results_on_dataframe(self, generation: int, population_fitness: list, 
                                  population_list: list, generation_time: float, total_time: float) -> None:
        """Save the results of this generation on a dataframe."""
        
        self.df.save_results_on_dataframe(
            generation, population_fitness, population_list, 
            generation_time, total_time, self.best_chromosome
                                          )
       
    def save_report(self) -> None:
        """Save the results of the Genetic Algorithm on a plot."""
        
        self.df.save_report(self.dataset_default_fitness, self.num_attributes)
        