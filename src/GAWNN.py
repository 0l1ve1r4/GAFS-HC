from src.GeneticAlgorithm import *
from src.neural_network import NeuralNetwork
from sklearn.linear_model import LinearRegression

import numpy as np
import multiprocessing
import time

class GAWNN(GeneticAlgorithm):
    """Genetic Algorithm with Neural Networks."""
    
    def __init__(self, dataset_path: str, population_size:int, num_generations:int, crossover_rate:float, 
                 mutation_rate:float, tournament_winner_rate:float, save_model = False, 
                 save_path = "", load_model = False, load_path = "") -> None:

        super().__init__(population_size, num_generations, crossover_rate, mutation_rate, 
                         tournament_winner_rate, dataset_path, 'GAwNN')
        
        self.database_lenght, self.train_epochs = self.calculate_parameters(self.num_attributes) 
        self.database_path = "generated-files/training_data.txt"
        self.neural_network = NeuralNetwork(self.num_attributes)
        self.save_model_path = save_path
        self.save_model = save_model                           
        self.load_model = load_model         
        self.load_path = load_path
        
        self.model_trained = self.load_model 
        
        self.utils.debug(f"CTRL + C to stop and plot early", "info")
        self.utils.debug(f"Starting Genetic Algorithm with Neural Networks", "warning")

    def get_model(self) -> int:
        """Load or train the neural network model. If no model is loaded, train a new one."""
        
        if self.load_model:
            self.neural_network.load_nn(self.load_path)
            self.utils.debug(f"Model loaded from {self.load_path}", type="warning")
        
        elif not self.model_trained and self.save_model:
            self.utils.debug(f"Model will be saved at {self.save_model_path}", type="warning")
            self.utils.debug(f"Model will be trained for {self.train_epochs} epochs", type="info")
            self.utils.debug(f"Database lenght: {self.database_lenght}", type="info")
            
            self.create_database()            
            self.neural_network.train_nn(self.database_path, self.train_epochs)  
            self.neural_network.save_nn(self.save_model_path)
            
        return 0  
            
    def create_database(self) -> None:        
        num_threads = multiprocessing.cpu_count()
        generated_objects = 0
        
        already_generated = 0
        with open(self.database_path, "r") as file:
            for _ in file:
                already_generated += 1
                
        generated_objects = already_generated
        if already_generated > 0:
            self.utils.debug(f"Database already has {already_generated} objects", "info")
        
        while generated_objects < self.database_lenght:
            generated_objects += num_threads
            
            population, _ = self.create_population(num_threads, self.num_attributes)
            fitness = self.cv.evaluate_population_parallel(population)

            with open(self.database_path, "a+") as file:
                for population, fitness in zip(population, fitness):
                    file.write(f"{population},{fitness}\n")
            
            print(f"[Database Creation Progress]: {generated_objects}/{self.database_lenght}", end="\r")

    def calculate_parameters(self, num_attributes: int) -> tuple:
        "Return the number of epochs and the database lenght for the Neural Network."
        
        dataset_attributes = np.array([27,   52,   63,   77,   79,   80,   173,  551]).reshape((-1, 1))
        database_lenght = np.array(   [1598, 3235, 3284, 3475, 3502, 2543, 4499, 8713]).reshape((-1, 1))
        trainings_epochs = np.array(  [300,  416, 579, 541, 607, 740, 1209, 2183]).reshape((-1, 1))
        
        new_dataset_attributes = np.array([num_attributes]).reshape((-1, 1))
        predicted_database_lenght = int(LinearRegression().fit(dataset_attributes, database_lenght).predict(new_dataset_attributes))
        predicted_trainings_epochs = int(LinearRegression().fit(dataset_attributes, trainings_epochs).predict(new_dataset_attributes))
        

        return predicted_database_lenght, predicted_trainings_epochs

    def run(self):
        self.start_time = time.time() 
        
        self.get_model()
        
        population_list, generation = self.create_population(self.population_size, self.num_attributes) 
         
        while generation < self.num_generations:    
            generation_time = time.time()
            
            try:    
                population_fitness = self.neural_network.evaluate_population(population_list)
                super().get_best_fitness(population_fitness, population_list, generation)
                 
                population_list = self.tournament(population_list, population_fitness, k = self.tournament_winner_rate) 
                population_list = self.pmx_crossover(population_list, self.crossover_rate) 
                population_list = self.swap_mutation(population_list, self.mutation_rate) #
                    
                estimated_time = (time.time() - self.start_time) / (generation + 1) * (self.num_generations - generation)
                print(f"Gen: {generation + 1}/{self.num_generations} | Estimated time {estimated_time:.2f} secs.", end= "\r")
                
            except KeyboardInterrupt:
                break   

            generation_time = time.time() - generation_time
            total_time = time.time() - self.start_time
            
            super().save_results_on_dataframe(generation, population_fitness, population_list, generation_time, total_time)

            generation += 1
        
        super().show_results()
        super().save_report()
        self.bestFitnessFoundByGMNB()

    def bestFitnessFoundByGMNB(self):
        train_files = [] #data/best_chromosome/test_0.arff
        test_files = []
        
        for i in range(0, 5):
            train_files.append(f'data/best_chromosome/train_{i}.arff')
            test_files.append(f'data/best_chromosome/test_{i}.arff')
        
        fitness = self.cv.cross_validation_5_fold(train_files, test_files)
        fitness = f"Fitness found by NN evaluated by GMNB: {fitness}"
        
        self.utils.debug(fitness, "info")
