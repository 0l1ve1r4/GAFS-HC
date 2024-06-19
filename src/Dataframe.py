import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Dataframe():
    
    def __init__(self, dataframe_path: str, plot_save_path:str) -> None:
        """Class to save the results on a dataframe and plot the results."""
        
        self.plot_save_path = plot_save_path
        self.dataframe_path = dataframe_path
        self.columns = ['Generation', 'Avarage Fitness', 'Best Fitness', 'Time', 'Avarage Attributes', 'Total Time', 'Best Chromosome']
  
    def open_dataframe(self) -> pd.DataFrame:
        """Open the dataframe or create a new one."""
        
        try:
            dataframe = pd.read_csv(self.dataframe_path)
    
        except FileNotFoundError:
            dataframe = pd.DataFrame(columns=self.columns)
            dataframe.to_csv(self.dataframe_path, index=False)
        
        return dataframe

    def save_results_on_dataframe(self, generation: int, population_fitness: list, population_list:list,
                                  generation_time: float, total_time: float, best_chromosome_params: tuple) -> None:
        """Save the results on a dataframe."""
                
        avarage_fitness = float(f"{(sum(population_fitness) / len(population_fitness)):.2f}")
        best_fitness = float(f"{best_chromosome_params[1]:.2f}")

        best_chromosome = ",".join([str(i) for i in best_chromosome_params[0]])
        avarage_attributes = sum([chromosome.count(1) for chromosome in population_list]) / len(population_list)
        params = [generation, avarage_fitness, best_fitness, generation_time, avarage_attributes, total_time, best_chromosome]

        new_row = pd.DataFrame([params], columns=self.columns)
        
        dataframe = self.open_dataframe()
        dataframe = pd.concat([dataframe.dropna(), new_row], ignore_index=True)
        dataframe.to_csv(self.dataframe_path, index=False)
       
    def save_report(self, dataset_fitness: float, dataset_num_attributes: int) -> None:
        """Save the dataframe infomations on a plot."""

        dataframe = self.open_dataframe()
        average_fitness = dataframe['Avarage Fitness'].tolist()
        best_fitness = dataframe['Best Fitness'].tolist()

        fig, ax = plt.subplots(figsize=(15, 7))

        
        for y_values, label in zip([average_fitness, best_fitness], ['Average Fitness', 'Best Fitness']):
            ax.plot(range(len(y_values)), y_values, marker='o', label=label)
    
        ax.set_xlabel('Generations')
        ax.set_ylabel('hF / Num Attributes')
        ax.set_title('Report')

        ax.grid(True)
        ax.legend(loc='upper right')
        
        fig.savefig(f'{self.plot_save_path}')

        plt.close(fig)
        
        
    def plot_dataframes(self, paths: list, title: str) -> None:
        """Open each dataframe, normalize its data, and plot its information."""

        fig, ax = plt.subplots(figsize=(15, 7))
        scaler = MinMaxScaler()

        for path in paths:
            try:
                dataframe = pd.read_csv(path)
                dataframe['Avarage Fitness'] = scaler.fit_transform(dataframe[['Avarage Fitness']])
                average_fitness = dataframe['Avarage Fitness'].tolist()

                path_name = path.split('/')[-2]
                path_name = path_name.split('_')[0]
                
            except FileNotFoundError:
                print(f"File {path} not found.")
                continue
            
            ax.plot(range(len(average_fitness)), average_fitness, label=f'{path_name}')  # Removed marker='o'

        ax.set_xlabel('Gerações')
        ax.set_ylabel('Aptidão normalizada da média aritmética de 10 gerações')
        ax.set_title(title)

        ax.grid(True)
        ax.legend(loc='lower right')
        

        plt.show()