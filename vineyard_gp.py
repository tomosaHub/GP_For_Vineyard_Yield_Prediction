import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from deap import base, creator, tools, gp, algorithms
import random
import numpy as np
import operator
import matplotlib.pyplot as plt
from statistics import mean, stdev
import time

# Function for safe division
def safe_div(x, y):
    return x / y if y != 0 else 1

# Load and preprocess the data
def preprocess_data():
    print("Loading and preprocessing data...")
    # Load the dataset
    data = pd.read_csv("dataset_2178_vineyard.csv")
    
    # Check for missing values
    print("Missing values:", data.isnull().sum())
    
    # Standardize features
    scaler = StandardScaler()
    data[['lugs_1989', 'lugs_1990']] = scaler.fit_transform(data[['lugs_1989', 'lugs_1990']])
    
    # Separate features (X) and target (y)
    X = data[['lugs_1989', 'lugs_1990']]  # Features
    y = data['lugs_1991']                  # Target
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to numpy arrays for GP
    train_inputs = X_train.values
    train_targets = y_train.values
    test_inputs = X_test.values
    test_targets = y_test.values
    
    print("Training data shape:", train_inputs.shape, train_targets.shape)
    print("Testing data shape:", test_inputs.shape, test_targets.shape)
    
    return train_inputs, train_targets, test_inputs, test_targets, scaler

# Set up genetic programming components
def setup_gp(train_inputs, train_targets):
    # Define primitive set
    pset = gp.PrimitiveSetTyped("MAIN", [float, float], float)
    
    # Function set: arithmetic operators
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(safe_div, [float, float], float)
    
    # Terminal set: input variables and ephemeral random constants
    pset.addEphemeralConstant("rand", lambda: random.uniform(-10, 10), float)
    pset.renameArguments(ARG0="lugs_1989", ARG1="lugs_1990")
    
    # Fitness function
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    
    # Toolbox setup
    toolbox = base.Toolbox()
    
    # Tree initialization
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=5)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    # Evaluation function
    def eval_mse(individual, points, targets):
        try:
            func = toolbox.compile(expr=individual)
            
            # Calculate squared errors
            sqerrors = []
            for i in range(len(points)):
                try:
                    result = func(points[i][0], points[i][1])
                    sqerrors.append((result - targets[i])**2)
                except Exception:
                    # Penalize invalid expressions
                    return (1000.0,)
                    
            return (np.mean(sqerrors),)
            
        except Exception:
            # Penalize invalid expressions
            return (1000.0,)
    
    toolbox.register("evaluate", eval_mse, points=train_inputs, targets=train_targets)
    
    # Selection method: Tournament selection
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    # Genetic operators
    toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Tree size limitations
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    
    return toolbox, pset

# Run a single GP experiment
def run_gp_experiment(toolbox, ngen=50, pop_size=300, cxpb=0.9, mutpb=0.1, verbose=False):
    # Create initial population
    pop = toolbox.population(n=pop_size)
    
    # Hall of Fame to keep track of the best individual
    hof = tools.HallOfFame(1)
    
    # Statistics to track
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Run the evolutionary algorithm
    final_pop, logbook = algorithms.eaSimple(
        pop, 
        toolbox, 
        cxpb=cxpb, 
        mutpb=mutpb, 
        ngen=ngen, 
        stats=stats, 
        halloffame=hof, 
        verbose=verbose
    )
    
    return hof[0], logbook

# Evaluate the best individual on test data
def evaluate_on_test_data(individual, toolbox, test_inputs, test_targets):
    func = toolbox.compile(expr=individual)
    
    # Calculate predictions and errors
    predictions = []
    actual = []
    squared_errors = []
    
    for i in range(len(test_inputs)):
        try:
            pred = func(test_inputs[i][0], test_inputs[i][1])
            predictions.append(pred)
            actual.append(test_targets[i])
            squared_errors.append((pred - test_targets[i])**2)
        except Exception:
            print(f"Error evaluating test point {i}")
    
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    
    return predictions, actual, mse, rmse

# Run multiple experiments and collect statistics
def run_multiple_experiments(num_runs=10):
    # Load and preprocess data
    train_inputs, train_targets, test_inputs, test_targets, scaler = preprocess_data()
    
    # Parameter settings
    pop_size = 300
    ngen = 50
    cxpb = 0.9
    mutpb = 0.1
    
    # Lists to store results
    best_fitnesses = []
    test_mses = []
    best_individuals = []
    execution_times = []
    
    print(f"\nRunning {num_runs} experiments with parameters:")
    print(f"Population size: {pop_size}")
    print(f"Number of generations: {ngen}")
    print(f"Crossover probability: {cxpb}")
    print(f"Mutation probability: {mutpb}")
    print(f"Maximum tree depth: 17")
    
    for run in range(num_runs):
        print(f"\nRun {run+1}/{num_runs}")
        
        # Setup GP components
        toolbox, pset = setup_gp(train_inputs, train_targets)
        
        # Measure execution time
        start_time = time.time()
        
        # Run experiment
        best_ind, logbook = run_gp_experiment(
            toolbox, 
            ngen=ngen, 
            pop_size=pop_size, 
            cxpb=cxpb, 
            mutpb=mutpb,
            verbose=False
        )
        
        # Record execution time
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        
        # Save best individual and its fitness
        best_individuals.append(best_ind)
        best_fitnesses.append(best_ind.fitness.values[0])
        
        # Evaluate on test data
        _, _, test_mse, test_rmse = evaluate_on_test_data(best_ind, toolbox, test_inputs, test_targets)
        test_mses.append(test_mse)
        
        print(f"Best training MSE: {best_ind.fitness.values[0]:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Execution time: {execution_time:.2f} seconds")
        
        # Print the best expression
        print(f"Best expression: {best_ind}")
    
    # Calculate statistics
    avg_best_fitness = mean(best_fitnesses)
    std_best_fitness = stdev(best_fitnesses)
    best_run_idx = best_fitnesses.index(min(best_fitnesses))
    best_overall = best_individuals[best_run_idx]
    
    avg_test_mse = mean(test_mses)
    std_test_mse = stdev(test_mses)
    
    avg_time = mean(execution_times)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Average best training MSE: {avg_best_fitness:.4f}")
    print(f"Std Dev of best training MSE: {std_best_fitness:.4f}")
    print(f"Average test MSE: {avg_test_mse:.4f}")
    print(f"Std Dev of test MSE: {std_test_mse:.4f}")
    print(f"Average execution time: {avg_time:.2f} seconds")
    print(f"Best overall expression: {best_overall}")
    
    # Generate detailed report
    generate_report(
        best_fitnesses,
        test_mses,
        best_individuals,
        best_overall,
        avg_best_fitness,
        std_best_fitness,
        avg_test_mse,
        std_test_mse,
        pop_size,
        ngen,
        cxpb,
        mutpb
    )
    
    return best_overall, toolbox

# Generate a report with results
def generate_report(
    best_fitnesses,
    test_mses,
    best_individuals,
    best_overall,
    avg_best_fitness,
    std_best_fitness,
    avg_test_mse,
    std_test_mse,
    pop_size,
    ngen,
    cxpb,
    mutpb
):
    # Create result tables
    results_table = pd.DataFrame({
        'Run': range(1, len(best_fitnesses) + 1),
        'Training MSE': best_fitnesses,
        'Test MSE': test_mses
    })
    
    summary_table = pd.DataFrame({
        'Metric': ['Training MSE', 'Test MSE'],
        'Best': [min(best_fitnesses), min(test_mses)],
        'Average': [avg_best_fitness, avg_test_mse],
        'Std Dev': [std_best_fitness, std_test_mse]
    })
    
    # Save tables to CSV
    results_table.to_csv('gp_results_by_run.csv', index=False)
    summary_table.to_csv('gp_summary_statistics.csv', index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(results_table['Run'], results_table['Training MSE'], color='blue', alpha=0.7)
    plt.axhline(y=avg_best_fitness, color='red', linestyle='--', label=f'Average: {avg_best_fitness:.4f}')
    plt.title('Training MSE by Run')
    plt.xlabel('Run')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.bar(results_table['Run'], results_table['Test MSE'], color='green', alpha=0.7)
    plt.axhline(y=avg_test_mse, color='red', linestyle='--', label=f'Average: {avg_test_mse:.4f}')
    plt.title('Test MSE by Run')
    plt.xlabel('Run')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('gp_results_visualization.png')
    
    # Write best expression to file
    with open('best_expression.txt', 'w') as f:
        f.write(str(best_overall))
    
    print("\nReport generated. Files saved:")
    print("- gp_results_by_run.csv: Detailed results for each run")
    print("- gp_summary_statistics.csv: Summary statistics")
    print("- gp_results_visualization.png: Visualization of results")
    print("- best_expression.txt: Best expression found")

# Main function
def main():
    # Clear any previous creator definitions
    if hasattr(creator, 'FitnessMin'):
        del creator.FitnessMin
    if hasattr(creator, 'Individual'):
        del creator.Individual
    
    # Run multiple experiments
    best_overall, toolbox = run_multiple_experiments(num_runs=10)
    
    # Load data for final evaluation
    train_inputs, train_targets, test_inputs, test_targets, scaler = preprocess_data()
    
    # Final evaluation of best individual
    predictions, actual, mse, rmse = evaluate_on_test_data(best_overall, toolbox, test_inputs, test_targets)
    
    print("\n=== Final Evaluation of Best Individual ===")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(actual, predictions, alpha=0.7)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--')
    plt.xlabel('Actual Yield')
    plt.ylabel('Predicted Yield')
    plt.title('Predicted vs Actual Yields')
    plt.savefig('predictions_vs_actual.png')
    print("Saved predictions_vs_actual.png")

if __name__ == "__main__":
    main()
