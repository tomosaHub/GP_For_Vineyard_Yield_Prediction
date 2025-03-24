# GP-For-Vineyard-Yield-Prediction
COS 710 : Artificial Intelligence | Assignment 1 - Regression 

M.T Letswalo 14****70
# Vineyard Yield Prediction using Genetic Programming

This project implements a genetic programming approach to predict vineyard yields for 1991 based on data from 1989 and 1990.

## Requirements

### Environment Setup with Conda
1. Create a new conda environment:
   ```
   conda create -n vineyard_gp python
   ```

2. Activate the environment:
   ```
   conda activate vineyard_gp
   ```

3. Install required dependencies:
   ```
   conda install pandas numpy scikit-learn matplotlib
   conda install -c conda-forge deap
   ```

### Dataset
The program requires the vineyard dataset file:
- `dataset_2178_vineyard.csv` (Should be placed in the same directory as the script)

You can download the original dataset from:
https://github.com/EpistasisLab/pmlb/tree/master/datasets/192_vineyard

## Running the Code

### Basic Execution
1. Ensure your conda environment is activated:
   ```
   conda activate vineyard_gp
   ```
   
2. Run the main script:
   ```
   python vineyard_gp.py
   ```
   
3. The program will:
   - Preprocess the data
   - Run 10 genetic programming experiments
   - Generate result statistics
   - Create visualizations
   - Save results to files

### Output Files
After running the program, the following files will be generated:
- `gp_results_by_run.csv`: Detailed results for each run
- `gp_summary_statistics.csv`: Summary statistics across all runs
- `gp_results_visualization.png`: Bar chart visualization of results
- `predictions_vs_actual.png`: Scatter plot of predicted vs. actual yields
- `best_expression.txt`: The best mathematical expression found

## Modifying Parameters

To modify the genetic programming parameters, open the script and locate the `run_multiple_experiments` function. The key parameters are:
- `pop_size`: Population size (default: 300)
- `ngen`: Number of generations (default: 50)
- `cxpb`: Crossover probability (default: 0.9)
- `mutpb`: Mutation probability (default: 0.1)

To change the number of runs, modify the `num_runs` parameter in the `main` function.

## Customizing the GP Components

### Function Set
The function set can be modified in the `setup_gp` function. To add new functions:
```python
pset.addPrimitive(new_function, [float, float], float)
```

### Fitness Function
The fitness function is defined within the `eval_mse` function. You can modify this to use different error metrics.

### Selection and Genetic Operators
Selection and genetic operators are defined in the `setup_gp` function:
- Selection: `toolbox.register("select", tools.selTournament, tournsize=3)`
- Crossover: `toolbox.register("mate", gp.cxOnePointLeafBiased, termpb=0.1)`
- Mutation: `toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)`

## Troubleshooting

### Common Issues
1. **Missing dataset**: Ensure the dataset file is in the same directory as the script
2. **Environment issues**: Make sure the conda environment is properly activated
3. **Package conflicts**: If you encounter package conflicts, try creating a fresh conda environment
4. **Memory errors**: Reduce population size or number of generations if encountering memory issues

### Conda Environment Export
You can export the exact environment configuration to share with others:
```
conda env export > vineyard_gp_environment.yml
```

Others can recreate your environment with:
```
conda env create -f vineyard_gp_environment.yml
```

### Error Handling
The code includes error handling for:
- Invalid expressions during evaluation
- Tree compilation errors

If you encounter persistent errors, try reducing the complexity of the genetic programming setup (smaller population, fewer generations, lower maximum tree depth).

## Reporting
Follow this link to view the reporting and conclusion of the current findings: https://www.overleaf.com/read/gxzcqpxjmsdx#f3fc0e

## Contact

For questions or issues, please submit an issue on the project repository or contact letswalomosa@tuks.co.za.