# **Vehicle Routing Project**

This project is a vehicle routing optimization system developed in Python using various libraries such as Pandas, NumPy, and Pandasql. The primary goal of this project is optimization of vehicle routing for delivery logistics.

## Libraries Used:

pandas: Used for data manipulation and analysis, particularly for handling tabular data.
numpy: Utilized for numerical computations and operations, such as array manipulation and mathematical functions.
pandasql: Used for querying DataFrame objects using SQL syntax.
warnings: Used for handling warnings during execution.
time: Utilized for measuring the execution time of the code.

## Functions:

Several functions are defined to perform specific tasks related to vehicle routing optimization, such as nearest neighbor algorithm, creating databases, calculating delivery times, creating vehicle tours, and exporting data to Excel files.

- sorted_nodes_indexes: Creates new node indexes based on the nearest neighbor algorithm.
- expand_list_for_matrix: Counts and adds distances from zip code numbers to create distances from each zip code to all other zip code numbers.
- nearest_neighbor_algorithm: Generates the shortest path between nodes using the nearest neighbor algorithm.
- create_database_demand_in_nodes: Creates a database to calculate delivery time for each delivery.
- vehicles_on_deliveries: Creates deliveries and finds the number of vehicles required.
- create_vehicles_deliveries_output: Assigns each order to a vehicle.
- tours_of_vehicles: Creates input parameters for tours of vehicles.

## Data Loading and Manipulation:

Data is loaded from an Excel file containing distance matrices, vehicle categories, and demand information.
Various calculations and manipulations are performed on the data to optimize vehicle routing.

## Output:

The final output includes vehicle tours, delivery details, and vehicle categories.
The output data is exported to Excel files for further analysis and processing.
