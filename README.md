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

## Input:
Input data is imported from excel using pandas. Below is a pictorial description of the input data.

![image](https://github.com/savavukmirovic/Vehicle-Routing-Project/assets/126354345/9cd2c445-46cb-451a-9310-927e1b8b67b7)

![image](https://github.com/savavukmirovic/Vehicle-Routing-Project/assets/126354345/5d23428f-dac4-41aa-a173-de43b77f66f8)

![image](https://github.com/savavukmirovic/Vehicle-Routing-Project/assets/126354345/48009acb-ea71-4583-8b76-de9d026c3e4b)

![image](https://github.com/savavukmirovic/Vehicle-Routing-Project/assets/126354345/9aebdba9-264d-467c-acaa-63279d6c6508)

## Output:

The final output includes vehicle tours, delivery details, and vehicle categories.
The output data is exported to Excel files for further analysis and processing.
Below is a pictorial description of the output data.

![image](https://github.com/savavukmirovic/Vehicle-Routing-Project/assets/126354345/fe1dee78-ad8f-4223-99a5-a6a8c64d150d)
