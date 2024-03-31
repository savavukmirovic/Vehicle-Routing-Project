import pandas as pd
import numpy as np
from pandasql import sqldf
import warnings
import time
warnings.filterwarnings("ignore")
start = time.time()
# TODO: Gigantic Route


def sorted_nodes_indexes(date_g_route: list, nodes_demand_by_date: pd.DataFrame) -> dict:
    """This function is creating new node indexes based on nearest neighbour algorithm"""
    demand_by_date = sorted(nodes_demand_by_date['PrimaocPTT'].tolist())
    new_g_route = {}
    for new_data in range(len(date_g_route)):
        new_g_route[date_g_route[new_data]] = demand_by_date[date_g_route[new_data] - 1]
    new_key_count = 0
    new_changed_g_route = {}
    for sort_data in new_g_route.values():
        new_key_count += 1
        new_changed_g_route[new_key_count] = sort_data
    return new_changed_g_route


def expand_list_for_matrix(matrix_list: list, date_demand: pd.DataFrame) -> list:
    """This function counts and adds as many distances from zipcode numbers as there are zipcode numbers in the database
    for that specific day and creates distances from each zipcode to all other zipcode numbers"""
    demand = date_demand['PrimaocPTT'].tolist()
    demand_unique_zipcode = sorted(date_demand['PrimaocPTT'].unique().tolist())
    new_matrix_list = []
    for data in range(len(matrix_list)):
        new_matrix_list += [matrix_list[data]] * demand.count(demand_unique_zipcode[data])
    return new_matrix_list


def nearest_neighbor_algorithm(distance: dict, date_demand: pd.DataFrame) -> list:
    """This function is used to generate the shortest path between nodes using the nearest neighbor algorithm"""
    distance_length = len(distance)
    demand_for_calculating_nodes = {key + 1: value
                                    for (key, value) in enumerate(sorted(date_demand['PrimaocPTT'].tolist()))}
    route = []
    # If we have more than one node in route (we are adding nodes by zipcode number,
    # so we always do one by one zipcode and then nearest node algorithm
    if list(demand_for_calculating_nodes.values()).count(demand_for_calculating_nodes[depot]) > 1:
        for demand_index in demand_for_calculating_nodes:
            if demand_for_calculating_nodes[demand_index] == demand_for_calculating_nodes[depot]:
                nearest_node = demand_index
                route.append(nearest_node)
                for depot_distance in range(1, distance_length + 1):
                    distance[depot_distance][nearest_node - 1] = np.inf
    else:
        route.append(depot)
        for depot_distance in range(1, distance_length + 1):
            distance[depot_distance][depot - 1] = np.inf
    nearest_node = depot
    for node_distance in range(distance_length - len(route)):
        nearest_node = distance[nearest_node].index(min(distance[nearest_node])) + 1
        if list(demand_for_calculating_nodes.values()).count(demand_for_calculating_nodes[nearest_node]) > 1 \
                and nearest_node not in route:
            for demand_index in demand_for_calculating_nodes:
                if demand_for_calculating_nodes[demand_index] == demand_for_calculating_nodes[nearest_node]:
                    nearest_node = demand_index
                    route.append(nearest_node)
                    for depot_distance in range(1, distance_length + 1):
                        distance[depot_distance][nearest_node - 1] = np.inf
        elif nearest_node not in route:
            route.append(nearest_node)
            for nearest_node_distance in range(1, distance_length + 1):
                distance[nearest_node_distance][nearest_node - 1] = np.inf
    return route


def create_database_demand_in_nodes(demand_in_n: pd.DataFrame, distance_m: pd.DataFrame) -> pd.DataFrame:
    """Create database to calculate delivery time for each delivery"""
    # First put the time at the unloading point in seconds
    demand_in_n.iloc[:, 14] = demand_in_n.iloc[:, 14].div(60)

    # Create connection to find time from depot zipcode to delivery place zipcode
    demand_in_n['DistanceDemandConnection'] = demand_in_n['CentarPTT'].astype(str).str.cat(
        demand_in_n['PrimaocPTT'].astype(str), sep='_')
    distance_m['DistanceDemandConnection'] = distance_m['PTTOD'].astype(str).str.cat(
        distance_m['PTTDO'].astype(str), sep='_')
    # Connect time from depot zipcode to delivery place zipcode for each delivery
    demand_in_n = pd.merge(demand_in_n, distance_m, on='DistanceDemandConnection', how='left')
    # Create new dataframe only with necessary data
    demand_in_n = demand_in_n.drop(columns=['DistanceDemandConnection', 'PTTOD', 'PTTDO', 'BrzinaKMH'], axis=1)
    demand_in_n = demand_in_n.rename(columns={'VremeH': 'VremeOdCentraDoPTT',
                                              'RastojanjeKm': 'RastojanjeOdCentraDoPTT'})
    # Create connection to find time from delivery place zipcode to delivery place zipcode
    recipient_zipcodes = demand_in_n['PrimaocPTT'].to_list()
    demand_in_n['DistanceDemandConnection'] = [f'{recipient_zipcodes[zipcode_num]}_'
                                               f'{recipient_zipcodes[zipcode_num - 1]}'
                                               for zipcode_num in range(len(recipient_zipcodes))]

    # Connect time from delivery place zipcode to delivery place zipcode for each delivery
    demand_in_n = pd.merge(demand_in_n, distance_m, on='DistanceDemandConnection', how='left')
    # Create new dataframe only with necessary data
    demand_in_n = demand_in_n.drop(columns=['DistanceDemandConnection', 'PTTOD', 'PTTDO', 'BrzinaKMH'], axis=1)
    demand_in_n = demand_in_n.rename(columns={'VremeH': 'VremeOdPTTDoPTT',
                                              'RastojanjeKm': 'RastojanjeOdPTTDoPTT'})
    # Create connection to find time from delivery place zipcode to depot zipcode
    demand_in_n['DistanceDemandConnection'] = demand_in_n['PrimaocPTT'].astype(str).str.cat(
        demand_in_n['CentarPTT'].astype(str), sep='_')

    # Connect time from depot zipcode to delivery place zipcode for each delivery
    demand_in_n = pd.merge(demand_in_n, distance_m, on='DistanceDemandConnection', how='left')
    # Create new dataframe only with necessary data
    demand_in_n = demand_in_n.drop(columns=['DistanceDemandConnection', 'PTTOD', 'PTTDO', 'BrzinaKMH'], axis=1)
    demand_in_n = demand_in_n.rename(columns={'VremeH': 'VremeOdPTTDoCentra',
                                              'RastojanjeKm': 'RastojanjeOdPTTDoCentra',
                                              'VremeZadrzavanjaNaIstovarnomMestuMINUTA': 'VremeIstovara'})
    return demand_in_n


def vehicles_on_deliveries(date_demand_in_n: pd.DataFrame,
                           vehicle_capacity: int | float | np.ndarray,
                           driver_wh: int | float | np.ndarray, del_depot: int = 0) -> list:
    """Create deliveries and find number of vehicles"""
    # Create global variable to assign vehicles to orders
    global vehicles
    # Calculate time and weight for each order to check if the order is on delivery
    time_on_delivery = [date_demand_in_n.values[del_depot, 17], date_demand_in_n.values[del_depot, 14]]
    weight_on_delivery = [date_demand_in_n.values[del_depot, 12]]
    for zipcode_num in range(del_depot + 1, len(date_demand_in_n)):
        time_on_orders = date_demand_in_n.values[zipcode_num, 14] + date_demand_in_n.values[zipcode_num, 19]
        weight_on_delivery.append(date_demand_in_n.values[zipcode_num, 12])
        time_on_delivery.append(time_on_orders)
        # Check delivery restrictions
        if sum(time_on_delivery) + date_demand_in_n.values[zipcode_num, 21] >= driver_wh \
                or sum(weight_on_delivery) >= vehicle_capacity:
            # Create deliveries for each day in base
            vehicles.append(zipcode_num)
            # Use recursion to create new delivery if restrictions are exceeded
            vehicles_on_deliveries(date_demand_in_n=date_demand_in_nodes,
                                   vehicle_capacity=vehicle_load_limit_capacity,
                                   driver_wh=driver_working_hours,
                                   del_depot=zipcode_num)
            break
    return vehicles


def create_vehicles_deliveries_output(veh_on_tours: list, date_demand_in_n: pd.DataFrame) -> list:
    """Assign each order to a vehicle"""
    # For each order append his vehicle to veh_on_tours_output list
    veh_on_tours_output = []
    # Check if there was only one vehicle in one day
    if len(veh_on_tours) == 1:
        if veh_on_tours[0] == len(date_demand_in_n):
            for order in range(veh_on_tours[0]):
                veh_on_tours_output.append('1')
        else:
            for order in range(veh_on_tours[0]):
                veh_on_tours_output.append('1')
            for order in range(veh_on_tours[0], len(date_demand_in_n)):
                veh_on_tours_output.append('2')
    elif len(veh_on_tours) == 0:
        for order in range(len(date_demand_in_n)):
            veh_on_tours_output.append('1')
    else:
        for vehicle in range(len(veh_on_tours)):
            if vehicle == 0:
                for order in range(veh_on_tours[vehicle]):
                    veh_on_tours_output.append('1')
            else:
                for order in range(veh_on_tours[vehicle - 1], veh_on_tours[vehicle]):
                    veh_on_tours_output.append(f'{vehicle + 1}')
                if veh_on_tours[vehicle] == len(date_demand_in_n) - 1:
                    veh_on_tours_output.append(f'{vehicle + 2}')
                elif vehicle == len(veh_on_tours) - 1 and veh_on_tours[vehicle] != len(date_demand_in_n) - 1:
                    for order in range(veh_on_tours[vehicle], len(date_demand_in_n)):
                        veh_on_tours_output.append(f'{vehicle + 2}')
    return veh_on_tours_output


def tours_of_vehicles(demand_in_n: pd.DataFrame,
                      distance_or_time: str, retour: bool,
                      without_delay_at_delivery_location: bool = False):
    """Create input parameters for tours of vehicles."""
    tours_of_vehicles_output = []
    tours_dates = demand_in_n['Datum'].unique().tolist()
    for tour_date in tours_dates:
        date_demand_in_n = demand_in_n[demand_in_n["Datum"] == tour_date]
        vehicles_tours = date_demand_in_n['VozilaNaIsporukama'].unique().tolist()
        for vehicle in vehicles_tours:
            tour_of_vehicle = date_demand_in_n[date_demand_in_n['VozilaNaIsporukama'] == vehicle]
            if distance_or_time == 'time':
                if retour:
                    output_parameter = tour_of_vehicle.iat[0, 17] + \
                                        tour_of_vehicle.iloc[1:, 19].sum() + \
                                        tour_of_vehicle.iloc[:, 14].sum() + \
                                        tour_of_vehicle.iat[len(tour_of_vehicle) - 1, 21]
                    if without_delay_at_delivery_location:
                        tours_of_vehicles_output.extend([output_parameter - tour_of_vehicle.iloc[:, 14].sum()]
                                                        * len(tour_of_vehicle))
                    else:
                        tours_of_vehicles_output.extend([output_parameter] * len(tour_of_vehicle))
                else:
                    output_parameter = tour_of_vehicle.iat[0, 17] + \
                                        tour_of_vehicle.iloc[1:, 19].sum() + \
                                        tour_of_vehicle.iloc[:, 14].sum()
                    if without_delay_at_delivery_location:
                        tours_of_vehicles_output.extend([output_parameter - tour_of_vehicle.iloc[:, 14].sum()]
                                                        * len(tour_of_vehicle))
                    else:
                        tours_of_vehicles_output.extend([output_parameter] * len(tour_of_vehicle))
            else:
                if retour:
                    output_parameter = tour_of_vehicle.iat[0, 16] + \
                                        tour_of_vehicle.iloc[1:, 18].sum() + \
                                        tour_of_vehicle.iat[len(tour_of_vehicle) - 1, 20]
                    tours_of_vehicles_output.extend([output_parameter] * len(tour_of_vehicle))
                else:
                    output_parameter = tour_of_vehicle.iat[0, 16] + \
                                        tour_of_vehicle.iloc[1:, 18].sum()
                    tours_of_vehicles_output.extend([output_parameter] * len(tour_of_vehicle))
    return tours_of_vehicles_output


# Loading data from excel file
pd_input_file = r'./RutiranjeVozila.xlsx'
distance_matrix = pd.DataFrame(pd.read_excel(pd_input_file, sheet_name='tbUlazMatricaRastojanja'))
distance_matrix = distance_matrix.sort_values(by=['PTTOD', 'PTTDO', 'RastojanjeKm'], ascending=True)
vehicle_category = pd.DataFrame(pd.read_excel(pd_input_file, sheet_name='tbUlazKategorijaVozila'))
demand_in_nodes = pd.DataFrame(pd.read_excel(pd_input_file, sheet_name='tbUlazKarticaZahteva'))

# Input parameters
input_parameters = pd.DataFrame(pd.read_excel(pd_input_file, sheet_name='tbUlazParametri'))

# Restrictions
driver_working_hours = input_parameters.values[0, 0]
vehicle_load_limit_capacity = input_parameters.values[0, 1]

# Creating list with dates to filter dataframes
# and create dataframe for each date(because we create nearest neighbor algorithm for each day)
dates = demand_in_nodes['Datum'].unique().tolist()
dates = sorted(dates)
gigantic_route = {}
for date in dates:
    date_demand_in_nodes = demand_in_nodes[demand_in_nodes['Datum'] == date]
    # If first delivery node is center(warehouse) zipcode
    try:
        depot = sorted(date_demand_in_nodes['PrimaocPTT'].tolist()).index(date_demand_in_nodes['CentarPTT'].iloc[0]) + 1
    # Except first delivery node isn't center(warehouse) zipcode
    except ValueError:
        df_depot = distance_matrix[(distance_matrix['PTTOD'] == date_demand_in_nodes['CentarPTT'].iloc[0])
                                   & (distance_matrix['PTTDO'].isin(sorted(
                                                                            date_demand_in_nodes['PrimaocPTT'].unique()
                                                                            .tolist())))]
        zipcode_index_df_depot = df_depot.loc[df_depot['RastojanjeKm'].idxmin()]['PTTDO']
        depot = sorted(date_demand_in_nodes['PrimaocPTT'].tolist()).index(zipcode_index_df_depot) + 1

    # Creating dictionary from distance_matrix for every date in database
    date_distance_matrix = {}
    counter = 1
    for zipcode_number in sorted(date_demand_in_nodes['PrimaocPTT'].iloc[:]):
        list_for_matrix = distance_matrix[(distance_matrix['PTTOD'] == zipcode_number) &
                                          (distance_matrix['PTTDO'].isin(sorted(
                                              date_demand_in_nodes['PrimaocPTT'].unique().tolist())))]\
                                                                                                .RastojanjeKm.tolist()
        list_for_matrix = expand_list_for_matrix(list_for_matrix, date_demand_in_nodes)
        # Creating distance matrix for all nodes
        date_distance_matrix[counter] = list_for_matrix
        counter += 1
    # Creating gigantic route for every date in database
    date_gigantic_route = nearest_neighbor_algorithm(date_distance_matrix, date_demand_in_nodes)

    # Sort values into gigantic route
    date_gigantic_route = sorted_nodes_indexes(date_gigantic_route, date_demand_in_nodes)

    # Creating a gigantic route for the whole year
    gigantic_route = gigantic_route | {key + (len(gigantic_route)): value
                                       for (key, value) in date_gigantic_route.items()}

# Checking order of zipcode numbers in database
demand_in_nodes_output = demand_in_nodes.sort_values(by=['Datum', 'PrimaocPTT'])
zipcode_num_order_in_database = demand_in_nodes_output['PrimaocPTT'].to_list()

# Creating output for the whole year
sequence_list = []

# For each value in zipcode_num_order_in_database
for value in zipcode_num_order_in_database:
    # Find a key for that value in gigantic_route
    key = next(key for key, val in gigantic_route.items() if val == value)
    # Append key into result_list
    sequence_list.append(key)

    # Remove key from gigantic_route
    del gigantic_route[key]

# Merging and creating output
demand_in_nodes_output['Sequence'] = sequence_list
demand_in_nodes_output = demand_in_nodes_output.loc[:, ['OtpremnicaID', 'Sequence']]
demand_in_nodes = pd.merge(
        left=demand_in_nodes,
        right=demand_in_nodes_output,
        left_on='OtpremnicaID',
        right_on='OtpremnicaID',
        how='left'
)
# Creating output and saving data to Excel file
demand_in_nodes['RB'] = demand_in_nodes['Sequence']
demand_in_nodes = demand_in_nodes.drop('Sequence', axis=1)
demand_in_nodes = demand_in_nodes.sort_values(by=['RB'], ascending=True)

# TODO: Sorted Gigantic Route
zipcode_list = demand_in_nodes['PrimaocPTT'].tolist()

# Create groups to sort data by specific column
counter = 1
output_zipcode_list = [counter]
for data_number in range(1, len(zipcode_list)):
    if zipcode_list[data_number] != zipcode_list[data_number - 1]:
        counter += 1
        output_zipcode_list.append(counter)
    else:
        output_zipcode_list.append(counter)

demand_in_nodes['Groups'] = output_zipcode_list

demand_in_nodes = demand_in_nodes.sort_values(['Datum', 'Groups', 'BrutoKG'])

# Create SERIAL datatype for the output
demand_in_nodes['RB'] = list(range(1, len(demand_in_nodes) + 1))

demand_in_nodes = demand_in_nodes.drop('Groups', axis=1)

# TODO: Routing_Zoning_Algorithm

# Calculate delivery time
distance_matrix['VremeH'] = distance_matrix['RastojanjeKm'] / distance_matrix['BrzinaKMH']

demand_in_nodes = create_database_demand_in_nodes(demand_in_n=demand_in_nodes, distance_m=distance_matrix)

# Delete unnecessary DataFrame
del distance_matrix

# Calculate vehicles for each day
vehicles_excel_output = []
for date in dates:
    date_demand_in_nodes = demand_in_nodes[demand_in_nodes['Datum'] == date]
    vehicles = []
    vehicles_on_tours = vehicles_on_deliveries(date_demand_in_n=date_demand_in_nodes,
                                               vehicle_capacity=vehicle_load_limit_capacity,
                                               driver_wh=driver_working_hours)
    vehicles_on_tours = create_vehicles_deliveries_output(veh_on_tours=vehicles_on_tours,
                                                          date_demand_in_n=date_demand_in_nodes)
    vehicles_excel_output += vehicles_on_tours
demand_in_nodes['VozilaNaIsporukama'] = vehicles_excel_output
demand_in_nodes['VozilaNaIsporukama'] = demand_in_nodes['VozilaNaIsporukama'].astype(int)

# TODO: Tours_Of_Vehicles
# Create parameters for tours of vehicles
demand_in_nodes['KilometaraTourRetour'] = tours_of_vehicles(demand_in_n=demand_in_nodes,
                                                            retour=True,
                                                            distance_or_time='distance')

demand_in_nodes['KilometaraTour'] = tours_of_vehicles(demand_in_n=demand_in_nodes,
                                                      retour=False,
                                                      distance_or_time='distance')

demand_in_nodes['UkupnoVremeNaTuri'] = tours_of_vehicles(demand_in_n=demand_in_nodes,
                                                         retour=True,
                                                         distance_or_time='time')

demand_in_nodes['VremeKretanja'] = tours_of_vehicles(demand_in_n=demand_in_nodes,
                                                     retour=True,
                                                     distance_or_time='time',
                                                     without_delay_at_delivery_location=True)

demand_in_nodes['VremeKretanjaDoPoslednjegIM'] = tours_of_vehicles(demand_in_n=demand_in_nodes,
                                                                   retour=False,
                                                                   distance_or_time='time',
                                                                   without_delay_at_delivery_location=True)

# Create new parameter to calculate delay at the delivery location
demand_in_nodes['UkupnoVremeOpslugeObjekata'] = demand_in_nodes['UkupnoVremeNaTuri'] - demand_in_nodes['VremeKretanja']

# Export data to Excel
demand_in_nodes = demand_in_nodes.rename(columns={'VremeIstovara': 'VremeZadrzavanjaNaIstovarnomMestuMINUTA'})
demand_in_nodes.iloc[:, 14] = demand_in_nodes.iloc[:, 14].mul(60)
with pd.ExcelWriter(pd_input_file, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
    demand_in_nodes.to_excel(writer, columns=['RB', 'Datum', 'CentarID', 'CentarNaziv', 'CentarPTT', 'KlijentID',
                                              'KlijentNaziv', 'PrimaocID', 'PrimaocNaziv', 'PrimaocGrad', 'PrimaocPTT',
                                              'OtpremnicaID', 'BrutoKG', 'Paleta',
                                              'VremeZadrzavanjaNaIstovarnomMestuMINUTA', 'VozilaNaIsporukama'],
                             sheet_name='tbUlazKarticaZahteva', index=False)
demand_in_nodes = demand_in_nodes.rename(columns={'VremeZadrzavanjaNaIstovarnomMestuMINUTA': 'VremeIstovara'})
demand_in_nodes.iloc[:, 14] = demand_in_nodes.iloc[:, 14].div(60)
# Group data to get tours of vehicles for each day
query = """
    SELECT Datum as KalendarskiDan,
        SUM(BrutoKG) as BrutoKG,
        VozilaNaIsporukama as BrojVozila,
        KilometaraTourRetour,
        KilometaraTour,
        VremeKretanjaDoPoslednjegIM,
        VremeKretanja,
        UkupnoVremeOpslugeObjekata,
        UkupnoVremeNaTuri,
        COUNT(VozilaNaIsporukama) as BrojIsporuka 
        FROM demand_in_nodes 
        GROUP BY Datum, VozilaNaIsporukama 
        ORDER BY Datum, VozilaNaIsporukama
        """
demand_in_nodes = sqldf(query=query)
demand_in_nodes['KalendarskiDan'] = pd.to_datetime(demand_in_nodes['KalendarskiDan'])


# Create SERIAL datatype for the output
demand_in_nodes['IDTure'] = list(range(1, len(demand_in_nodes) + 1))

# Assign vehicle categories to the DataFrame
all_categories = vehicle_category['RealizovanaNosivostUSRB'].unique().tolist()
weight_in_dataframe = demand_in_nodes['BrutoKG'].to_list()
vehicle_category_output = [
    next((category + 1 for category, upper_bound in enumerate(all_categories) if weight <= upper_bound), None)
    for weight in weight_in_dataframe
]
demand_in_nodes['TipVozilaID'] = vehicle_category_output
demand_in_nodes = pd.merge(demand_in_nodes, vehicle_category, on='TipVozilaID', how='left')

# Export data to Excel file
demand_in_nodes = demand_in_nodes.iloc[:, [10, 0, 2, 9, 1, 3, 4, 5, 6, 7, 8, 11, 12, 13]]
demand_in_nodes.to_excel('Ture.xlsx', sheet_name='Ture', index=False)

stop = time.time()
print(stop - start)
