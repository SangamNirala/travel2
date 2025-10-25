import networkx as nx
import pickle
import numpy as np
from tensorflow import keras

class RouteOptimizer:
    def __init__(self, graph_path, model_path, scaler_path, mappings_path):
        """
        Initialize the route optimizer with trained model and graph
        """
        # Load graph
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load station mappings
        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            self.station_to_idx = mappings['station_to_idx']
            self.all_stations = mappings['all_stations']
        
        # Create reverse mapping
        self.idx_to_station = {idx: station for station, idx in self.station_to_idx.items()}
    
    def predict_travel_time(self, from_station, to_station, distance):
        """
        Predict travel time between two stations using neural network
        """
        if from_station not in self.station_to_idx or to_station not in self.station_to_idx:
            return None
        
        from_idx = self.station_to_idx[from_station]
        to_idx = self.station_to_idx[to_station]
        
        # Prepare input
        X = np.array([[from_idx, to_idx, distance]])
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled, verbose=0)
        return float(prediction[0][0])
    
    def find_optimal_route(self, source, destination):
        """
        Find optimal route between two stations using Dijkstra's algorithm
        with NN-predicted travel times
        """
        if source not in self.graph or destination not in self.graph:
            return None
        
        try:
            # Use Dijkstra's algorithm with travel time as weight
            path = nx.shortest_path(self.graph, source, destination, weight='weight')
            
            # Calculate total distance and time
            total_distance = 0
            total_time = 0
            route_details = []
            
            for i in range(len(path) - 1):
                current = path[i]
                next_station = path[i + 1]
                
                # Get edge data
                edge_data = self.graph[current][next_station]
                distance = edge_data.get('distance', 0)
                time = edge_data.get('weight', 0)
                
                total_distance += distance
                total_time += time
                
                route_details.append({
                    'from': current,
                    'from_name': self.graph.nodes[current]['name'],
                    'to': next_station,
                    'to_name': self.graph.nodes[next_station]['name'],
                    'distance': round(distance, 2),
                    'time': round(time, 2)
                })
            
            # Get station names for the complete path
            path_with_names = [{
                'code': station,
                'name': self.graph.nodes[station]['name']
            } for station in path]
            
            return {
                'path': path_with_names,
                'route_details': route_details,
                'total_distance': round(total_distance, 2),
                'total_time': round(total_time, 2),
                'total_time_hours': round(total_time / 60, 2)
            }
        
        except nx.NetworkXNoPath:
            return None
    
    def get_all_stations(self):
        """
        Get all available stations
        """
        stations = [{
            'code': node,
            'name': self.graph.nodes[node]['name']
        } for node in self.graph.nodes()]
        
        return sorted(stations, key=lambda x: x['name'])
    
    def get_station_connections(self, station_code):
        """
        Get all direct connections from a station
        """
        if station_code not in self.graph:
            return []
        
        connections = []
        for neighbor in self.graph.successors(station_code):
            edge_data = self.graph[station_code][neighbor]
            connections.append({
                'code': neighbor,
                'name': self.graph.nodes[neighbor]['name'],
                'distance': edge_data.get('distance', 0),
                'time': edge_data.get('weight', 0)
            })
        
        return connections