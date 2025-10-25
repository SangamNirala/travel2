import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import networkx as nx

def parse_time(time_str):
    """Parse time string to timedelta"""
    try:
        if pd.isna(time_str) or time_str == '00:00:00':
            return timedelta(0)
        parts = time_str.split(':')
        return timedelta(hours=int(parts[0]), minutes=int(parts[1]), seconds=int(parts[2]))
    except:
        return timedelta(0)

def preprocess_data(csv_path):
    """
    Load and preprocess the railway dataset
    """
    # Load dataset with proper dtypes
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Convert Distance to numeric
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    
    # Parse time columns
    df['Arrival_Time_Delta'] = df['Arrival time'].apply(parse_time)
    df['Departure_Time_Delta'] = df['Departure Time'].apply(parse_time)
    
    # Calculate travel duration (time at this station)
    df['Travel_Duration'] = df['Departure_Time_Delta'] - df['Arrival_Time_Delta']
    df['Travel_Duration_Minutes'] = df['Travel_Duration'].dt.total_seconds() / 60
    
    # Handle negative durations (next day departure)
    df.loc[df['Travel_Duration_Minutes'] < 0, 'Travel_Duration_Minutes'] += 24 * 60
    
    # Sort by train and sequence
    df = df.sort_values(['Train No', 'SEQ'])
    
    # Calculate journey time between consecutive stations
    df['Next_Station_Code'] = df.groupby('Train No')['Station Code'].shift(-1)
    df['Next_Station_Name'] = df.groupby('Train No')['Station Name'].shift(-1)
    df['Next_Distance'] = df.groupby('Train No')['Distance'].shift(-1)
    df['Next_Arrival_Time'] = df.groupby('Train No')['Arrival_Time_Delta'].shift(-1)
    
    # Journey duration to next station
    df['Journey_Duration'] = (df['Next_Arrival_Time'] - df['Departure_Time_Delta']).dt.total_seconds() / 60
    df.loc[df['Journey_Duration'] < 0, 'Journey_Duration'] += 24 * 60
    
    # Distance between consecutive stations
    df['Distance_To_Next'] = df['Next_Distance'] - df['Distance']
    
    # Remove last station of each train (no next station)
    df_routes = df[df['Next_Station_Code'].notna()].copy()
    
    return df, df_routes

def build_station_graph(df_routes):
    """
    Build a graph of railway network
    """
    G = nx.DiGraph()
    
    # Add all unique stations
    stations = pd.concat([
        df_routes[['Station Code', 'Station Name']],
        df_routes[['Next_Station_Code', 'Next_Station_Name']].rename(
            columns={'Next_Station_Code': 'Station Code', 'Next_Station_Name': 'Station Name'}
        )
    ]).drop_duplicates()
    
    for _, row in stations.iterrows():
        G.add_node(row['Station Code'], name=row['Station Name'])
    
    # Add edges with weights (average travel time and distance)
    edge_data = df_routes.groupby(['Station Code', 'Next_Station_Code']).agg({
        'Journey_Duration': 'mean',
        'Distance_To_Next': 'mean',
        'Train No': 'count'
    }).reset_index()
    
    for _, row in edge_data.iterrows():
        if row['Journey_Duration'] > 0 and row['Distance_To_Next'] > 0:
            G.add_edge(
                row['Station Code'],
                row['Next_Station_Code'],
                weight=row['Journey_Duration'],
                distance=row['Distance_To_Next'],
                train_count=row['Train No']
            )
    
    return G

def prepare_training_data(df_routes):
    """
    Prepare data for neural network training
    """
    # Remove rows with invalid data
    train_df = df_routes[
        (df_routes['Journey_Duration'] > 0) & 
        (df_routes['Distance_To_Next'] > 0) &
        (df_routes['Journey_Duration'] < 1440)  # Less than 24 hours
    ].copy()
    
    # Create station code mappings
    all_stations = pd.concat([
        train_df['Station Code'],
        train_df['Next_Station_Code']
    ]).unique()
    
    station_to_idx = {station: idx for idx, station in enumerate(all_stations)}
    
    # Features: station codes as indices, distance
    train_df['station_idx'] = train_df['Station Code'].map(station_to_idx)
    train_df['next_station_idx'] = train_df['Next_Station_Code'].map(station_to_idx)
    
    X = train_df[['station_idx', 'next_station_idx', 'Distance_To_Next']].values
    y = train_df['Journey_Duration'].values
    
    return X, y, station_to_idx, all_stations

if __name__ == '__main__':
    print("Preprocessing data...")
    df, df_routes = preprocess_data('Train_details_22122017.csv')
    print(f"Total records: {len(df)}")
    print(f"Route segments: {len(df_routes)}")
    
    print("\nBuilding station graph...")
    G = build_station_graph(df_routes)
    print(f"Stations: {G.number_of_nodes()}")
    print(f"Connections: {G.number_of_edges()}")
    
    print("\nPreparing training data...")
    X, y, station_to_idx, all_stations = prepare_training_data(df_routes)
    print(f"Training samples: {len(X)}")
    
    # Save processed data
    with open('station_graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    
    with open('station_mappings.pkl', 'wb') as f:
        pickle.dump({
            'station_to_idx': station_to_idx,
            'all_stations': all_stations
        }, f)
    
    np.save('X_train.npy', X)
    np.save('y_train.npy', y)
    
    print("\nData preprocessing complete!")