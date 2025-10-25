import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import { Button } from './components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card';
import { toast } from 'sonner';
import { Train, MapPin, Clock, Route, ArrowRight } from 'lucide-react';

const API_URL = process.env.REACT_APP_BACKEND_URL;

function App() {
  const [stations, setStations] = useState([]);
  const [sourceStation, setSourceStation] = useState('');
  const [destinationStation, setDestinationStation] = useState('');
  const [route, setRoute] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStations, setLoadingStations] = useState(true);

  useEffect(() => {
    fetchStations();
  }, []);

  const fetchStations = async () => {
    try {
      console.log('Fetching stations...');
      const response = await axios.get(`${API_URL}/api/stations`, {
        timeout: 30000 // 30 second timeout
      });
      console.log(`Loaded ${response.data.length} stations`);
      setStations(response.data);
      setLoadingStations(false);
      toast.success(`${response.data.length} stations loaded successfully!`);
    } catch (error) {
      console.error('Error fetching stations:', error);
      toast.error('Failed to load stations. Please refresh the page.');
      setLoadingStations(false);
    }
  };

  const findRoute = async () => {
    if (!sourceStation || !destinationStation) {
      toast.error('Please select both source and destination stations');
      return;
    }

    if (sourceStation === destinationStation) {
      toast.error('Source and destination cannot be the same');
      return;
    }

    setLoading(true);
    setRoute(null);

    try {
      const response = await axios.post(`${API_URL}/api/route`, {
        source: sourceStation,
        destination: destinationStation
      });
      setRoute(response.data);
      toast.success('Route found successfully!');
    } catch (error) {
      console.error('Error finding route:', error);
      toast.error(error.response?.data?.detail || 'Failed to find route');
    } finally {
      setLoading(false);
    }
  };

  const resetSearch = () => {
    setSourceStation('');
    setDestinationStation('');
    setRoute(null);
  };

  const getStationName = (code) => {
    const station = stations.find(s => s.code === code);
    return station ? station.name : code;
  };

  return (
    <div className="app-container">
      <div className="hero-section">
        <div className="hero-content">
          <div className="logo-container">
            <Train className="logo-icon" />
            <h1 className="app-title">Railway Path Optimizer</h1>
          </div>
          <p className="app-subtitle">AI-Powered Route Planning with Neural Networks</p>
        </div>
      </div>

      <div className="main-content">
        <Card className="search-card" data-testid="search-card">
          <CardHeader>
            <CardTitle>Find Your Optimal Route</CardTitle>
            <CardDescription>Select source and destination stations to find the best path</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="station-selection">
              <div className="station-input">
                <label>
                  <MapPin size={18} />
                  Source Station
                </label>
                <Select 
                  value={sourceStation} 
                  onValueChange={setSourceStation}
                  disabled={loadingStations}
                  data-testid="source-station-select"
                >
                  <SelectTrigger>
                    <SelectValue placeholder={loadingStations ? "Loading..." : "Select source station"} />
                  </SelectTrigger>
                  <SelectContent>
                    {stations.map((station) => (
                      <SelectItem key={station.code} value={station.code}>
                        {station.name} ({station.code})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="station-input">
                <label>
                  <MapPin size={18} />
                  Destination Station
                </label>
                <Select 
                  value={destinationStation} 
                  onValueChange={setDestinationStation}
                  disabled={loadingStations}
                  data-testid="destination-station-select"
                >
                  <SelectTrigger>
                    <SelectValue placeholder={loadingStations ? "Loading..." : "Select destination station"} />
                  </SelectTrigger>
                  <SelectContent>
                    {stations.map((station) => (
                      <SelectItem key={station.code} value={station.code}>
                        {station.name} ({station.code})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="action-buttons">
              <Button 
                onClick={findRoute} 
                disabled={loading || !sourceStation || !destinationStation}
                className="find-route-btn"
                data-testid="find-route-button"
              >
                <Route size={18} />
                {loading ? 'Finding Route...' : 'Find Optimal Route'}
              </Button>
              <Button 
                onClick={resetSearch} 
                variant="outline"
                disabled={loading}
                data-testid="reset-button"
              >
                Reset
              </Button>
            </div>
          </CardContent>
        </Card>

        {route && (
          <div className="results-section" data-testid="route-results">
            <Card className="summary-card">
              <CardHeader>
                <CardTitle>Route Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="summary-grid">
                  <div className="summary-item">
                    <div className="summary-icon">
                      <Route size={24} />
                    </div>
                    <div className="summary-details">
                      <span className="summary-label">Total Stations</span>
                      <span className="summary-value" data-testid="total-stations">{route.path.length}</span>
                    </div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-icon">
                      <MapPin size={24} />
                    </div>
                    <div className="summary-details">
                      <span className="summary-label">Total Distance</span>
                      <span className="summary-value" data-testid="total-distance">{route.total_distance} km</span>
                    </div>
                  </div>
                  <div className="summary-item">
                    <div className="summary-icon">
                      <Clock size={24} />
                    </div>
                    <div className="summary-details">
                      <span className="summary-label">Estimated Time</span>
                      <span className="summary-value" data-testid="total-time">{route.total_time_hours} hrs</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="path-card">
              <CardHeader>
                <CardTitle>Station Path</CardTitle>
                <CardDescription>Optimal route from {getStationName(sourceStation)} to {getStationName(destinationStation)}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="station-path" data-testid="station-path">
                  {route.path.map((station, index) => (
                    <div key={index} className="path-station">
                      <div className="station-marker">
                        <div className="marker-circle">{index + 1}</div>
                        {index < route.path.length - 1 && <div className="marker-line" />}
                      </div>
                      <div className="station-info">
                        <div className="station-name">{station.name}</div>
                        <div className="station-code">{station.code}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="details-card">
              <CardHeader>
                <CardTitle>Route Details</CardTitle>
                <CardDescription>Segment-by-segment journey information</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="route-details" data-testid="route-details">
                  {route.route_details.map((segment, index) => (
                    <div key={index} className="route-segment">
                      <div className="segment-header">
                        <span className="segment-number">Segment {index + 1}</span>
                      </div>
                      <div className="segment-content">
                        <div className="segment-stations">
                          <span className="from-station">{segment.from_name}</span>
                          <ArrowRight className="arrow-icon" size={20} />
                          <span className="to-station">{segment.to_name}</span>
                        </div>
                        <div className="segment-metrics">
                          <div className="metric">
                            <MapPin size={16} />
                            <span>{segment.distance} km</span>
                          </div>
                          <div className="metric">
                            <Clock size={16} />
                            <span>{Math.round(segment.time)} min</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
