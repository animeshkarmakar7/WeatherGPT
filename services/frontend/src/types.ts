export interface StructuredWeatherData {
  location: string;
  target_date: string;
  summary: string;
  will_rain: boolean;
  precipitation_probability_pct?: number | null;
  temp_c?: number | null;
  temp_min_c?: number | null;
  temp_max_c?: number | null;
  humidity_pct?: number | null;
  wind_speed_kph?: number | null;
  conditions: string;
  confidence: number;
  data_sources: string[];
  quality_flags: string[];
  observed_at?: string | null;
  fetched_at?: string | null;
  freshness: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  structured?: StructuredWeatherData;
  timestamp: string;
}

export interface WeatherCurrent {
  location: string;
  latitude: number;
  longitude: number;
  query_time?: string;
  target_date: string;
  temp_c?: number | null;
  temp_min_c?: number | null;
  temp_max_c?: number | null;
  humidity_pct?: number | null;
  precipitation_mm?: number | null;
  precipitation_probability_pct?: number | null;
  wind_speed_kph?: number | null;
  weather_code?: string | null;
  condition_description: string;
  will_rain: boolean;
  source: string;
  source_url?: string | null;
  observed_at?: string | null;
  fetched_at?: string | null;
  freshness: string;
  cached: boolean;
}

export interface WeatherForecastDay {
  date: string;
  temperature_max_c: number;
  temperature_min_c: number;
  precipitation_mm: number;
  precipitation_probability_pct: number;
  wind_speed_kph: number;
  weather_code: number | string;
  sunrise: string;
  sunset: string;
}

export interface WeatherDashboard {
  location: string;
  latitude: number;
  longitude: number;
  current: WeatherCurrent;
  forecast: WeatherForecastDay[];
  map: {
    latitude: number;
    longitude: number;
    provider: string;
  };
  generated_at: string;
}

export interface LocationResult {
  name: string;
  admin1?: string;
  country?: string;
  country_code?: string;
  latitude: number;
  longitude: number;
  timezone?: string;
}
