import React, { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  ChevronRight,
  Cloud,
  CloudDrizzle,
  CloudRain,
  CloudSun,
  Droplets,
  LocateFixed,
  MapPin,
  Menu,
  MessageCircle,
  Navigation,
  Search,
  Send,
  ShieldCheck,
  Star,
  Sun,
  Thermometer,
  Wind,
  X,
  Zap,
} from "lucide-react";
import type { ChatMessage, LocationResult, WeatherDashboard, WeatherForecastDay } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8082";
const FAVORITES_KEY = "weathergpt-favorite-cities";

type CitySelection = {
  name: string;
  latitude: number;
  longitude: number;
};

const wmoIcon = (code?: number | string | null, size = 24) => {
  const value = Number(code);
  if (value >= 95) return <Zap size={size} />;
  if (value >= 80) return <CloudRain size={size} />;
  if (value >= 51) return <CloudDrizzle size={size} />;
  if (value >= 45) return <Cloud size={size} />;
  if (value >= 1) return <CloudSun size={size} />;
  return <Sun size={size} />;
};

const formatDay = (value: string) =>
  new Intl.DateTimeFormat("en-IN", { weekday: "short" }).format(new Date(`${value}T12:00:00`));

const formatDate = (value: string) =>
  new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short" }).format(new Date(`${value}T12:00:00`));

const formatTime = (value?: string | null) => {
  if (!value) return "Unavailable";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return "Unavailable";
  }

  return new Intl.DateTimeFormat("en-IN", {
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
};

const readFavorites = (): CitySelection[] => {
  try {
    return JSON.parse(localStorage.getItem(FAVORITES_KEY) || "[]");
  } catch {
    return [];
  }
};

export const App: React.FC = () => {
  const [dashboard, setDashboard] = useState<WeatherDashboard | null>(null);
  const [locationLoading, setLocationLoading] = useState(true);
  const [locationError, setLocationError] = useState<string | null>(null);
  const [searchValue, setSearchValue] = useState("");
  const [searchResults, setSearchResults] = useState<LocationResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [favorites, setFavorites] = useState<CitySelection[]>(readFavorites);
  const [activeTab, setActiveTab] = useState("Overview");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Ask about the weather, forecasts, or official guidance.",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [sessionId] = useState(() => `session-${Math.random().toString(36).slice(2, 10)}`);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const favoriteActive = useMemo(() => {
    if (!dashboard) return false;
    return favorites.some((item) => item.name.toLowerCase() === dashboard.location.toLowerCase());
  }, [dashboard, favorites]);

  const loadDashboard = async (selection: CitySelection) => {
    setLocationError(null);
    setLocationLoading(true);
    try {
      const params = new URLSearchParams({
        latitude: String(selection.latitude),
        longitude: String(selection.longitude),
        location_name: selection.name,
      });
      const response = await fetch(`${API_BASE_URL}/api/v1/weather/dashboard?${params.toString()}`);
      if (!response.ok) throw new Error("Verified live weather is unavailable");
      const data: WeatherDashboard = await response.json();
      setDashboard(data);
    } catch (error) {
      setLocationError(error instanceof Error ? error.message : "Verified live weather is unavailable");
    } finally {
      setLocationLoading(false);
    }
  };

  const requestCurrentLocation = () => {
    setLocationError(null);
    setLocationLoading(true);
    if (!navigator.geolocation) {
      setLocationError("Location access is not available in this browser.");
      setLocationLoading(false);
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (position) => {
        void loadDashboard({ name: "Your location", latitude: position.coords.latitude, longitude: position.coords.longitude });
      },
      () => {
        setLocationError("Location permission is required to show weather for your current position.");
        setLocationLoading(false);
      },
      { enableHighAccuracy: true, timeout: 12000, maximumAge: 300000 },
    );
  };

  useEffect(() => {
    requestCurrentLocation();
  }, []);

  useEffect(() => {
    const handle = window.setTimeout(async () => {
      const query = searchValue.trim();
      if (query.length < 2) {
        setSearchResults([]);
        return;
      }
      setSearchLoading(true);
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/weather/search?q=${encodeURIComponent(query)}`);
        if (!response.ok) throw new Error("Location search is unavailable");
        setSearchResults(await response.json());
      } catch {
        setSearchResults([]);
      } finally {
        setSearchLoading(false);
      }
    }, 260);
    return () => window.clearTimeout(handle);
  }, [searchValue]);

  useEffect(() => {
    localStorage.setItem(FAVORITES_KEY, JSON.stringify(favorites));
  }, [favorites]);

  const toggleFavorite = () => {
    if (!dashboard) return;
    const next: CitySelection = {
      name: dashboard.location,
      latitude: dashboard.latitude,
      longitude: dashboard.longitude,
    };
    setFavorites((current) =>
      currentActive(current, next) ? current.filter((item) => item.name.toLowerCase() !== next.name.toLowerCase()) : [...current, next].slice(-8),
    );
  };

  const sendChat = async (value?: string) => {
    const text = (value ?? chatInput).trim();
    if (!text || chatLoading) return;
    const userMessage: ChatMessage = { id: `u-${Date.now()}`, role: "user", content: text, timestamp: new Date().toISOString() };
    setChatMessages((current) => [...current, userMessage]);
    setChatInput("");
    setChatLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/chat/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text, location_hint: dashboard?.location || null }),
      });
      if (!response.ok) throw new Error("Verified answer is unavailable right now.");
      const data = await response.json();
      setChatMessages((current) => [
        ...current,
        { id: data.message_id || `a-${Date.now()}`, role: "assistant", content: data.response_text, structured: data.structured_data, timestamp: new Date().toISOString() },
      ]);
    } catch (error) {
      setChatMessages((current) => [
        ...current,
        { id: `e-${Date.now()}`, role: "assistant", content: error instanceof Error ? error.message : "Verified answer is unavailable right now.", timestamp: new Date().toISOString() },
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const current = dashboard?.current;
  const forecast = dashboard?.forecast || [];
  const mapUrl = dashboard
    ? `https://www.openstreetmap.org/export/embed.html?bbox=${dashboard.longitude - 0.12}%2C${dashboard.latitude - 0.08}%2C${dashboard.longitude + 0.12}%2C${dashboard.latitude + 0.08}&layer=mapnik&marker=${dashboard.latitude}%2C${dashboard.longitude}`
    : "";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><CloudSun size={23} /></div>
          <div>
            <div className="brand-name">WeatherGPT</div>
            <div className="brand-subtitle">Weather at a glance</div>
          </div>
        </div>
        <div className="desktop-search">
          <Search size={18} />
          <input value={searchValue} onChange={(event) => setSearchValue(event.target.value)} placeholder="Search a city" aria-label="Search a city" />
          {searchValue && <button className="icon-button subtle" onClick={() => setSearchValue("")} aria-label="Clear search"><X size={16} /></button>}
          {(searchResults.length > 0 || searchLoading) && (
            <div className="search-menu">
              {searchLoading && <div className="search-state">Searching locations…</div>}
              {!searchLoading && searchResults.map((result) => (
                <button key={`${result.name}-${result.latitude}`} className="search-result" onClick={() => { void loadDashboard({ name: result.name, latitude: result.latitude, longitude: result.longitude }); setSearchValue(""); setSearchResults([]); }}>
                  <MapPin size={16} />
                  <span><strong>{result.name}</strong><small>{[result.admin1, result.country].filter(Boolean).join(", ")}</small></span>
                  <ChevronRight size={16} />
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="mobile-menu-button" onClick={() => setMobileNavOpen((value) => !value)} aria-label="Open navigation"><Menu size={22} /></button>
        <button className="location-button" onClick={requestCurrentLocation}><LocateFixed size={17} /> Use my location</button>
      </header>

      <div className={`mobile-search ${mobileNavOpen ? "open" : ""}`}>
        <Search size={18} />
        <input value={searchValue} onChange={(event) => setSearchValue(event.target.value)} placeholder="Search a city" aria-label="Search a city" />
      </div>

      <main className="dashboard">
        <div className="tabs">
          {["Overview", "Forecast", "Map"].map((tab) => <button key={tab} className={activeTab === tab ? "active" : ""} onClick={() => setActiveTab(tab)}>{tab}</button>)}
        </div>

        {locationError && <section className="notice"><div><strong>Live weather unavailable</strong><span>{locationError}</span></div><button onClick={requestCurrentLocation}>Try again</button></section>}

        {locationLoading && !dashboard ? (
          <section className="loading-board"><div className="loading-orb"></div><h2>Getting live weather for your location</h2><p>Allow location access or search for a city to continue.</p></section>
        ) : dashboard && current ? (
          <>
            <section className={`hero-grid ${activeTab === "Map" ? "focus-map" : ""}`}>
              <div className="hero-card">
                <div className="hero-topline">
                  <div>
                    <span className="eyebrow"><MapPin size={14} /> {dashboard.location === "Your location" ? "Current location" : "Selected city"}</span>
                    <h1>{dashboard.location}</h1>
                    <p>{current.condition_description} <span className="dot-separator">•</span> Updated {formatTime(current.fetched_at)}</p>
                  </div>
                  <button className={`favorite-button ${favoriteActive ? "saved" : ""}`} onClick={toggleFavorite} aria-label="Save city"><Star size={18} fill={favoriteActive ? "currentColor" : "none"} /></button>
                </div>
               <div className="hero-weather">
  <div className="hero-icon">
    {wmoIcon(current.weather_code, 64)}
  </div>

  <div>
    <div className="hero-temp">
      {current.temp_c != null ? `${current.temp_c.toFixed(1)}°` : "—"}
      <span>C</span>
    </div>

    <div className="hero-range">Live conditions</div>
  </div>
</div>
                <div className="hero-meta">
                  <div><span>Feels</span><strong>{current.temp_c != null ? `${current.temp_c.toFixed(1)}°C` : "Unavailable"}</strong></div>
                  <div><span>Humidity</span><strong>{current.humidity_pct != null ? `${Math.round(current.humidity_pct)}%` : "Unavailable"}</strong></div>
                  <div><span>Wind</span><strong>{current.wind_speed_kph != null ? `${current.wind_speed_kph.toFixed(1)} km/h` : "Unavailable"}</strong></div>
                  <div><span>Rain now</span><strong>{current.precipitation_mm != null ? `${current.precipitation_mm.toFixed(1)} mm` : "Unavailable"}</strong></div>
                </div>
                <div className="source-line"><ShieldCheck size={15} /> {current.source.replaceAll("_", " ")} <span>•</span> {current.freshness}</div>
              </div>

              <div className="map-card">
                <div className="map-card-header"><div><span className="eyebrow"><Navigation size={14} /> Live map</span><h2>Where you are</h2></div><span className="live-pill"><i></i> Live</span></div>
                <div className="map-frame"><iframe title="Current weather location map" src={mapUrl} loading="lazy" /></div>
                <div className="map-footer"><span>{dashboard.latitude.toFixed(3)}°, {dashboard.longitude.toFixed(3)}°</span><span>{current.source.replaceAll("_", " ")}</span></div>
              </div>
            </section>

            <section className="section-block forecast-section">
              <div className="section-heading"><div><span className="eyebrow">10-day outlook</span><h2>Forecast timeline</h2></div><button className="text-button" onClick={() => setActiveTab("Forecast")}>View all <ArrowRight size={15} /></button></div>
              <div className="forecast-strip">
                {forecast.map((day: WeatherForecastDay, index) => (
                  <article className={`forecast-day ${index === 0 ? "today" : ""}`} key={day.date}>
                    <div className="day-name">{index === 0 ? "Today" : formatDay(day.date)}</div>
                    <div className="day-date">{formatDate(day.date)}</div>
                    <div className="day-icon">{wmoIcon(day.weather_code, 30)}</div>
                    <div className="day-temp"><strong>{Math.round(day.temperature_max_c)}°</strong><span>{Math.round(day.temperature_min_c)}°</span></div>
                    <div className="rain-chip"><Droplets size={13} /> {Math.round(day.precipitation_probability_pct)}%</div>
                    <div className="day-wind"><Wind size={13} /> {Math.round(day.wind_speed_kph)} km/h</div>
                  </article>
                ))}
              </div>
            </section>

            <section className="content-grid">
              <div className="section-block detail-card">
                <div className="section-heading"><div><span className="eyebrow">Today</span><h2>Conditions</h2></div></div>
                <div className="metric-grid">
                  <div className="metric"><Thermometer size={18} /><div><span>Temperature</span><strong>{current.temp_c != null ? `${current.temp_c.toFixed(1)}°C` : "Unavailable"}</strong></div></div>
                  <div className="metric"><Droplets size={18} /><div><span>Humidity</span><strong>{current.humidity_pct != null ? `${Math.round(current.humidity_pct)}%` : "Unavailable"}</strong></div></div>
                  <div className="metric"><Wind size={18} /><div><span>Wind speed</span><strong>{current.wind_speed_kph != null ? `${current.wind_speed_kph.toFixed(1)} km/h` : "Unavailable"}</strong></div></div>
                  <div className="metric"><CloudRain size={18} /><div><span>Precipitation</span><strong>{current.precipitation_mm != null ? `${current.precipitation_mm.toFixed(1)} mm` : "Unavailable"}</strong></div></div>
                </div>
              </div>

              <aside className="section-block favorites-card">
                <div className="section-heading"><div><span className="eyebrow">Saved places</span><h2>Favorites</h2></div></div>
                {favorites.length === 0 ? <div className="empty-favorites">Save cities you check often.</div> : favorites.map((favorite) => (
                  <button className="favorite-row" key={favorite.name} onClick={() => void loadDashboard(favorite)}><span className="favorite-city"><Star size={15} fill="currentColor" /> {favorite.name}</span><ChevronRight size={16} /></button>
                ))}
              </aside>
            </section>
          </>
        ) : null}
      </main>

      <button className="chat-fab" onClick={() => setChatOpen(true)} aria-label="Open weather assistant"><MessageCircle size={23} /><span>Ask WeatherGPT</span></button>

      {chatOpen && <div className="chat-overlay" onClick={() => setChatOpen(false)}>
        <aside className="chat-panel" onClick={(event) => event.stopPropagation()}>
          <div className="chat-header"><div><span className="eyebrow">Weather assistant</span><h2>Ask anything</h2></div><button className="icon-button" onClick={() => setChatOpen(false)} aria-label="Close chat"><X size={19} /></button></div>
          <div className="chat-thread">
            {chatMessages.map((message) => <div key={message.id} className={`chat-bubble ${message.role}`}>{message.content}{message.structured && <div className="chat-source">{message.structured.data_sources.join(", ")} • {message.structured.freshness}</div>}</div>)}
            {chatLoading && <div className="chat-bubble assistant">Checking live information…</div>}
          </div>
          <div className="chat-suggestions">{["Temperature here now", "Will it rain tomorrow?", "Official cyclone guidance"].map((text) => <button key={text} onClick={() => void sendChat(text)} disabled={chatLoading}>{text}</button>)}</div>
          <form className="chat-form" onSubmit={(event) => { event.preventDefault(); void sendChat(); }}><input value={chatInput} onChange={(event) => setChatInput(event.target.value)} placeholder="Ask about weather or guidance" /><button type="submit" disabled={!chatInput.trim() || chatLoading}><Send size={18} /></button></form>
        </aside>
      </div>}

      <footer className="page-footer">Live weather is shown only when verified data is available.</footer>
    </div>
  );
};

function currentActive(items: CitySelection[], target: CitySelection): boolean {
  return items.some((item) => item.name.toLowerCase() === target.name.toLowerCase());
}

export default App;
