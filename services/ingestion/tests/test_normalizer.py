from datetime import UTC, datetime

from weathergpt_ingestion.models import Provenance, RawWeatherEvent, SourceName
from weathergpt_ingestion.normalizer import normalize_event
from weathergpt_ingestion.topics import IMD_CURRENT, OPEN_METEO_CURRENT


def test_open_meteo_current_payload_normalizes_to_observation_contract():
    event = RawWeatherEvent(
        source=SourceName.OPEN_METEO,
        topic=OPEN_METEO_CURRENT,
        external_id="open-meteo:19.0760:72.8777:2026-09-12T12:00",
        location_name="Mumbai",
        latitude=19.076,
        longitude=72.8777,
        observed_at=datetime(2026, 9, 12, 12, 0, tzinfo=UTC),
        payload={
            "current": {
                "temperature_2m": 28.4,
                "relative_humidity_2m": 81,
                "precipitation": 0.2,
                "weather_code": 3,
                "surface_pressure": 1006.2,
                "wind_speed_10m": 18.0,
                "wind_direction_10m": 240,
            }
        },
        provenance=Provenance(source=SourceName.OPEN_METEO, connector="open_meteo_current"),
    )

    observation = normalize_event(event)

    assert observation.location_name == "mumbai"
    assert observation.temp_c == 28.4
    assert observation.humidity_pct == 81
    assert observation.weather_code == "3"
    assert observation.provenance.source == SourceName.OPEN_METEO


def test_imd_current_payload_normalizes_to_observation_contract():
    event = RawWeatherEvent(
        source=SourceName.IMD,
        topic=IMD_CURRENT,
        external_id="imd:42182:2026-09-12T12:00:00+00:00",
        location_name="Delhi",
        latitude=28.6139,
        longitude=77.2090,
        observed_at=datetime(2026, 9, 12, 12, 0, tzinfo=UTC),
        payload={
            "data": [
                {
                    "Station Id": "42182",
                    "Station": "NEW DELHI/SAFDARJUNG",
                    "Date of Observation": "2026-09-12",
                    "Time of Observation": "12:00",
                    "M.S.L.P": "1003.0",
                    "Wind Direction": "189",
                    "Wind Speed": "11",
                    "Temperature": "34.2",
                    "Weather Code": "05",
                    "Humidity": "45",
                    "Last 24 hrs Rainfall": "0.0",
                }
            ]
        },
        provenance=Provenance(source=SourceName.IMD, connector="imd_current_weather"),
    )

    observation = normalize_event(event)

    assert observation.location_name == "delhi"
    assert observation.temp_c == 34.2
    assert observation.humidity_pct == 45
    assert observation.wind_speed_kph == 11
    assert observation.wind_direction_deg == 189
    assert observation.pressure_hpa == 1003
    assert observation.precipitation_mm == 0
    assert observation.weather_code == "05"
