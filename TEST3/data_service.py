import pandas as pd
from datetime import datetime
from database_dao import run_query_data

def get_state_times(from_date: str, until_date: str) -> pd.DataFrame:
    """
    Calculates the total time (in Hours) spent in each state.
    Version: FAST & HONEST (No fake data filling).
    """
    
    # --- 1. PYTHON PREPARATION ---
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        dt_start = datetime.strptime(from_date, fmt)
        dt_end = datetime.strptime(until_date, fmt)
    except ValueError:
        dt_start = datetime.strptime(from_date, "%Y-%m-%d")
        dt_end = datetime.strptime(until_date, "%Y-%m-%d")

    ts_start = int(dt_start.timestamp())
    ts_end = int(dt_end.timestamp())

    ms_start = ts_start * 1000
    ms_end = ts_end * 1000

    # --- 2. OPTIMIZED SQL QUERY ---
    sql_query = """
    WITH RawSignal AS (
        SELECT
            to_timestamp(floor(CAST(date AS BIGINT) / 1000)) AS timestamp,
            COUNT(DISTINCT id_var) AS distinct_vars_count
        FROM
            public.variable_log_float
        WHERE
            CAST(date AS BIGINT) >= :ms_start 
            AND CAST(date AS BIGINT) <= :ms_end
        GROUP BY
            timestamp
    ),
    IdleGaps AS (
        SELECT
            'True Idle (Off)' AS state,
            SUM(EXTRACT(EPOCH FROM gap_duration)) / 3600.0 as total_hours
        FROM (
            SELECT
                timestamp - (LAG(timestamp) OVER (ORDER BY timestamp) + interval '1 second') AS gap_duration
            FROM
                RawSignal
        ) AS Gaps
        WHERE
            gap_duration > interval '0 seconds'
    ),
    SmoothedSignal AS (
        SELECT
            timestamp,
            AVG(distinct_vars_count) OVER (
                PARTITION BY date(timestamp) 
                ORDER BY timestamp
                ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
            ) AS smoothed_count,
            ROW_NUMBER() OVER (PARTITION BY date(timestamp) ORDER BY timestamp) as row_num_per_day
        FROM
            RawSignal
    ),
    ActiveStateTotals AS (
        SELECT
            CASE
                WHEN smoothed_count <= 14 THEN 'Low Activity'
                WHEN smoothed_count <= 20 THEN 'Intermediate Activity'
                ELSE 'High Activity'
            END AS state,
            (COUNT(*) * 1.0) / 3600.0 as total_hours 
        FROM
            SmoothedSignal
        WHERE
            row_num_per_day > 14 
        GROUP BY
            state
    )
    SELECT * FROM IdleGaps
    UNION ALL
    SELECT * FROM ActiveStateTotals;
    """
    
    params = {"ms_start": ms_start, "ms_end": ms_end}
    
    df = run_query_data(sql_query, params)
    
    if df.empty:
        return pd.DataFrame(columns=['state', 'total_hours'])
        
    return df

def get_machine_alarms(from_date: str, until_date: str) -> pd.DataFrame:
    """
    Returns AGGREGATED statistics for alarms.
    COLUMNS: alarm_code, alarm_text, occurrence_count, last_seen.
    OPTIMIZATION: Early Filtering (Regex) to skip noise lines.
    """
    
    # --- 1. PYTHON PREPARATION ---
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        dt_start = datetime.strptime(from_date, fmt)
        dt_end = datetime.strptime(until_date, fmt)
    except ValueError:
        dt_start = datetime.strptime(from_date, "%Y-%m-%d")
        dt_end = datetime.strptime(until_date, "%Y-%m-%d")

    ts_start = int(dt_start.timestamp())
    ts_end = int(dt_end.timestamp())
    ms_start = ts_start * 1000
    ms_end = ts_end * 1000

    # --- 2. SQL QUERY ---
    sql_query = r"""
    WITH raw AS (
        SELECT
            to_timestamp(floor(CAST(date AS BIGINT) / 1000)) AS ts,
            value,
            LEAD(to_timestamp(floor(CAST(date AS BIGINT) / 1000))) OVER (ORDER BY date) AS next_ts
        FROM variable_log_string
        WHERE id_var = 447
          AND CAST(date AS BIGINT) >= :ms_start 
          AND CAST(date AS BIGINT) <= :ms_end
          
          -- ⚡ EARLY FILTER (Noise Suppression) ⚡
          AND value !~ '(PLC00054|PLC00010|PLC01005|PLC00499|PLC00051|PLC00050|PLC00474|PLC00475|2a8-0003|130-019c|PLC00052|PLC00761)'
    ),
    flat AS (
        SELECT
            r.ts,
            r.next_ts,
            (m)[1] AS alarm_code,
            (m)[2] AS alarm_text
        FROM raw r
        CROSS JOIN LATERAL regexp_matches(
            r.value,
            '\["([^"]+)","([^"]+)",([0-9]+),([0-9]+),([0-9]+)\]',
            'g'
        ) AS m
        WHERE r.next_ts IS NOT NULL
    ),
    segments AS (
        SELECT
            alarm_code, alarm_text, ts, next_ts,
            LAG(next_ts) OVER (PARTITION BY alarm_code, alarm_text ORDER BY ts) AS prev_end
        FROM flat
    ),
    marked AS (
        SELECT *, CASE WHEN prev_end IS NULL OR prev_end < ts THEN 1 ELSE 0 END AS new_group
        FROM segments
    ),
    islands AS (
        SELECT *, SUM(new_group) OVER (PARTITION BY alarm_code, alarm_text ORDER BY ts) AS grp
        FROM marked
    ),
    periods AS (
        SELECT
            alarm_code,
            alarm_text,
            MIN(ts) AS start_time
            -- duration is no longer needed
        FROM islands
        GROUP BY alarm_code, alarm_text, grp
    )

    -- OUTPUT: Light Version
    SELECT
        alarm_code,
        alarm_text,
        -- 'reason' removed
        COUNT(*) AS occurrence_count,
        MAX(start_time) AS last_seen
        -- 'total_duration' removed
    FROM periods
    GROUP BY alarm_code, alarm_text
    ORDER BY occurrence_count DESC;
    """

    params = {"ms_start": ms_start, "ms_end": ms_end}
    return run_query_data(sql_query, params)

def get_energy_consumption(from_date: str, until_date: str) -> pd.DataFrame:
    """
    Calculates Energy (kWh) from Load Percentage (Variable 630).
    Data Team Formula: (Value% / 100) * 15kW * Hours.
    """
    # --- 1. PYTHON PREPARATION (Dates -> Milliseconds) ---
    fmt = "%Y-%m-%d %H:%M:%S"
    try:
        dt_start = datetime.strptime(from_date, fmt)
        dt_end = datetime.strptime(until_date, fmt)
    except ValueError:
        dt_start = datetime.strptime(from_date, "%Y-%m-%d")
        dt_end = datetime.strptime(until_date, "%Y-%m-%d")

    # Add margin to catch the point just before the start (for continuity)
    ts_start = int(dt_start.timestamp())
    ts_end = int(dt_end.timestamp())
    ms_start = ts_start * 1000
    ms_end = ts_end * 1000

    # --- 2. SQL QUERY ---
    # Based on '00_Duration_and_energy_per_day' file
    sql_query = """
    WITH RawData AS (
        SELECT 
            to_timestamp(date / 1000.0) as ts,
            GREATEST(LEAST(value, 100), 0) as pct -- Clamp value between 0 and 100%
        FROM variable_log_float
        WHERE id_var = 630  -- Variable: MANDRINO_CONSUMO_VISUALIZADO
          AND date >= :ms_start 
          AND date <= :ms_end
    ),
    TimeGaps AS (
        SELECT 
            ts,
            LEAD(ts) OVER (ORDER BY ts) as ts_next,
            pct
        FROM RawData
    ),
    ValidIntervals AS (
        SELECT * FROM TimeGaps
        WHERE ts_next IS NOT NULL 
          AND ts_next > ts
          -- Ignore data gaps > 1 hour (machine off or connection lost)
          AND EXTRACT(EPOCH FROM (ts_next - ts)) < 3600 
    ),
    DailySplit AS (
        -- Split intervals crossing midnight for accurate attribution
        SELECT
            (gs)::date as date,
            EXTRACT(EPOCH FROM (
                LEAST(ts_next, gs + interval '1 day') - GREATEST(ts, gs)
            )) / 3600.0 as hours,
            pct
        FROM ValidIntervals
        JOIN LATERAL generate_series(
            date_trunc('day', ts),
            date_trunc('day', ts_next),
            interval '1 day'
        ) gs ON TRUE
    )
    SELECT
        date,
        SUM((pct / 100.0) * 15.0 * hours) as total_energy_kwh
    FROM DailySplit
    GROUP BY date
    ORDER BY date;
    """
    
    params = {"ms_start": ms_start, "ms_end": ms_end}
    
    df = run_query_data(sql_query, params)
    
    if df.empty:
        return pd.DataFrame(columns=['date', 'total_energy_kwh'])
        
    return df