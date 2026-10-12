import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import time
import urllib.request
import tempfile
import os

warnings.filterwarnings('ignore')

# =====================================================================
# USER CONFIGURATION
# =====================================================================
GLOBAL_MAX_DATE = '2026-07-31'


# =====================================================================

# =====================================================================
# UNIVERSAL DEDUPLICATION ENGINE (FIXES DOUBLE-COUNTING EVERYWHERE)
# =====================================================================
def deduplicate_to_intervals(target_df):
    records = []
    for region, group in target_df.groupby('region_name'):
        intervals = group[['started_at', 'finished_at']].dropna().values.tolist()
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            if not merged:
                merged.append([interval[0], interval[1]])
            else:
                if interval[0] <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], interval[1])
                else:
                    merged.append([interval[0], interval[1]])

        for m_start, m_end in merged:
            records.append({
                'region_name': region,
                'started_at': m_start,
                'finished_at': m_end
            })

    df_clean = pd.DataFrame(records)
    if not df_clean.empty:
        df_clean['duration_hours'] = (df_clean['finished_at'] - df_clean['started_at']).dt.total_seconds() / 3600.0
        df_clean['date'] = df_clean['started_at'].dt.floor('D')
        df_clean['year'] = df_clean['started_at'].dt.year
    return df_clean


# =====================================================================
# FORECASTING MODELS
# =====================================================================
def forecast_daily(series, horizon):
    vals = series.values
    idx = series.index
    if len(vals) < 15: return None, None

    df = pd.DataFrame({'y': vals}, index=idx)
    df['sin_t'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['cos_t'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(float)

    for i in range(1, 8): df[f'lag_{i}'] = df['y'].shift(i)
    df['roll_mean_7'] = df['y'].shift(1).rolling(7).mean()

    df = df.dropna()
    if len(df) < 5: return None, None

    X = df.drop('y', axis=1).values
    y = df['y'].values

    decay_rate = 0.1
    N = len(df)
    weights = np.exp(-decay_rate * (N - 1 - np.arange(N)))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=0.5)
    model.fit(X_scaled, y, sample_weight=weights)

    last_date = series.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq='D')

    preds, hist = [], list(vals)
    for date in future_dates:
        sin_t, cos_t = np.sin(2 * np.pi * date.dayofweek / 7), np.cos(2 * np.pi * date.dayofweek / 7)
        is_w = float(date.dayofweek in [5, 6])
        feat = [sin_t, cos_t, is_w]
        feat.extend([hist[-i] for i in range(1, 8)])
        feat.append(np.mean(hist[-7:]))
        pred = max(0.0, model.predict(scaler.transform([feat]))[0])
        preds.append(pred)
        hist.append(pred)
    return future_dates, preds


def forecast_monthly(series, horizon):
    vals = series.values
    idx = series.index
    if len(vals) < 10: return None, None

    df = pd.DataFrame({'y': vals}, index=idx)
    df['lag_1'], df['lag_2'], df['lag_3'] = df['y'].shift(1), df['y'].shift(2), df['y'].shift(3)
    df['roll_mean_3'], df['roll_mean_6'] = df['y'].shift(1).rolling(3).mean(), df['y'].shift(1).rolling(6).mean()

    df = df.dropna()
    if len(df) < 5: return None, None

    X, y = df.drop('y', axis=1).values, df['y'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = HuberRegressor(epsilon=1.35)
    model.fit(X_scaled, y)

    last_date = series.index.max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='ME')

    preds, hist = [], list(vals)
    for _ in range(horizon):
        feat = [hist[-1], hist[-2], hist[-3], np.mean(hist[-3:]), np.mean(hist[-6:])]
        pred = max(0.0, model.predict(scaler.transform([feat]))[0])
        preds.append(pred)
        hist.append(pred)
    return future_dates, preds


def safe_savefig(filename):
    try:
        plt.savefig(filename, dpi=300)
    except OSError:
        plt.savefig(filename.replace('.png', f'_fallback_{int(time.time())}.png'), dpi=300)


# =====================================================================
# DASHBOARD PIPELINE
# =====================================================================
def get_clean_daily_aggregates(target_df, start_date, end_date):
    daily_records = []
    for _, row in target_df.iterrows():
        m_start, m_end = row['started_at'], row['finished_at']
        current_day, end_day = m_start.floor('D'), m_end.floor('D')

        while current_day <= end_day:
            day_end = current_day + pd.Timedelta(days=1)
            overlap_start, overlap_end = max(m_start, current_day), min(m_end, day_end)
            if overlap_start < overlap_end:
                dur = (overlap_end - overlap_start).total_seconds() / 3600.0
                cnt = 1 if current_day == m_start.floor('D') else 0
                daily_records.append({'date': current_day, 'total_hours': dur, 'alert_count': cnt})
            current_day += pd.Timedelta(days=1)

    idx = pd.date_range(start=start_date, end=end_date, freq='D')
    if not daily_records: return pd.DataFrame(index=idx, columns=['total_hours', 'alert_count']).fillna(0)

    daily_df = pd.DataFrame(daily_records)
    full_daily = daily_df.groupby('date').sum().reindex(idx, fill_value=0)
    return full_daily


def plot_dashboard(hist_df, truth_df, freq_name, freq_code, horizon, title, filename, scope):
    if freq_code == 'D':
        f_dates_h, f_preds_h = forecast_daily(hist_df['total_hours'], horizon)
        f_dates_c, f_preds_c = forecast_daily(hist_df['alert_count'], horizon)
    else:
        f_dates_h, f_preds_h = forecast_monthly(hist_df['total_hours'], horizon)
        f_dates_c, f_preds_c = forecast_monthly(hist_df['alert_count'], horizon)

    if f_dates_h is None or len(f_dates_h) == 0:
        return

    error_text = ""
    if truth_df is not None and not truth_df.empty:
        mae_h = mean_absolute_error(truth_df['total_hours'].values, f_preds_h[:len(truth_df)])
        rmse_h = np.sqrt(mean_squared_error(truth_df['total_hours'].values, f_preds_h[:len(truth_df)]))
        mae_c = mean_absolute_error(truth_df['alert_count'].values, f_preds_c[:len(truth_df)])
        rmse_c = np.sqrt(mean_squared_error(truth_df['alert_count'].values, f_preds_c[:len(truth_df)]))

        model_name = "Ridge w/ Exp Decay" if freq_code == 'D' else "Huber Robust Regressor"
        error_text = (f"Backtest Evaluation ({model_name})\n"
                      f"Hours -> MAE: {mae_h:.1f} | RMSE: {rmse_h:.1f}\n"
                      f"Count -> MAE: {mae_c:.1f} | RMSE: {rmse_c:.1f}")

    fig, (ax_macro, ax_zoom) = plt.subplots(2, 1, figsize=(16, 13), gridspec_kw={'height_ratios': [1, 1.3]})
    fig.suptitle(f"{title}\n({scope})", fontsize=18, fontweight='bold')
    dataset_cutoff = hist_df.index[-1].strftime('%Y-%m-%d')
    asterisk = f"* Note: Timeline bounded to {dataset_cutoff} as configured."

    for ax, is_zoomed in [(ax_macro, False), (ax_zoom, True)]:
        ax_tw = ax.twinx()
        alpha = 0.25 if (not is_zoomed and freq_code == 'D') else 0.8

        ax.plot(hist_df.index, hist_df['total_hours'], color='midnightblue', alpha=alpha, label='Historical Hours')
        ax_tw.plot(hist_df.index, hist_df['alert_count'], color='darkcyan', alpha=alpha, label='Historical Count')
        ax.plot([hist_df.index[-1]] + list(f_dates_h), [hist_df['total_hours'].iloc[-1]] + f_preds_h,
                color='darkorange', ls='--', lw=3, label='Forecast (Hours)')
        ax_tw.plot([hist_df.index[-1]] + list(f_dates_c), [hist_df['alert_count'].iloc[-1]] + f_preds_c,
                   color='deepskyblue', ls='--', lw=3, label='Forecast (Count)')

        if truth_df is not None and len(truth_df) > 0:
            ax.plot([hist_df.index[-1]] + list(truth_df.index),
                    [hist_df['total_hours'].iloc[-1]] + list(truth_df['total_hours']), color='crimson', lw=2.5,
                    label='Actual (Hours)')
            ax_tw.plot([hist_df.index[-1]] + list(truth_df.index),
                       [hist_df['alert_count'].iloc[-1]] + list(truth_df['alert_count']), color='dodgerblue', lw=2.5,
                       label='Actual (Count)')

        ax.set_ylabel(f'Total Hours ({freq_name})', color='midnightblue', fontweight='bold')
        ax_tw.set_ylabel(f'Total Count ({freq_name})', color='darkcyan', fontweight='bold')

        if not is_zoomed and freq_code == 'D':
            ax.plot(hist_df.index, gaussian_filter1d(hist_df['total_hours'].astype(float), 7), color='darkorange',
                    lw=2.5, label='Hours Trend')
            ax_tw.plot(hist_df.index, gaussian_filter1d(hist_df['alert_count'].astype(float), 7), color='deepskyblue',
                       lw=2.5, label='Count Trend')

        # FIX: Anchor the prediction span to the last historical date to visually cover all predicted months
        ax.axvspan(hist_df.index[-1], f_dates_h[-1], color='gold', alpha=0.10, label='Prediction Window')
        ax.text(0.01, 0.02, asterisk, transform=ax.transAxes, color='crimson', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.85))

        if is_zoomed:
            ax.set_title("🔍 Zoomed View (Backtest)", fontsize=14, pad=10)
            ax.set_xlim([f_dates_h[-1] - pd.Timedelta(days=90 if freq_code == 'D' else 365),
                         f_dates_h[-1] + pd.Timedelta(days=1)])
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_tw.get_legend_handles_labels()
            by_label = dict(zip(labels + labels2, lines + lines2))
            ax.legend(by_label.values(), by_label.keys(), loc='upper left')

            if error_text:
                props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray')
                ax.text(0.98, 0.95, error_text, transform=ax.transAxes, fontsize=11,
                        verticalalignment='top', horizontalalignment='right', bbox=props, fontweight='bold',
                        color='black')
        else:
            ax.set_title("Macro Timeline", fontsize=14, pad=10)

        ax.grid(True, ls='--', alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    safe_savefig(filename)
    plt.close()


def generate_temporal_dashboards(target_df, scope_label, file_prefix):
    print(f"\n[-->] Generating Dashboards for Scope: {scope_label}")
    last_complete_day, start_date = target_df['date'].max(), target_df['date'].min()
    if pd.isna(last_complete_day): return

    full_daily = get_clean_daily_aggregates(target_df, start_date, last_complete_day)

    if len(full_daily) > 40:
        plot_dashboard(full_daily.iloc[:-30], full_daily.iloc[-30:], "Day", "D", 30, "Daily Alerts",
                       f"{file_prefix}_daily.png", scope_label)

    full_monthly = full_daily.resample('ME').sum()
    if not last_complete_day.is_month_end and len(full_monthly) > 1: full_monthly = full_monthly.iloc[:-1]
    if len(full_monthly) > 4:
        plot_dashboard(full_monthly.iloc[:-3], full_monthly.iloc[-3:], "Month", "ME", 3, "Monthly Alerts",
                       f"{file_prefix}_monthly.png", scope_label)



def map_matcher(name):
    n = str(name).lower()

    # Idempotency check: if already mapped, return it immediately to prevent
    # cross-contamination (e.g. 'kyiv_city' becoming 'kyiv_oblast' on a second pass)
    canonical = {'kyiv_city', 'kyiv_oblast', 'crimea', 'sevastopol', 'dnipropetrovsk',
                 'zaporizhzhia', 'mykolaiv', 'luhansk', 'donetsk', 'kharkiv', 'kherson',
                 'odesa', 'poltava', 'sumy', 'chernihiv', 'cherkasy', 'vinnytsia',
                 'zhytomyr', 'khmelnytskyi', 'rivne', 'volyn', 'ternopil',
                 'ivano-frankivsk', 'zakarpattia', 'chernivtsi', 'kirovohrad', 'lviv'}
    if n in canonical: return n

    if any(x in n for x in ['kyiv city', 'misto kyiv', 'kyyiv city', 'misto kyyiv', 'м. київ']) or n in ['kyiv', 'kiev',
                                                                                                         'kyyiv',
                                                                                                         'київ']: return 'kyiv_city'
    if any(x in n for x in ['kyiv', 'kiev', 'kyyiv', 'київськ', 'киевск']): return 'kyiv_oblast'
    if any(x in n for x in ['krym', 'crimea', 'avtonomna', 'крим', 'крым']): return 'crimea'
    if any(x in n for x in ['sevastopol', 'севастопол']): return 'sevastopol'
    if any(x in n for x in ['dnipro', 'дніпро', 'днепр']): return 'dnipropetrovsk'
    if any(x in n for x in ['zapor', 'запор']): return 'zaporizhzhia'
    if any(x in n for x in ['mykola', 'микола', 'никола']): return 'mykolaiv'
    if any(x in n for x in ['luhan', 'луган']): return 'luhansk'
    if any(x in n for x in ['donets', 'донец']): return 'donetsk'
    if any(x in n for x in ['khark', 'харків', 'харьков']): return 'kharkiv'
    if any(x in n for x in ['kherson', 'херсон']): return 'kherson'
    if any(x in n for x in ['odes', 'одес']): return 'odesa'
    if any(x in n for x in ['poltav', 'полтав']): return 'poltava'
    if any(x in n for x in ['sumy', 'sumsk', 'сумс', 'суми']): return 'sumy'
    if any(x in n for x in ['chernihiv', 'чернігів', 'чернигов']): return 'chernihiv'
    if any(x in n for x in ['cherkas', 'черкас']): return 'cherkasy'
    if any(x in n for x in ['vinnyt', 'вінниц', 'винниц']): return 'vinnytsia'
    if any(x in n for x in ['zhytomyr', 'житомир']): return 'zhytomyr'
    if any(x in n for x in ['khmelnyt', 'хмельниц']): return 'khmelnytskyi'
    if any(x in n for x in ['rivn', 'рівнен', 'ровен']): return 'rivne'
    if any(x in n for x in ['volyn', 'волин']): return 'volyn'
    if any(x in n for x in ['ternop', 'тернопіл', 'тернопол']): return 'ternopil'
    if any(x in n for x in ['ivano', 'frankivsk', 'івано', 'ивано']): return 'ivano-frankivsk'
    if any(x in n for x in ['zakarpat', 'transcarpath', 'закарпат']): return 'zakarpattia'
    if any(x in n for x in ['chernivts', 'чернівец', 'черновиц']): return 'chernivtsi'
    if any(x in n for x in ['kirovohrad', 'kropyvnyts', 'кіровоград', 'кропивниц']): return 'kirovohrad'
    if any(x in n for x in ['lviv', 'львів']): return 'lviv'
    return n


# =====================================================================
# GEOSPATIAL MAP GENERATION (PER-YEAR NORMALIZATION + PERPETUAL REGIONS)
# =====================================================================
# =====================================================================
# GEOSPATIAL MAP GENERATION (PER-YEAR NORMALIZATION + DEEP RED PERPETUAL)
# =====================================================================
def generate_geospatial_heatmaps(df):
    import time
    import json
    import urllib.request, tempfile, os

    start_time = time.time()
    print(f"\n[-->] Generating 4-Year Geospatial Heatmap Grid...")

    try:
        import geopandas as gpd
    except ImportError:
        return print("[!] ERROR: 'geopandas' is not installed.")

    print("    [*] Standardizing keys and re-flattening timeline to cap at 8760 max yearly hours...")
    df_map = df.copy()
    df_map['merge_key'] = df_map['region_name'].apply(map_matcher)

    if 'started_at' in df_map.columns and 'finished_at' in df_map.columns:
        records = []
        for region, group in df_map.groupby('merge_key'):
            intervals = group[['started_at', 'finished_at']].dropna().values.tolist()
            intervals.sort(key=lambda x: x[0])
            merged = []
            for interval in intervals:
                if not merged:
                    merged.append([interval[0], interval[1]])
                else:
                    if interval[0] <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], interval[1])
                    else:
                        merged.append([interval[0], interval[1]])
            for m_start, m_end in merged:
                records.append({'merge_key': region, 'started_at': m_start, 'finished_at': m_end})

        clean_map_df = pd.DataFrame(records)
        if not clean_map_df.empty:
            clean_map_df['duration_hours'] = (clean_map_df['finished_at'] - clean_map_df[
                'started_at']).dt.total_seconds() / 3600.0
            clean_map_df['year'] = clean_map_df['started_at'].dt.year
            heat_df = clean_map_df.groupby(['year', 'merge_key'])['duration_hours'].sum().reset_index()
        else:
            heat_df = pd.DataFrame(columns=['year', 'merge_key', 'duration_hours'])
    else:
        heat_df = df_map.groupby(['year', 'merge_key'])['duration_hours'].sum().reset_index()

    if heat_df.empty:
        return print("[!] No data available to plot on maps.")

    heat_df['duration_hours'] = heat_df['duration_hours'].clip(upper=8784)
    valid_keys = set(heat_df['merge_key'].unique())

    urls = []
    print("    [*] Querying official geoBoundaries API for live map routes...")
    try:
        api_req = urllib.request.Request("https://www.geoboundaries.org/api/current/gbOpen/UKR/ADM1/",
                                         headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(api_req, timeout=10) as response:
            api_data = json.loads(response.read())
            if 'simplifiedGeometryGeoJSON' in api_data:
                urls.append(api_data['simplifiedGeometryGeoJSON'])
            if 'gjDownloadURL' in api_data:
                urls.append(api_data['gjDownloadURL'])
    except Exception as e:
        print(f"    [-] API unreachable: {e}")

    urls.extend([
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_UKR_1.json",
        "https://raw.githubusercontent.com/geohacker/ukraine/master/geojson/ukraine.geojson"
    ])

    if not urls:
        return print("[!] No map URLs could be resolved.")

    print(f"    [*] Fetching and evaluating geospatial topologies...")
    ua_map = None
    best_col = None
    best_score = -1

    for url in urls:
        tmp_path = ""
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                geojson_data = response.read()

            try:
                json_check = json.loads(geojson_data)
                if 'features' not in json_check:
                    print(f"    [-] Map {url.split('/')[-1]} is missing features array. Skipping...")
                    continue
            except json.JSONDecodeError:
                print(f"    [-] Map {url.split('/')[-1]} returned invalid JSON (Likely 404). Skipping...")
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp:
                tmp.write(geojson_data)
                tmp_path = tmp.name

            try:
                tmp_map = gpd.read_file(tmp_path)
                if not tmp_map.empty:
                    current_best_col = None
                    current_best_score = -1

                    for col in tmp_map.columns:
                        if col == 'geometry': continue

                        mapped_geojson_col = tmp_map[col].astype(str).apply(map_matcher)
                        score = mapped_geojson_col.isin(valid_keys).sum()
                        if score > current_best_score:
                            current_best_score = score
                            current_best_col = col

                    if current_best_score > 0:
                        ua_map = tmp_map
                        best_col = current_best_col
                        best_score = current_best_score
                        print(
                            f"    [+] Successfully matched map: {url.split('/')[-1]} (Score: {best_score} on '{best_col}')")
                        break
                    else:
                        print(
                            f"    [-] Map {url.split('/')[-1]} loaded but lacks matching text properties. Skipping...")
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

        except Exception as e:
            print(f"    [!] Failed to load {url[-15:]}: {e}")
            if os.path.exists(tmp_path): os.remove(tmp_path)
            continue

    if ua_map is None or ua_map.empty:
        return print("[!] Failed to load or match any viable maps.")

    ua_map['merge_key'] = ua_map[best_col].astype(str).apply(map_matcher)
    valid_map_keys = set(ua_map['merge_key'].unique())
    matched_heat_df = heat_df[heat_df['merge_key'].isin(valid_map_keys)]

    years = sorted(matched_heat_df['year'].unique())
    if len(years) > 4: years = years[-4:]

    # Perpetual / ongoing alert regions to isolate from the dynamic color range
    perpetual_keys = {'crimea', 'sevastopol', 'luhansk'}
    deep_red_color = plt.cm.YlOrRd(1.0)  # Hex #800026: highest saturation on the YlOrRd colormap

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Total Air Raid Alert Hours by Region', fontsize=24, fontweight='bold', y=0.98)

    for ax, year in zip(axes.flatten(), years):
        year_data = heat_df[heat_df['year'] == year]
        if year_data.empty: continue

        merged = ua_map.merge(year_data, on='merge_key', how='left')
        merged['duration_hours'] = merged['duration_hours'].fillna(0)

        # Partition regular oblasts from perpetual alert regions
        is_perpetual = merged['merge_key'].isin(perpetual_keys)
        regular_gdf = merged[~is_perpetual]
        perpetual_gdf = merged[is_perpetual]

        # Year-specific normalization: max hours in the current year (excluding perpetual areas)
        vmax_year = regular_gdf['duration_hours'].max()
        if pd.isna(vmax_year) or vmax_year <= 0:
            vmax_year = 1

        # 1. Plot regular oblasts with the per-year normalized color scale
        regular_gdf.plot(
            column='duration_hours',
            ax=ax,
            cmap='YlOrRd',
            linewidth=0.6,
            edgecolor='black',
            vmin=0,
            vmax=vmax_year,
            legend=True,
            legend_kwds={
                'label': f"Total Alert Hours (Year Max: {vmax_year:.0f}h)",
                'orientation': "horizontal",
                'shrink': 0.6
            }
        )

        # 2. Overlay Crimea & Luhansk in the deepest YlOrRd red
        if not perpetual_gdf.empty:
            perpetual_gdf.plot(ax=ax, color=deep_red_color, linewidth=0.6, edgecolor='black')

        ax.set_title(f"Year: {year}", fontsize=18, fontweight='bold')
        ax.axis('off')

    for i in range(len(years), 4): axes.flatten()[i].axis('off')

    # Asterisk notice updated to reflect deep red
    fig.text(
        0.5, 0.02,
        '* Note: Crimea and Luhansk are highlighted in deep red due to continuous / perpetual alert status.',
        ha='center', fontsize=13, fontweight='bold', color='#800026'
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.06)
    safe_savefig('geospatial_heatmap_4years.png')
    plt.close()

    print(f"    [+] Total Map Pipeline Time: {time.time() - start_time:.2f}s.")
# =====================================================================
# TIME HEATMAPS
# =====================================================================
def generate_time_heatmaps(df):
    print(f"\n[-->] Generating Time-Analysis Heatmaps...")
    df_hm = df.copy()
    df_hm['hour'] = df_hm['started_at'].dt.hour
    df_hm['weekday'] = df_hm['started_at'].dt.dayofweek
    days_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

    hm_2d = df_hm.groupby(['weekday', 'hour']).size().unstack(fill_value=0).reindex(index=range(7), columns=range(24),
                                                                                    fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 6))
    cax = ax.imshow(hm_2d.values, cmap='YlOrRd', aspect='auto')
    fig.colorbar(cax, label='Cumulative Alert Count')
    ax.set_xticks(range(24));
    ax.set_yticks(range(7));
    ax.set_yticklabels([days_map[i] for i in hm_2d.index])
    ax.set_xlabel('Hour of the Day (UTC)');
    ax.set_title("2D Heatmap: Alert Frequency by Hour and Day", fontweight='bold')
    plt.tight_layout();
    safe_savefig('heatmap_2d_time_analysis.png');
    plt.close()

    hm_days = df_hm.groupby('weekday').size().reindex(range(7), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([days_map[i] for i in hm_days.index], hm_days.values,
           color=plt.cm.YlOrRd(plt.Normalize(hm_days.min(), hm_days.max())(hm_days.values)), edgecolor='black')
    ax.set_title("Cumulative Alert Frequency by Day of the Week", fontweight='bold')
    plt.tight_layout();
    safe_savefig('heatmap_1d_days.png');
    plt.close()

    hm_hours = df_hm.groupby('hour').size().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hm_hours.index, hm_hours.values,
           color=plt.cm.YlOrRd(plt.Normalize(hm_hours.min(), hm_hours.max())(hm_hours.values)), edgecolor='black')
    ax.set_xticks(range(24));
    ax.set_xlabel('Hour of the Day (UTC)')
    ax.set_title("Cumulative Alert Frequency by Hour of the Day", fontweight='bold')
    plt.tight_layout();
    safe_savefig('heatmap_1d_hours.png');
    plt.close()


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    print("[1/3] Fetching dataset from repository...")
    url = "https://raw.githubusercontent.com/Vadimkin/ukrainian-air-raid-sirens-dataset/main/datasets/official_data_en.csv"
    try:
        df = pd.read_csv(url, low_memory=False)
    except Exception as e:
        return print(f"CRITICAL ERROR: {e}")

    print("[2/3] Preprocessing (Filtering & Formatting)...")
    col_map = {'region': ['region_title', 'region', 'region_name', 'oblast', 'title'],
               'start': ['started_at', 'start_time', 'start', 'alert_start'],
               'end': ['finished_at', 'end_time', 'end', 'alert_end']}

    def find_col(target):
        return next(col for col in col_map[target] if col in df.columns)

    df = df.rename(
        columns={find_col('region'): 'region_name', find_col('start'): 'started_at', find_col('end'): 'finished_at'})

    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce', utc=True)
    df = df[df['started_at'] <= pd.to_datetime(GLOBAL_MAX_DATE, utc=True)]
    df['finished_at'] = pd.to_datetime(df['finished_at'], errors='coerce', utc=True).fillna(
        df['started_at'] + pd.Timedelta(minutes=30))
    df = df.dropna(subset=['started_at'])

    df['duration_raw'] = (df['finished_at'] - df['started_at']).dt.total_seconds() / 3600.0
    df = df[(df['duration_raw'] >= 0) & (df['duration_raw'] < 48)]

    def is_macro(name):
        return not any(x in str(name).lower() for x in ['raion', 'hromada', 'community'])

    df = df[df['region_name'].apply(is_macro)]

    # FIX: Apply the map_matcher BEFORE deduplication. This ensures overlapping variants
    # (e.g. "Kyiv" and "Kyiv City") are merged into a single interval maxing out at 24 hours/day.
    print("      -> Standardizing Region Names...")
    df['region_name'] = df['region_name'].apply(map_matcher)

    print("      -> Applying Universal Deduplication Engine...")
    clean_national_df = deduplicate_to_intervals(df)

    # FIX: Update filter to use the mapped key
    kyiv_city_df = clean_national_df[clean_national_df['region_name'] == 'kyiv_city']

    print("[3/3] Generating Dashboards & Analytics...")
    generate_temporal_dashboards(clean_national_df, "National Level (All Regions Summed)", "national_forecast")
    generate_temporal_dashboards(kyiv_city_df, "Capital Level (Kyiv City Only)", "kyiv_city_forecast")

    generate_geospatial_heatmaps(clean_national_df)
    generate_time_heatmaps(clean_national_df)

    try:
        yearly = clean_national_df.groupby(['year', 'region_name'])['duration_hours'].sum().unstack(fill_value=0)
        top = yearly.sum(axis=0).sort_values(ascending=False).head(15).index
        ax = yearly[top].plot(kind='bar', figsize=(16, 8), width=0.8, colormap='tab20')
        plt.title('Total Alert Hours per Year by Oblast (Top 15)', fontsize=16, fontweight='bold')
        plt.ylabel('Total Hours (Summed over Year)')
        plt.xlabel('Year')
        plt.legend(title='Oblast', bbox_to_anchor=(1.01, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        safe_savefig("regional_hours_yearly.png")
        plt.close()
    except Exception as e:
        print(f"Failed to plot regional chart: {e}")

    print("\nPipeline Complete! Dashboards and Heatmaps successfully generated.")


if __name__ == "__main__": main()