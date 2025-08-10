import os
import math
import time
import random
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from dotenv import load_dotenv

# =========================
# Parámetros (ajústalos aquí)
# =========================
# EV mínimo por mercado (no-vig con Pinnacle)
EV_MIN_H2H = 0.04          # 4% para H2H
EV_MIN_LINES = 0.05        # 5% para Totals/Spreads

# Rango de cuotas y probabilidad mínima (anti-longshots)
ODDS_MIN = 1.25
ODDS_MAX_H2H = 3.00
ODDS_MAX_LINES = 2.50      # totals/spreads suelen rondar 1.80–2.20
P_FAIR_MIN = 0.30          # prob. justa mínima (30%)

# Frescura (min desde last_update de la casa SOFT). 0 = desactivar
FRESH_MAX_MIN = 120

# Ventana de evento
EVENT_MAX_DAYS = 30

# Diferencia mínima soft - Pinnacle (0 = no exigir; el EV ya controla)
MIN_DIFF_ODDS = 0.00

# Umbral “consenso” anti-outliers entre softs
OUTLIER_DIFF_CAP     = 0.15   # best_soft − second_best > 0.15 ⇒ sospechoso
CONSENSUS_RATIO_CAP  = 1.12   # best_soft / mediana_soft > 1.12 ⇒ sospechoso
EV_EXTRA_FOR_OUTLIER = 0.03   # permitir outlier solo si EV ≥ (umbral + 3pp)

# Kelly / banca (en unidades)
KELLY_FRACTION = 0.25      # 25%
BANKROLL_UNITS = 100.0     # tú decides cuántos € es 1u
STAKE_MAX_UNITS = 10.0

# Comisión por casa (aplicada a la cuota para EV)
COMMISSION_BY_BOOK = {
    "betfair_ex_eu": 0.05,   # Betfair Exchange EU 5%
    # "matchbook": 0.04,     # Actívala si la añades a SOFT_BOOKIES_TARGET
}

# HTTP
TIMEOUT = 10
MAX_RETRIES = 5
CHUNK_SOFT_LIMIT = 3500

# Archivos / TZ
CSV_FILENAME = "valuebets_resultados.csv"
TZ_LOCAL = ZoneInfo("Europe/Madrid")

# =========================
# Mapeo de ligas bonitas
# =========================
LEAGUE_MAP = {
    # Football (claves v4 correctas)
    "soccer_epl": "Premier League (ENG)",
    "soccer_efl_champ": "Championship (ENG)",
    "soccer_spain_la_liga": "LaLiga EA Sports (ESP)",
    "soccer_spain_segunda_division": "LaLiga Hypermotion (ESP)",
    "soccer_italy_serie_a": "Serie A (ITA)",
    "soccer_italy_serie_b": "Serie B (ITA)",
    "soccer_germany_bundesliga": "Bundesliga (GER)",
    "soccer_germany_bundesliga2": "2. Bundesliga (GER)",
    "soccer_france_ligue_one": "Ligue 1 (FRA)",
    "soccer_france_ligue_two": "Ligue 2 (FRA)",
    "soccer_portugal_primeira_liga": "Primeira Liga (POR)",
    "soccer_netherlands_eredivisie": "Eredivisie (NED)",
    "soccer_turkey_super_league": "Süper Lig (TUR)",
    "soccer_usa_mls": "MLS (USA)",
    "soccer_uefa_champs_league": "UEFA Champions League",
    "soccer_uefa_europa_league": "UEFA Europa League",
    "soccer_conmebol_copa_libertadores": "Copa Libertadores",
    "soccer_brazil_campeonato": "Brasileirão Série A (BRA)",
    # Basketball
    "basketball_nba": "NBA (USA)",
    "basketball_euroleague": "EuroLeague",
    # Tennis (dinámico por torneo, se añadirá tal cual)
}

# =========================
# Deporte y casas objetivo
# =========================
PINNACLE_KEY = "pinnacle"  # sharp

# Softs disponibles en EU (según tu lista detectada)
SOFT_BOOKIES_TARGET = {
    "sport888", "betfair_ex_eu", "williamhill", "marathonbet",
    "betclic_fr", "unibet_fr", "unibet_it", "unibet_nl",
    "winamax_fr", "winamax_de", "betsson", "nordicbet",
    "coolbet", "parionssport_fr", "tipico_de", "suprabets",
    # "matchbook",  # activa si quieres usarla (añade comisión en COMMISSION_BY_BOOK)
}

BOOKIE_LABEL = {
    "pinnacle": "Pinnacle",
    "sport888": "888sport",
    "betfair_ex_eu": "Betfair EX",
    "williamhill": "William Hill",
    "marathonbet": "Marathonbet",
    "betclic_fr": "Betclic",
    "unibet_fr": "Unibet FR",
    "unibet_it": "Unibet IT",
    "unibet_nl": "Unibet NL",
    "winamax_fr": "Winamax FR",
    "winamax_de": "Winamax DE",
    "betsson": "Betsson",
    "nordicbet": "NordicBet",
    "coolbet": "Coolbet",
    "parionssport_fr": "ParionsSport",
    "tipico_de": "Tipico DE",
    "suprabets": "Suprabets",
    # "matchbook": "Matchbook",
}

markets = ["h2h", "totals", "spreads"]

# =========================
# Utilidades
# =========================
def fmt_dec(x: float, nd=2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.{nd}f}".replace(".", ",")

def fmt_pct(x: float, nd=2) -> str:
    return f"{(x*100):.{nd}f}%".replace(".", ",")

def to_local_str(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_LOCAL).strftime("%d/%m/%Y %H:%M")

def parse_iso(iso_str: str) -> datetime:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def minutes_since(iso_str: str) -> float:
    try:
        dt = parse_iso(iso_str)
        now = datetime.now(timezone.utc)
        return (now - dt).total_seconds() / 60.0
    except Exception:
        return float("inf")

def normalize_point(market: str, point):
    if market == "h2h":
        return None
    if point is None:
        return None
    try:
        p = float(point)
    except Exception:
        return None
    p = round(p, 2)
    if market == "spreads":
        p = round(abs(p), 2)
    return p

def normalize_option(market: str, outcome_name: str, home_team: str, away_team: str):
    name = (outcome_name or "").strip().lower()
    ht = (home_team or "").strip().lower()
    at = (away_team or "").strip().lower()
    if market == "h2h":
        if name in {"draw", "empate", "x"}:
            return "DRAW", "Empate"
        if name in {"home", "local"} or (ht and name == ht):
            return "HOME", "Local"
        if name in {"away", "visitante"} or (at and name == at):
            return "AWAY", "Visitante"
        if ht and ht in name:
            return "HOME", "Local"
        if at and at in name:
            return "AWAY", "Visitante"
        return None, outcome_name
    if market == "totals":
        if name.startswith("over") or name == "over":
            return "OVER", "Más"
        if name.startswith("under") or name == "under":
            return "UNDER", "Menos"
        return None, outcome_name
    if market == "spreads":
        if name in {"home", "local"} or (ht and name == ht):
            return "HOME", "Local"
        if name in {"away", "visitante"} or (at and name == at):
            return "AWAY", "Visitante"
        if ht and ht in name:
            return "HOME", "Local"
        if at and at in name:
            return "AWAY", "Visitante"
        return None, outcome_name
    return None, outcome_name

def market_label(market: str) -> str:
    return {"h2h": "1X2", "totals": "Totales", "spreads": "Hándicap"}.get(market, market)

def backoff_sleep(attempt: int):
    base = 0.5 * (2 ** attempt)
    jitter = random.uniform(0, 0.3)
    time.sleep(base + jitter)

def effective_odds(bookie_key: str, odds: float) -> float:
    """Ajusta la cuota por comisión si la casa está en COMMISSION_BY_BOOK."""
    if odds and odds > 1:
        fee = COMMISSION_BY_BOOK.get(bookie_key, 0.0)
        if fee > 0:
            return 1 + (odds - 1) * (1 - fee)
    return odds

# =========================
# Cargar entorno
# =========================
load_dotenv()
api_key = os.getenv("ODDS_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_user_id = os.getenv("TELEGRAM_CHAT_ID")

if not all([api_key, telegram_token, telegram_user_id]):
    print("❌ Faltan variables en el archivo .env")
    raise SystemExit(1)
print("✅ API key cargada correctamente\n")

# =========================
# HTTP con reintentos
# =========================
session = requests.Session()
API_HEADERS_SNAPSHOT = []

def request_with_retries(url, params):
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"⚠️  Error de red: {e}. Reintento {attempt+1}/{MAX_RETRIES}...")
            backoff_sleep(attempt)
            continue
        API_HEADERS_SNAPSHOT.append({
            "remaining": resp.headers.get("x-requests-remaining"),
            "used": resp.headers.get("x-requests-used"),
            "ts": datetime.now(TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S"),
            "status": resp.status_code
        })
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503, 504):
            print(f"⏳ HTTP {resp.status_code}. Reintento {attempt+1}/{MAX_RETRIES}...")
            backoff_sleep(attempt)
            continue
        print(f"❌ Error HTTP {resp.status_code}: {resp.text[:200]}")
        return None
    print("❌ Máximo de reintentos alcanzado.")
    return None

# =========================
# Descubrir deportes/torneos activos
# =========================
def discover_sports(api_key: str):
    url = "https://api.the-odds-api.com/v4/sports/"
    r = request_with_retries(url, {"apiKey": api_key})
    if not r:
        print("⚠️ No se pudo descubrir deportes. Usaré lista base.")
        return set()
    try:
        data = r.json()
    except Exception:
        print("⚠️ Respuesta no JSON en /v4/sports. Usaré lista base.")
        return set()
    return {s.get("key") for s in data if s.get("key")}

active_keys = discover_sports(api_key)

# Base fútbol (claves correctas v4)
base_soccer = {
    "soccer_epl", "soccer_efl_champ",
    "soccer_spain_la_liga", "soccer_spain_segunda_division",
    "soccer_italy_serie_a", "soccer_italy_serie_b",
    "soccer_germany_bundesliga", "soccer_germany_bundesliga2",
    "soccer_france_ligue_one", "soccer_france_ligue_two",
    "soccer_portugal_primeira_liga",
    "soccer_netherlands_eredivisie",
    "soccer_turkey_super_league",
    "soccer_usa_mls",
    "soccer_uefa_champs_league", "soccer_uefa_europa_league",
    "soccer_conmebol_copa_libertadores",
    "soccer_brazil_campeonato",
}
basket = {"basketball_nba", "basketball_euroleague"}
tennis_dyn = {k for k in active_keys if k.startswith(("tennis_atp_", "tennis_wta_"))}

# Solo mantener los que estén activos para evitar 404
sports = list((base_soccer | basket | tennis_dyn) & (active_keys if active_keys else base_soccer | basket))
print("🏷️  Deportes/torneos a consultar:", len(sports))

# =========================
# Scrapeo
# =========================
rows = []
bookies_seen = set()

for sport in sports:
    print(f"📦 Procesando: {sport}")
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu",            # mantenemos EU
        "markets": ",".join(markets),
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    resp = request_with_retries(url, params)
    if not resp:
        print(f"❌ Error en {sport}. Siguiente.")
        continue
    try:
        data = resp.json()
    except ValueError:
        print(f"❌ Respuesta no JSON en {sport}.")
        continue

    for match in data:
        home_team = match.get("home_team")
        away_team = match.get("away_team")
        fecha_iso = match.get("commence_time")
        evento = f"{home_team} vs {away_team}"

        for bookmaker in match.get("bookmakers", []):
            casa = bookmaker.get("key")
            bookies_seen.add(casa)
            if casa not in {PINNACLE_KEY} | SOFT_BOOKIES_TARGET:
                continue
            book_last_update = bookmaker.get("last_update")

            for market in bookmaker.get("markets", []):
                mk = market.get("key")
                if mk not in markets:
                    continue

                for outcome in market.get("outcomes", []):
                    nombre = outcome.get("name")
                    price = outcome.get("price")
                    point = outcome.get("point")
                    opt_std, opt_label = normalize_option(mk, nombre, home_team, away_team)
                    if opt_std is None:
                        continue
                    point_norm = normalize_point(mk, point)

                    rows.append({
                        "liga": sport,
                        "evento": evento,
                        "fecha": fecha_iso,
                        "home_team": home_team,
                        "away_team": away_team,
                        "mercado": mk,
                        "opcion_raw": nombre,
                        "opcion_std": opt_std,
                        "opcion_lbl": opt_label,
                        "point": point,
                        "point_norm": point_norm,
                        "casa": casa,
                        "cuota": price,
                        "book_last_update": book_last_update
                    })

print("\n👀 Bookies detectadas en EU:", ", ".join(sorted(bookies_seen)))
print("✅ Cuotas recolectadas. Procesando...\n")
if not rows:
    print("⚠️  No se recolectaron cuotas. Saliendo.")
    raise SystemExit(0)

df = pd.DataFrame(rows)
print(f"🔍 eventos únicos: {df[['liga','evento']].drop_duplicates().shape[0]}")
print(f"   filas totales: {len(df)} | Pinnacle: {(df['casa']==PINNACLE_KEY).sum()} | SOFTS: {(df['casa'].isin(SOFT_BOOKIES_TARGET)).sum()}")

# =========================
# No-vig con Pinnacle (+ validación 3-way en soccer h2h)
# =========================
df_pin = df[df["casa"] == PINNACLE_KEY].copy()
if df_pin.empty:
    print("⚠️  No hay Pinnacle. No se puede calcular p_fair.")
    raise SystemExit(0)

gkeys = ["liga", "evento", "fecha", "mercado", "point_norm", "home_team", "away_team"]

# Asegurar que en fútbol 1X2 haya 3 opciones (HOME/AWAY/DRAW)
df_pin["n_opts"] = df_pin.groupby(gkeys)["opcion_std"].transform("nunique")
mask_threeway = ~((df_pin["mercado"] == "h2h") & (df_pin["liga"].str.startswith("soccer_")) & (df_pin["n_opts"] < 3))
df_pin = df_pin[mask_threeway].copy()

df_pin["imp"] = 1.0 / df_pin["cuota"]
grp = df_pin.groupby(gkeys, dropna=False)
df_pin["sum_imp"] = grp["imp"].transform("sum")
df_pin["cnt_opt"] = grp["imp"].transform("count")
df_pin = df_pin[df_pin["cnt_opt"] >= 2].copy()
df_pin["p_fair"] = df_pin["imp"] / df_pin["sum_imp"]

# =========================
# Softs: mejor cuota efectiva (ajuste comisiones) + consenso
# =========================
df_soft = df[df["casa"].isin(SOFT_BOOKIES_TARGET)].copy()
if df_soft.empty:
    print("⚠️  No hay casas soft en EU de la lista. Saliendo.")
    raise SystemExit(0)

df_soft["odds_eff"] = df_soft.apply(lambda r: effective_odds(r["casa"], r["cuota"]), axis=1)

# Estadísticas de consenso por grupo (para detectar outliers)
keys_soft = gkeys + ["opcion_std"]

def second_best(s):
    return s.nlargest(2).min() if len(s) >= 2 else float("nan")

soft_stats = (
    df_soft.groupby(keys_soft, dropna=False)["odds_eff"]
           .agg(n_soft="size", median_soft="median", second_best=second_best)
           .reset_index()
)

# Mejor soft por grupo
soft_group = df_soft.groupby(keys_soft, dropna=False)
idx_max = soft_group["odds_eff"].idxmax()
best_soft = df_soft.loc[idx_max].copy().rename(columns={
    "point": "point_soft",
    "cuota": "odds_soft_raw",
    "odds_eff": "odds_soft",
    "book_last_update": "last_update_soft",
    "casa": "best_bookie"
})

# Añadir stats de consenso a la mejor soft
best_soft = best_soft.merge(soft_stats, on=keys_soft, how="left")

# =========================
# Merge Pinnacle vs mejor SOFT
# =========================
pin_slim = df_pin[gkeys + ["opcion_std", "opcion_lbl", "point", "cuota", "p_fair", "book_last_update"]].rename(columns={
    "point": "point_pin",
    "cuota": "odds_pinnacle",
    "book_last_update": "last_update_pin"
})

merged = pd.merge(
    pin_slim,
    best_soft[gkeys + ["opcion_std", "point_soft", "odds_soft_raw", "odds_soft", "last_update_soft", "best_bookie", "n_soft", "median_soft", "second_best"]],
    on=gkeys + ["opcion_std"],
    how="inner",
    validate="m:1"
)

print(f"   emparejadas (misma línea/opción): {len(merged)}")
if merged.empty:
    print("⚠️  No hay intersección Pinnacle vs softs. Saliendo.")
    raise SystemExit(0)

# =========================
# EV, Kelly y umbral escalonado por cuota
# =========================
merged["EV"] = (merged["odds_soft"] * merged["p_fair"]) - 1.0

# Kelly fraccional (solo si EV>0)
b = merged["odds_soft"] - 1.0
p = merged["p_fair"]
q = 1.0 - p
kelly_raw = (b * p - q) / b
kelly_raw = kelly_raw.where(merged["EV"] > 0, 0.0)
kelly_frac = (kelly_raw.clip(lower=0.0) * KELLY_FRACTION).fillna(0.0)
merged["stake_units"] = (kelly_frac * BANKROLL_UNITS).clip(upper=STAKE_MAX_UNITS).round(2)

# Auxiliares
merged["liga_pretty"] = merged["liga"].map(lambda x: LEAGUE_MAP.get(x, x))
merged["fecha_dt"] = merged["fecha"].apply(parse_iso)
merged["mins_since_soft"] = merged["last_update_soft"].apply(lambda s: minutes_since(s) if pd.notna(s) else float("inf"))
merged["odds_diff"] = merged["odds_soft"] - merged["odds_pinnacle"]

def ev_min_for_market(mk):
    return EV_MIN_H2H if mk == "h2h" else EV_MIN_LINES

def ev_min_row(base_ev: float, odds_soft: float) -> float:
    # EV mínimo escalonado por tramos de cuota (más alta → exigimos más EV)
    if pd.isna(odds_soft):
        return base_ev
    if odds_soft <= 2.20:
        bump = 0.00
    elif odds_soft <= 3.00:
        bump = 0.01   # +1%
    else:
        bump = 0.02   # +2%
    return base_ev + bump

merged["ev_min_req_base"] = merged["mercado"].apply(ev_min_for_market)
merged["ev_min_req"] = merged.apply(lambda r: ev_min_row(r["ev_min_req_base"], r["odds_soft"]), axis=1)
merged["odds_max_req"] = merged["mercado"].apply(lambda m: ODDS_MAX_H2H if m == "h2h" else ODDS_MAX_LINES)

# Señales de outlier
merged["best_minus_second"] = merged["odds_soft"] - merged["second_best"]
merged["consensus_ratio"]  = merged["odds_soft"] / merged["median_soft"]

# =========================
# Filtros
# =========================
mask = (
    (merged["EV"] >= merged["ev_min_req"]) &
    (merged["p_fair"] >= P_FAIR_MIN) &
    (merged["odds_soft"] >= ODDS_MIN) &
    (merged["odds_soft"] <= merged["odds_max_req"]) &
    (merged["fecha_dt"] >= datetime.now(timezone.utc)) &
    (merged["fecha_dt"] <= datetime.now(timezone.utc) + timedelta(days=EVENT_MAX_DAYS))
)

# Frescura (si está activada)
if FRESH_MAX_MIN and FRESH_MAX_MIN > 0:
    mask &= (merged["mins_since_soft"] <= FRESH_MAX_MIN)

# Delta mínimo contra Pinnacle (si lo exiges)
if MIN_DIFF_ODDS and MIN_DIFF_ODDS > 0:
    mask &= (merged["odds_diff"] >= MIN_DIFF_ODDS)

# Filtro de consenso anti-outliers (solo si hay ≥2 softs)
mask_consensus = ~(
    (merged["n_soft"] >= 2) & (
        (merged["best_minus_second"] > OUTLIER_DIFF_CAP) |
        (merged["consensus_ratio"]  > CONSENSUS_RATIO_CAP)
    ) & (merged["EV"] < merged["ev_min_req"] + EV_EXTRA_FOR_OUTLIER)
)
mask &= mask_consensus

valuebets = merged[mask].copy().sort_values(by="EV", ascending=False)
print(f"   EV ≥ umbral tras filtros: {len(valuebets)} picks")

# =========================
# CSV
# =========================
def export_csv(df_out: pd.DataFrame, filename: str):
    if df_out.empty:
        top = merged.sort_values(by="EV", ascending=False).head(8).copy()
        top["fecha_local"] = top["fecha"].apply(to_local_str)
        top["updated_at"] = datetime.now(TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S")
        cols = [
            "liga_pretty","evento","fecha_local","mercado","point_norm",
            "opcion_std","opcion_lbl",
            "odds_pinnacle","odds_soft","odds_soft_raw","odds_diff",
            "median_soft","second_best","best_minus_second","consensus_ratio",
            "p_fair","EV","stake_units",
            "best_bookie","home_team","away_team","point_pin","point_soft",
            "last_update_soft","updated_at"
        ]
        top[cols].to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"💾 CSV (top inspección) guardado como {filename}")
        return
    df_out["fecha_local"] = df_out["fecha"].apply(to_local_str)
    df_out["updated_at"] = datetime.now(TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S")
    cols = [
        "liga_pretty","evento","fecha_local","mercado","point_norm",
        "opcion_std","opcion_lbl",
        "odds_pinnacle","odds_soft","odds_soft_raw","odds_diff",
        "median_soft","second_best","best_minus_second","consensus_ratio",
        "p_fair","EV","stake_units",
        "best_bookie","home_team","away_team","point_pin","point_soft",
        "last_update_soft","updated_at"
    ]
    df_out[cols].to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"💾 CSV guardado como {filename}")

export_csv(valuebets, CSV_FILENAME)

# =========================
# Telegram
# =========================
def format_pick_row(row) -> str:
    liga = row["liga_pretty"]
    evento = row["evento"]
    fecha = to_local_str(row["fecha"])
    mk = market_label(row["mercado"])

    linea_txt = ""
    if row["mercado"] == "totals":
        pt = row["point_soft"] if pd.notna(row["point_soft"]) else row["point_pin"]
        if pd.notna(pt): linea_txt = f" {fmt_dec(float(pt))}"
    elif row["mercado"] == "spreads":
        pt = row["point_soft"] if pd.notna(row["point_soft"]) else row["point_pin"]
        if pd.notna(pt):
            signo = "+" if float(pt) > 0 else ""
            linea_txt = f" {signo}{fmt_dec(abs(float(pt)))}" if float(pt) != 0 else " 0"

    opcion_lbl = row["opcion_lbl"]
    if row["mercado"] == "h2h":
        if row["opcion_std"] == "HOME":
            opcion_lbl = f"{row['home_team']} (Local)"
        elif row["opcion_std"] == "AWAY":
            opcion_lbl = f"{row['away_team']} (Visitante)"

    best_label = BOOKIE_LABEL.get(row["best_bookie"], row["best_bookie"])
    odds_pin = fmt_dec(row["odds_pinnacle"])
    odds_soft_raw = fmt_dec(row["odds_soft_raw"])
    odds_soft_eff = fmt_dec(row["odds_soft"])
    diff = fmt_dec(row["odds_diff"], nd=2)
    ev_pct = fmt_pct(row["EV"])
    stake = fmt_dec(row["stake_units"], nd=2)
    mins = int(row["mins_since_soft"]) if pd.notna(row["mins_since_soft"]) else -1
    freshness = f"{mins} min" if mins >= 0 else "n/d"

    parts = [
        f"🏆 {liga}",
        f"{evento} — {fecha}",
        f"🎯 {mk}{linea_txt} · {opcion_lbl}",
        f"🔝 Mejor casa: {best_label}",
        f"🔢 Pinnacle: {odds_pin} | {best_label}: {odds_soft_raw} (efectiva {odds_soft_eff}) (Δ {diff})",
        f"📈 EV: {ev_pct} | Stake (Kelly {int(KELLY_FRACTION*100)}%): {stake} u",
        f"⏱️ Actualizado: {freshness}"
    ]
    # Marca visual si era “muy fuera de consenso”
    if pd.notna(row.get("best_minus_second")) and pd.notna(row.get("consensus_ratio")):
        if (row.get("n_soft", 0) >= 2) and (
            (row["best_minus_second"] > OUTLIER_DIFF_CAP) or (row["consensus_ratio"] > CONSENSUS_RATIO_CAP)
        ):
            parts.append("⚠️ Posible outlier vs consenso")
    return "\n".join(parts)

def build_messages(df: pd.DataFrame):
    header = f"🎯 {len(df)} apuestas con valor (H2H ≥ {int(EV_MIN_H2H*100)}% · Líneas ≥ {int(EV_MIN_LINES*100)}%):\n\n"
    messages = []
    current = header
    for _, row in df.iterrows():
        block = format_pick_row(row) + "\n" + ("─" * 20) + "\n"
        if len(current) + len(block) > CHUNK_SOFT_LIMIT:
            messages.append(current.rstrip())
            current = block
        else:
            current += block
    if current.strip():
        messages.append(current.rstrip())
    return messages

def send_telegram_messages(token: str, chat_id: str, texts):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    ok = True
    for idx, text in enumerate(texts, start=1):
        try:
            r = session.post(url, data={"chat_id": chat_id, "text": text}, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"❌ Error enviando a Telegram (parte {idx}): {e}")
            ok = False
            continue
        if r.status_code != 200:
            print(f"❌ Telegram {r.status_code}: {r.text[:200]}")
            ok = False
        else:
            print(f"📬 Mensaje enviado a Telegram (parte {idx}/{len(texts)}).")
    return ok

if valuebets.empty:
    send_telegram_messages(telegram_token, telegram_user_id, [
        "🎯 0 apuestas con valor con los filtros actuales. Prueba en horas de mercado activo o ajusta límites/EV."
    ])
else:
    mensajes = build_messages(valuebets)
    send_telegram_messages(telegram_token, telegram_user_id, mensajes)

# =========================
# Resumen de cuota API
# =========================
if API_HEADERS_SNAPSHOT:
    last = API_HEADERS_SNAPSHOT[-1]
    print("\n📊 Resumen cuota API (última respuesta):")
    print(f"   x-requests-remaining: {last.get('remaining')}")
    print(f"   x-requests-used:      {last.get('used')}")
else:
    print("\nℹ️ No se pudieron leer headers de cuota de la API.")