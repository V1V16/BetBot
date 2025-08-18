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
# Parámetros (ajustables)
# =========================
# EV mínimo por mercado (no-vig con Pinnacle)
EV_MIN_H2H   = 0.03   # 3% para 1X2
EV_MIN_LINES = 0.04   # 4% para Totales/Hándicap

# Extra de EV si la métrica no es “goles” (para cuando actives tarjetas/córners)
EV_EXTRA_BY_METRIC = {
    "goals":   0.00,
    "cards":   0.02,
    "corners": 0.02
}

# Rango de cuotas y prob. mínima (descarta rarezas)
ODDS_MIN        = 1.25
ODDS_MAX_H2H    = 3.25
ODDS_MAX_LINES  = 2.75
P_FAIR_MIN      = 0.25    # prob. justa mínima del pick

# Frescura de la cuota soft (última actualización)
#   "off"     -> sin filtro
#   "static"  -> corte fijo (FRESH_MAX_MIN)
#   "dynamic" -> ventana en minutos según el tiempo que falta al evento (RECOM.)
FRESH_POLICY   = "dynamic"
FRESH_MAX_MIN  = 180      # si usas "static"

# Ventana de eventos futura (no mirar más allá de X días)
EVENT_MAX_DAYS = 30

# Diferencia mínima soft − Pinnacle (normalmente 0: el EV ya lo controla)
MIN_DIFF_ODDS = 0.00

# Anti-outliers por consenso entre softs (sólo si n_soft ≥ 3)
OUTLIER_DIFF_CAP     = 0.20  # best_soft − second_best > 0.20 ⇒ sospechoso
CONSENSUS_RATIO_CAP  = 1.18  # best_soft / mediana_soft > 1.18 ⇒ sospechoso
EV_EXTRA_FOR_OUTLIER = 0.03  # permitir si el EV supera el umbral por +3pp

# Kelly fraccional / banca (en “unidades”, define tú el valor € de 1u)
KELLY_FRACTION   = 0.25
BANKROLL_UNITS   = 100.0
STAKE_MAX_UNITS  = 10.0

# Comisiones de casas (aplicadas a la cuota para EV)
COMMISSION_BY_BOOK = {
    "betfair_ex_eu": 0.05,  # Betfair Exchange EU 5%
    # "matchbook": 0.04,
}

# HTTP
TIMEOUT          = 10
MAX_RETRIES      = 5
CHUNK_SOFT_LIMIT = 3500     # < 4096 Telegram

# Archivos / TZ
CSV_FILENAME = "valuebets_resultados.csv"
TZ_LOCAL     = ZoneInfo("Europe/Madrid")

# =========================
# Módulo opcional: Tarjetas y Córners
# =========================
ENABLE_CARDS_CORNERS       = False   # pon True si quieres intentarlo
EXTRA_MAX_EVENTS_PER_SPORT = 400
EXTRA_EVENTS_MAX_DAYS      = 45

# Markets de extras (si el feed los ofrece para ese evento/casa)
EXTRA_MARKETS_RAW = [
    "totals_cards", "spreads_cards",
    "alternate_totals_cards", "alternate_spreads_cards",
    "totals_corners", "spreads_corners",
    "alternate_totals_corners", "alternate_spreads_corners",
]

def classify_extra_market(market_key: str):
    mk = market_key.lower()
    metric = "cards" if "cards" in mk else ("corners" if "corners" in mk else "goals")
    base = "totals" if "totals" in mk else ("spreads" if "spreads" in mk else None)
    return base, metric

# =========================
# Etiquetas de ligas (para mostrar)
# =========================
LEAGUE_MAP = {
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
    "basketball_nba": "NBA (USA)",
    "basketball_euroleague": "EuroLeague",
}

# =========================
# Casas & markets
# =========================
PINNACLE_KEY = "pinnacle"  # sharp

SOFT_BOOKIES_TARGET = {
    "sport888", "betfair_ex_eu", "williamhill", "marathonbet",
    "betclic_fr", "unibet_fr", "unibet_it", "unibet_nl", "unibet_uk",
    "winamax_fr", "winamax_de", "betsson", "nordicbet",
    "coolbet", "parionssport_fr", "tipico_de", "suprabets",
    "skybet", "ladbrokes_uk", "paddypower", "coral", "betvictor",
    "betway", "casumo", "leovegas", "virginbet", "betonlineag", "everygame",
    "boylesports", "grosvenor", "livescorebet", "betanysports", "gtbets",
    # "matchbook",
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
    "unibet_uk": "Unibet UK",
    "winamax_fr": "Winamax FR",
    "winamax_de": "Winamax DE",
    "betsson": "Betsson",
    "nordicbet": "NordicBet",
    "coolbet": "Coolbet",
    "parionssport_fr": "ParionsSport",
    "tipico_de": "Tipico DE",
    "suprabets": "Suprabets",
    "skybet": "Sky Bet",
    "ladbrokes_uk": "Ladbrokes",
    "paddypower": "PaddyPower",
    "coral": "Coral",
    "betvictor": "BetVictor",
    "betway": "Betway",
    "casumo": "Casumo",
    "leovegas": "LeoVegas",
    "virginbet": "Virgin Bet",
    "betonlineag": "BetOnline",
    "everygame": "Everygame",
    "boylesports": "BoyleSports",
    "grosvenor": "Grosvenor",
    "livescorebet": "LiveScore Bet",
    "betanysports": "BetAnySports",
    "gtbets": "GTbets",
}

markets_base = ["h2h", "totals", "spreads"]

# =========================
# Utils
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

def market_label(market: str, metric: str = "goals") -> str:
    base = {"h2h": "1X2", "totals": "Totales", "spreads": "Hándicap"}.get(market, market)
    if metric == "cards":
        return base + " (Tarjetas)"
    if metric == "corners":
        return base + " (Córners)"
    return base

def metric_label(metric: str) -> str:
    return {"goals":"Goles","cards":"Tarjetas","corners":"Córners"}.get(metric, metric)

def backoff_sleep(attempt: int):
    base = 0.5 * (2 ** attempt)
    jitter = random.uniform(0, 0.3)
    time.sleep(base + jitter)

def effective_odds(bookie_key: str, odds: float) -> float:
    if odds and odds > 1:
        fee = COMMISSION_BY_BOOK.get(bookie_key, 0.0)
        if fee > 0:
            return 1 + (odds - 1) * (1 - fee)
    return odds

def allowed_staleness_minutes(tti_hours: float) -> int:
    if tti_hours > 7*24:   return 7*24*60     # >7 días  → 7 días
    if tti_hours > 72:     return 48*60       # 3–7 días → 48 h
    if tti_hours > 12:     return 6*60        # 12–72 h  → 6 h
    if tti_hours > 2:      return 60          # 2–12 h   → 60 min
    return 15                                 # ≤2 h     → 15 min

# =========================
# Entorno
# =========================
load_dotenv()
api_key        = os.getenv("ODDS_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_user_id = os.getenv("TELEGRAM_CHAT_ID")

if not all([api_key, telegram_token, telegram_user_id]):
    print("❌ Faltan variables en el archivo .env")
    raise SystemExit(1)

print("✅ API key cargada correctamente\n")
print(f"⚙️  CONFIG → EV_MIN_H2H={EV_MIN_H2H:.2%}  EV_MIN_LINES={EV_MIN_LINES:.2%}  "
      f"P_FAIR_MIN={P_FAIR_MIN:.0%}  FRESH_POLICY={FRESH_POLICY}  FRESH_MAX_MIN={FRESH_MAX_MIN}  "
      f"ENABLE_CARDS_CORNERS={ENABLE_CARDS_CORNERS}")

# =========================
# HTTP con reintentos + autorregulación
# =========================
session = requests.Session()
API_HEADERS_SNAPSHOT = []

def maybe_throttle():
    if not API_HEADERS_SNAPSHOT:
        return
    last = API_HEADERS_SNAPSHOT[-1]
    rem = last.get("remaining")
    try:
        rem = int(rem) if rem is not None else None
    except Exception:
        rem = None
    if rem is not None and rem < 300:
        time.sleep(1.0)  # afloja si estás bajo de cuota

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
            maybe_throttle()
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

sports = list((base_soccer | basket | tennis_dyn) & (active_keys if active_keys else base_soccer | basket))
print("🏷️  Deportes/torneos a consultar:", len(sports))

# =========================
# Scrapeo (markets base)
# =========================
rows = []
bookies_seen = set()
event_ids_by_sport = {}

for sport in sports:
    print(f"📦 Procesando: {sport}")
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
    params = {
        "apiKey": api_key,
        "regions": "eu,uk",
        "markets": ",".join(markets_base),
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

    ev_ids = []
    for match in data:
        event_id = match.get("id")
        if event_id:
            ev_ids.append(event_id)

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
                if mk not in markets_base:
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
                        "event_id": event_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "mercado": mk,          # h2h/totals/spreads
                        "metric": "goals",      # por defecto
                        "opcion_raw": nombre,
                        "opcion_std": opt_std,
                        "opcion_lbl": opt_label,
                        "point": point,
                        "point_norm": point_norm,
                        "casa": casa,
                        "cuota": price,
                        "book_last_update": book_last_update
                    })
    if ev_ids:
        event_ids_by_sport[sport] = ev_ids

# =========================
# Extras: tarjetas/córners (opcional)
# =========================
def list_events_for_sport(sport_key: str) -> list[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/"
    r = request_with_retries(url, {"apiKey": api_key, "dateFormat": "iso"})
    if not r:
        return []
    try:
        return r.json()
    except Exception:
        return []

def list_event_markets(sport_key: str, event_id: str) -> list[str]:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/markets"
    r = request_with_retries(url, {"apiKey": api_key})
    if not r:
        return []
    try:
        return r.json()
    except Exception:
        return []

def fetch_event_odds_extras(sport_key: str, event_id: str, markets_to_fetch: list[str]) -> dict:
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "markets": ",".join(markets_to_fetch),
        "bookmakers": f"{PINNACLE_KEY},sport888",   # clave para evitar 422
        "regions": "eu",
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }
    if random.random() < 0.01:
        req = requests.Request("GET", url, params=params).prepare()
        print("DEBUG extras URL:", req.url[:300])

    r = request_with_retries(url, params)
    if not r:
        return {}
    try:
        return r.json()
    except Exception:
        return {}

if ENABLE_CARDS_CORNERS:
    for sport in sports:
        if not sport.startswith("soccer_"):
            continue

        cand_ids = set(event_ids_by_sport.get(sport, []))
        if not cand_ids:
            evs = list_events_for_sport(sport)
            now_utc = datetime.now(timezone.utc)
            for ev in evs:
                try:
                    dt = parse_iso(ev.get("commence_time"))
                except Exception:
                    continue
                if dt < now_utc or dt > now_utc + timedelta(days=EXTRA_EVENTS_MAX_DAYS):
                    continue
                if ev.get("id"):
                    cand_ids.add(ev["id"])

        if not cand_ids:
            continue

        print(f"➕ Extras tarjetas/córners en {sport}: {min(len(cand_ids), EXTRA_MAX_EVENTS_PER_SPORT)} eventos…")

        for ev_id in list(cand_ids)[:EXTRA_MAX_EVENTS_PER_SPORT]:
            mk_available = set(list_event_markets(sport, ev_id))
            if not mk_available:
                continue

            mk_to_fetch = sorted(m for m in EXTRA_MARKETS_RAW if m in mk_available)
            if not mk_to_fetch:
                continue

            data = fetch_event_odds_extras(sport, ev_id, mk_to_fetch)
            if not data:
                continue

            fecha_iso = data.get("commence_time")
            home_team = data.get("home_team")
            away_team = data.get("away_team")
            evento = f"{home_team} vs {away_team}"

            for bookmaker in data.get("bookmakers", []):
                casa = bookmaker.get("key")
                book_last_update = bookmaker.get("last_update")

                for market in bookmaker.get("markets", []):
                    mk_raw = market.get("key")
                    base, metric = classify_extra_market(mk_raw)
                    if base not in {"totals", "spreads"}:
                        continue

                    for outcome in market.get("outcomes", []):
                        nombre = outcome.get("name")
                        price  = outcome.get("price")
                        point  = outcome.get("point")

                        opt_std, opt_label = normalize_option(base, nombre, home_team, away_team)
                        if opt_std is None:
                            continue
                        point_norm = normalize_point(base, point)

                        rows.append({
                            "liga": sport,
                            "evento": evento,
                            "fecha": fecha_iso,
                            "event_id": ev_id,
                            "home_team": home_team,
                            "away_team": away_team,
                            "mercado": base,         # totals/spreads
                            "metric": metric,        # cards/corners
                            "opcion_raw": nombre,
                            "opcion_std": opt_std,
                            "opcion_lbl": opt_label,
                            "point": point,
                            "point_norm": point_norm,
                            "casa": casa,
                            "cuota": price,
                            "book_last_update": book_last_update
                        })

print("\n👀 Bookies detectadas:", ", ".join(sorted(bookies_seen)))
print("✅ Cuotas recolectadas. Procesando...\n")
if not rows:
    print("⚠️  No se recolectaron cuotas. Saliendo.")
    raise SystemExit(0)

df = pd.DataFrame(rows)
print(f"🔍 eventos únicos: {df[['liga','evento']].drop_duplicates().shape[0]}")
print(f"   filas totales: {len(df)} | Pinnacle: {(df['casa']==PINNACLE_KEY).sum()} | SOFTS: {(df['casa']!=PINNACLE_KEY).sum()}")

# =========================
# No-vig con Pinnacle (+ validación 3-way en soccer h2h)
# =========================
df_pin = df[df["casa"] == PINNACLE_KEY].copy()
if df_pin.empty:
    print("⚠️  No hay Pinnacle. No se puede calcular p_fair.")
    raise SystemExit(0)

gkeys = ["liga", "evento", "fecha", "mercado", "metric", "point_norm", "home_team", "away_team"]
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
# Softs: mejor cuota efectiva + consenso
# =========================
df_soft = df[df["casa"] != PINNACLE_KEY].copy()
if df_soft.empty:
    print("⚠️  No hay casas soft. Saliendo.")
    raise SystemExit(0)

df_soft["odds_eff"] = df_soft.apply(lambda r: effective_odds(r["casa"], r["cuota"]), axis=1)

def second_best(s):
    return s.nlargest(2).min() if len(s) >= 2 else float("nan")

keys_soft = gkeys + ["opcion_std"]
soft_stats = (
    df_soft.groupby(keys_soft, dropna=False)["odds_eff"]
           .agg(n_soft="size", median_soft="median", second_best=second_best)
           .reset_index()
)

soft_group = df_soft.groupby(keys_soft, dropna=False)
idx_max = soft_group["odds_eff"].idxmax()
best_soft = df_soft.loc[idx_max].copy().rename(columns={
    "point": "point_soft",
    "cuota": "odds_soft_raw",
    "odds_eff": "odds_soft",
    "book_last_update": "last_update_soft",
    "casa": "best_bookie"
})

best_soft = best_soft.merge(soft_stats, on=keys_soft, how="left")

# =========================
# Merge Pinnacle vs mejor soft
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
# EV, Kelly y umbral por cuota + métrica
# =========================
merged["EV"] = (merged["odds_soft"] * merged["p_fair"]) - 1.0

b = merged["odds_soft"] - 1.0
p = merged["p_fair"]
q = 1.0 - p
kelly_raw = (b * p - q) / b
kelly_raw = kelly_raw.where(merged["EV"] > 0, 0.0)
kelly_frac = (kelly_raw.clip(lower=0.0) * KELLY_FRACTION).fillna(0.0)
merged["stake_units"] = (kelly_frac * BANKROLL_UNITS).clip(upper=STAKE_MAX_UNITS).round(2)

merged["liga_pretty"] = merged["liga"].map(lambda x: LEAGUE_MAP.get(x, x))
merged["fecha_dt"] = merged["fecha"].apply(parse_iso)
merged["tti_h"] = (merged["fecha_dt"] - datetime.now(timezone.utc)).dt.total_seconds() / 3600.0
merged["mins_since_soft"] = merged["last_update_soft"].apply(lambda s: minutes_since(s) if pd.notna(s) else float("inf"))
merged["odds_diff"] = merged["odds_soft"] - merged["odds_pinnacle"]

def ev_min_for_row(mk: str, metric: str, odds_soft: float) -> float:
    base = EV_MIN_H2H if mk == "h2h" else EV_MIN_LINES
    base += EV_EXTRA_BY_METRIC.get(metric, 0.0)
    if pd.isna(odds_soft):
        return base
    if odds_soft <= 2.20:
        bump = 0.00
    elif odds_soft <= 3.00:
        bump = 0.01
    else:
        bump = 0.02
    return base + bump

merged["ev_min_req"] = merged.apply(lambda r: ev_min_for_row(r["mercado"], r["metric"], r["odds_soft"]), axis=1)
merged["odds_max_req"] = merged["mercado"].apply(lambda m: ODDS_MAX_H2H if m == "h2h" else ODDS_MAX_LINES)

merged["best_minus_second"] = merged["odds_soft"] - merged["second_best"]
merged["consensus_ratio"]  = merged["odds_soft"] / merged["median_soft"]

# =========================
# Filtros
# =========================
now_utc = datetime.now(timezone.utc)

ok_ev        = (merged["EV"] >= merged["ev_min_req"])
ok_pfair     = (merged["p_fair"] >= P_FAIR_MIN)
ok_odds_low  = (merged["odds_soft"] >= ODDS_MIN)
ok_odds_high = (merged["odds_soft"] <= merged["odds_max_req"])
ok_future    = (merged["fecha_dt"] >= now_utc)
ok_window    = (merged["fecha_dt"] <= now_utc + timedelta(days=EVENT_MAX_DAYS))

# Frescura
if FRESH_POLICY == "static":
    if FRESH_MAX_MIN and FRESH_MAX_MIN > 0:
        ok_fresh = (merged["mins_since_soft"] <= FRESH_MAX_MIN)
    else:
        ok_fresh = pd.Series(True, index=merged.index)
elif FRESH_POLICY == "dynamic":
    merged["fresh_limit_min"] = merged["tti_h"].apply(allowed_staleness_minutes)
    ok_fresh = (merged["mins_since_soft"] <= merged["fresh_limit_min"])
else:
    ok_fresh = pd.Series(True, index=merged.index)

# Delta mínimo soft vs Pinnacle
if MIN_DIFF_ODDS and MIN_DIFF_ODDS > 0:
    ok_delta = (merged["odds_diff"] >= MIN_DIFF_ODDS)
else:
    ok_delta = pd.Series(True, index=merged.index)

# Anti-outliers (consenso)
bad_consensus = (
    (merged["n_soft"] >= 3) &
    (
        (merged["best_minus_second"] > OUTLIER_DIFF_CAP) |
        (merged["consensus_ratio"]  > CONSENSUS_RATIO_CAP)
    ) &
    (merged["EV"] < merged["ev_min_req"] + EV_EXTRA_FOR_OUTLIER)
)
ok_consensus = ~bad_consensus

mask = ok_ev & ok_pfair & ok_odds_low & ok_odds_high & ok_future & ok_window & ok_fresh & ok_delta & ok_consensus
valuebets = merged[mask].copy().sort_values(by="EV", ascending=False)

print(f"   EV ≥ umbral tras filtros: {len(valuebets)} picks")

# Breakdown rápido (para entender por qué no pasan)
total = len(merged)
def show(label, ok_mask):
    print(f"   {label:<16}: {int(ok_mask.sum()):4d}/{total}")

print("   ── Breakdown filtros ──")
show("EV", ok_ev)
show("p_fair", ok_pfair)
show("odds_min", ok_odds_low)
show("odds_max", ok_odds_high)
show("fecha>=hoy", ok_future)
show("ventana_días", ok_window)
show("frescura", ok_fresh)
show("Δ min odds", ok_delta)
show("consenso", ok_consensus)

# =========================
# CSV
# =========================
def export_csv(df_out: pd.DataFrame, filename: str):
    if df_out.empty:
        top = merged.sort_values(by="EV", ascending=False).head(12).copy()
        top["fecha_local"] = top["fecha"].apply(to_local_str)
        top["updated_at"]  = datetime.now(TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S")
        cols = [
            "liga_pretty","evento","fecha_local","mercado","metric","point_norm",
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
    df_out["updated_at"]  = datetime.now(TZ_LOCAL).strftime("%Y-%m-%d %H:%M:%S")
    cols = [
        "liga_pretty","evento","fecha_local","mercado","metric","point_norm",
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
    mk = market_label(row["mercado"], metric=row.get("metric","goals"))

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
    metric_txt = metric_label(row.get("metric","goals"))

    parts = [
        f"🏆 {liga}",
        f"{evento} — {fecha}",
        f"🎯 {mk}{linea_txt} · {opcion_lbl}",
        f"🧭 Métrica: {metric_txt}",
        f"🔝 Mejor casa: {best_label}",
        f"🔢 Pinnacle: {odds_pin} | {best_label}: {odds_soft_raw} (efectiva {odds_soft_eff}) (Δ {diff})",
        f"📈 EV: {ev_pct} | Stake (Kelly {int(KELLY_FRACTION*100)}%): {stake} u",
        f"⏱️ Actualizado: {freshness}"
    ]
    if pd.notna(row.get("best_minus_second")) and pd.notna(row.get("consensus_ratio")):
        if (row.get("n_soft", 0) >= 3) and (
            (row["best_minus_second"] > OUTLIER_DIFF_CAP) or (row["consensus_ratio"] > CONSENSUS_RATIO_CAP)
        ):
            parts.append("⚠️ Posible outlier vs consenso")
    return "\n".join(parts)

def build_messages(df: pd.DataFrame):
    header = (
        f"🎯 {len(df)} apuestas con valor "
        f"(H2H ≥ {int(EV_MIN_H2H*100)}% · Líneas ≥ {int(EV_MIN_LINES*100)}% "
        f"· extra métricas +{int(EV_EXTRA_BY_METRIC.get('cards',0)*100)}pp):\n\n"
    )
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
        "🎯 0 apuestas con valor con los filtros actuales. (CSV incluye top para inspección)"
    ])
else:
    mensajes = build_messages(valuebets)
    send_telegram_messages(telegram_token, telegram_user_id, mensajes)

# =========================
# Resumen cuota API
# =========================
if API_HEADERS_SNAPSHOT:
    last = API_HEADERS_SNAPSHOT[-1]
    print("\n📊 Resumen cuota API (última respuesta):")
    print(f"   x-requests-remaining: {last.get('remaining')}")
    print(f"   x-requests-used:      {last.get('used')}")
else:
    print("\nℹ️ No se pudieron leer headers de cuota de la API.")