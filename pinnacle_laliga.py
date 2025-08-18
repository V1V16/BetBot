# laliga_pinnacle_to_telegram.py
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from dotenv import load_dotenv

# ========= Config =========
SPORT_KEY = "soccer_spain_la_liga"   # Primera División. Para Segunda: "soccer_spain_segunda_division"
MARKETS = ["h2h", "totals", "spreads"]
REGIONS = "eu,uk"                     # amplía cobertura del feed (no afecta a Pinnacle)
INCLUDE_TOTALS = True
INCLUDE_SPREADS = True
N_TOTALS_LINES = 2                    # nº de líneas Over/Under a mostrar (cercanas a 2.5)
N_SPREADS_LINES = 2                   # nº de líneas de hándicap a mostrar (cercanas a 0)
TOTALS_TARGET = 2.50
TELEGRAM_CHUNK = 3500                 # < 4096 por límite de Telegram
TIMEOUT_S = 12
TZ_LOCAL = ZoneInfo("Europe/Madrid")

# ========= Utilidades =========
def to_local_str(iso_str: str) -> str:
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(TZ_LOCAL).strftime("%d/%m/%Y %H:%M")

def fmt_odds(x, nd=2):
    try:
        return f"{float(x):.{nd}f}".replace(".", ",")
    except Exception:
        return "-"

def pick_totals_lines(totals_dict, target=2.5, n=2):
    """totals_dict: {line -> {'OVER': odd, 'UNDER': odd}}"""
    if not totals_dict:
        return []
    df = pd.DataFrame(
        [{"line": float(k), "has_over": 'OVER' in v, "has_under": 'UNDER' in v} | v
         for k, v in totals_dict.items()]
    )
    df["dist"] = (df["line"] - target).abs()
    df = df.sort_values(["dist", "line"]).head(n)
    out = []
    for _, r in df.iterrows():
        out.append((r["line"], r.get("OVER", None), r.get("UNDER", None)))
    return out

def pick_spreads_lines(spreads_dict, n=2):
    """spreads_dict: {abs_line -> {'HOME': (point,odd), 'AWAY': (point,odd)}}"""
    if not spreads_dict:
        return []
    df = pd.DataFrame([{"abs_line": float(k), **{k2: v2 for k2, v2 in v.items()}}
                       for k, v in spreads_dict.items()])
    df = df.sort_values(["abs_line"]).head(n)
    out = []
    for _, r in df.iterrows():
        home = r.get("HOME", (None, None))
        away = r.get("AWAY", (None, None))
        out.append((home, away))  # ((point_home, odd_home), (point_away, odd_away))
    return out

def send_telegram(token: str, chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    return requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=TIMEOUT_S)

# ========= Cargar .env =========
load_dotenv()
API_KEY = os.getenv("ODDS_API_KEY")
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID")
if not all([API_KEY, TG_TOKEN, TG_CHAT]):
    raise SystemExit("❌ Faltan ODDS_API_KEY / TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID en .env")

# ========= Llamada API =========
url = f"https://api.the-odds-api.com/v4/sports/{SPORT_KEY}/odds/"
params = {
    "apiKey": API_KEY,
    "regions": REGIONS,
    "markets": ",".join(MARKETS),
    "oddsFormat": "decimal",
    "dateFormat": "iso",
}
print(f"📦 Descargando cuotas Pinnacle para {SPORT_KEY}…")
r = requests.get(url, params=params, timeout=TIMEOUT_S)
if r.status_code != 200:
    print("Headers:", dict(r.headers))
    raise SystemExit(f"❌ Error HTTP {r.status_code}: {r.text[:200]}")

data = r.json()

# ========= Procesar solo Pinnacle =========
cards = []  # cada elemento es un bloque de texto de un partido
count_events = 0
count_with_pinn = 0

for match in data:
    count_events += 1
    home = match.get("home_team")
    away = match.get("away_team")
    fecha = match.get("commence_time")
    evento = f"{home} vs {away}"

    # localizar Pinnacle
    pinn = None
    for b in match.get("bookmakers", []):
        if b.get("key") == "pinnacle":
            pinn = b
            break
    if not pinn:
        continue
    count_with_pinn += 1

    h2h = {"HOME": None, "DRAW": None, "AWAY": None}
    totals = {}   # line -> {"OVER": odd, "UNDER": odd}
    spreads = {}  # abs_line -> {"HOME": (point,odd), "AWAY": (point,odd)}

    for mkt in pinn.get("markets", []):
        mk = mkt.get("key")
        if mk not in MARKETS:
            continue
        for o in mkt.get("outcomes", []):
            name = (o.get("name") or "").strip().lower()
            price = o.get("price")
            point = o.get("point")

            if mk == "h2h":
                if name in {"draw", "empate", "x"}:
                    h2h["DRAW"] = price
                elif name in {"home", "local"} or (home and home.lower() in name):
                    h2h["HOME"] = price
                elif name in {"away", "visitante"} or (away and away.lower() in name):
                    h2h["AWAY"] = price

            elif mk == "totals" and INCLUDE_TOTALS:
                try:
                    ln = round(float(point), 2)
                except Exception:
                    continue
                side = "OVER" if name.startswith("over") or name == "over" else ("UNDER" if name.startswith("under") or name == "under" else None)
                if not side:
                    continue
                totals.setdefault(ln, {})
                totals[ln][side] = price

            elif mk == "spreads" and INCLUDE_SPREADS:
                try:
                    p = float(point)
                except Exception:
                    continue
                abs_ln = round(abs(p), 2)
                side = "HOME" if (name in {"home", "local"} or (home and home.lower() in name)) else \
                       ("AWAY" if (name in {"away", "visitante"} or (away and away.lower() in name)) else None)
                if not side:
                    continue
                spreads.setdefault(abs_ln, {})
                spreads[abs_ln][side] = (p, price)

    # Si no hay nada válido, saltar
    if not any([h2h["HOME"], h2h["DRAW"], h2h["AWAY"], totals, spreads]):
        continue

    # Construir bloque de texto
    lines = [
        f"🏆 LaLiga (ESP)",
        f"{home} vs {away} — {to_local_str(fecha)}",
    ]

    if any(h2h.values()):
        lines.append(f"🎯 1X2  · Local {fmt_odds(h2h['HOME'])} | Empate {fmt_odds(h2h['DRAW'])} | Visitante {fmt_odds(h2h['AWAY'])}")

    if INCLUDE_SPREADS and spreads:
        sel = pick_spreads_lines(spreads, n=N_SPREADS_LINES)
        if sel:
            lines.append("📏 Hándicap")
            for (ph, oh), (pa, oa) in sel:
                sh = f"{ph:+.2f}".replace(".", ",") if ph is not None else "—"
                sa = f"{pa:+.2f}".replace(".", ",") if pa is not None else "—"
                lines.append(f"   Home {sh} @ {fmt_odds(oh)}  |  Away {sa} @ {fmt_odds(oa)}")

    if INCLUDE_TOTALS and totals:
        sel = pick_totals_lines(totals, target=TOTALS_TARGET, n=N_TOTALS_LINES)
        if sel:
            lines.append("⚖️ Totales")
            for ln, o, u in sel:
                lines.append(f"   Over {str(ln).replace('.', ',')} @ {fmt_odds(o)}  |  Under {str(ln).replace('.', ',')} @ {fmt_odds(u)}")

    lines.append("─" * 18)
    cards.append("\n".join(lines))

# ========= Envío a Telegram =========
if not cards:
    body = "ℹ️ LaLiga: ahora mismo Pinnacle no tiene cuotas activas o el feed no trae mercados para mostrar."
else:
    header = f"🧭 Pinnacle · LaLiga — {count_with_pinn}/{count_events} partidos con cuotas\n\n"
    body = header + "\n".join(cards)

# troceo por límite de Telegram
chunks = []
current = ""
for part in body.split("\n"):
    if len(current) + len(part) + 1 > TELEGRAM_CHUNK:
        chunks.append(current.rstrip())
        current = part + "\n"
    else:
        current += part + "\n"
if current.strip():
    chunks.append(current.rstrip())

ok = True
for i, ch in enumerate(chunks, 1):
    resp = send_telegram(TG_TOKEN, TG_CHAT, ch)
    if resp.status_code != 200:
        print(f"❌ Telegram {i}/{len(chunks)}: {resp.status_code}: {resp.text[:200]}")
        ok = False
    else:
        print(f"📬 Enviado a Telegram ({i}/{len(chunks)})")

print(f"✅ Hecho. Partidos en feed: {count_events} | con Pinnacle: {count_with_pinn}")
if not ok:
    raise SystemExit(1)