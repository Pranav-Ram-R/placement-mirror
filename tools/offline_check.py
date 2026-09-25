"""Log every network attempt while the app runs, to check that nothing leaves localhost.

serve      runs app.server with a Python audit hook (sys.addaudithook) that logs every
           socket.connect, socket.sendto, socket.bind, socket.getaddrinfo,
           socket.gethostbyname, socket.gethostbyaddr, http.client.connect and
           urllib.Request event to a JSON lines file. An attempt to reach an address
           that is not loopback is logged and then refused, as if the network were off.
summarize  reads that log and, optionally, a Chromium net log written by the browser
           (msedge --log-net-log=FILE), and lists every host each one tried to reach.
           Browser requests are split by initiator: requests the app pages started
           (initiator on loopback) and requests the browser started for its own
           services (update checks, sync and similar).

Usage:
  python tools/offline_check.py serve LOG.jsonl [app.server arguments]
  python tools/offline_check.py summarize LOG.jsonl [--net-log NETLOG.json]

For a check with the network really off, turn off Wi-Fi and unplug Ethernet before
starting. The logs record every attempt either way.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
EVENTS = {"socket.connect", "socket.sendto", "socket.bind", "socket.getaddrinfo", "socket.gethostbyname",
          "socket.gethostbyaddr", "http.client.connect", "urllib.Request"}
OUTBOUND = EVENTS - {"socket.bind"}
LOCAL_SCHEMES = {"data", "blob", "about", "chrome", "edge", "devtools", "chrome-extension", "extension"}


def is_loopback(host: str) -> bool:
    h = host.strip("[]").lower()
    if h in ("", "localhost", "localhost."):  # "" is getaddrinfo(None, ...) for a local bind
        return True
    try:
        return ipaddress.ip_address(h.split("%")[0]).is_loopback
    except ValueError:
        return False


def target(event: str, args: tuple) -> tuple[str, object]:
    if event in ("socket.connect", "socket.sendto", "socket.bind"):
        address = args[1]
        return (str(address[0]), address[1]) if isinstance(address, tuple) else (str(address), None)
    if event == "socket.getaddrinfo":
        return ("" if args[0] is None else str(args[0])), args[1]
    if event in ("socket.gethostbyname", "socket.gethostbyaddr"):
        return str(args[0]), None
    if event == "http.client.connect":
        return str(args[1]), args[2]
    url = urlsplit(args[0])  # urllib.Request
    return url.hostname or "", url.port


def install(log_path: Path) -> None:
    log = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115  open for the process lifetime
    lock = threading.Lock()

    def hook(event: str, args: tuple) -> None:
        if event not in EVENTS:
            return
        host, port = target(event, args)
        local = is_loopback(host)
        with lock:
            log.write(json.dumps({"t": round(time.time(), 3), "event": event, "host": host, "port": port,
                                  "loopback": local, "thread": threading.current_thread().name}, default=str) + "\n")
        if not local and event in OUTBOUND:
            raise OSError(f"offline check: refused {event} to {host}")

    sys.addaudithook(hook)


def serve(log_path: Path, server_args: list[str]) -> None:
    import runpy

    install(log_path)
    sys.path.insert(0, str(ROOT))
    sys.argv = ["app.server", *server_args]
    runpy.run_module("app.server", run_name="__main__", alter_sys=True)


def load_net_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:  # the browser was stopped before it closed the file
        return json.loads(text.rstrip().rstrip(",") + "]}")


def summarize_net_log(path: Path) -> dict:
    data = load_net_log(path)
    names = {v: k for k, v in data["constants"]["logEventTypes"].items()}
    urls, hosts, connects = Counter(), Counter(), Counter()
    starts, answered = [], set()  # (source id, url, initiator), sources that read response headers
    for ev in data.get("events", []):
        params = ev.get("params") or {}
        name = names.get(ev.get("type"), "")
        source = ev.get("source", {}).get("id")
        if isinstance(params.get("url"), str):
            urls[params["url"]] += 1
        if name == "URL_REQUEST_START_JOB" and isinstance(params.get("url"), str):
            starts.append((source, params["url"], str(params.get("initiator", ""))))
        if name == "HTTP_TRANSACTION_READ_RESPONSE_HEADERS":
            answered.add(source)
        if name.startswith("HOST_RESOLVER") and isinstance(params.get("host"), str):
            hosts[params["host"]] += 1
        if name in ("TCP_CONNECT_ATTEMPT", "UDP_CONNECT") and isinstance(params.get("address"), str):
            connects[params["address"]] += 1

    def outbound_url(u: str) -> bool:
        parts = urlsplit(u)
        return parts.scheme not in LOCAL_SCHEMES and not is_loopback(parts.hostname or "")

    def host_part(h: str) -> str:
        h = h.split("://")[-1]
        return h[1:h.index("]")] if h.startswith("[") else h.rsplit(":", 1)[0]

    def from_page(initiator: str) -> bool:
        return initiator.startswith(("http:", "https:")) and is_loopback(urlsplit(initiator).hostname or "")

    outbound = [(s, u, i) for s, u, i in starts if outbound_url(u)]
    return {
        "events": len(data.get("events", [])),
        "page_requests": len([1 for _, _, i in starts if from_page(i)]),
        "page_outbound": sorted({u for _, u, i in outbound if from_page(i)}),
        "browser_outbound": sorted({u for _, u, i in outbound if not from_page(i)}),
        "outbound_answered": sorted({u for s, u, _ in outbound if s in answered}),
        "urls": dict(urls), "outbound_urls": sorted(u for u in urls if outbound_url(u)),
        "resolved_hosts": dict(hosts), "outbound_hosts": sorted(h for h in hosts if not is_loopback(host_part(h))),
        "connects": dict(connects), "outbound_connects": sorted(c for c in connects if not is_loopback(host_part(c))),
    }


def summarize(log_path: Path, net_log: Path | None) -> dict:
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_kind = Counter((r["event"], r["host"], r["loopback"]) for r in rows)
    out = {"python": {"events": len(rows),
                      "attempts": [{"event": e, "host": h, "loopback": lb, "count": n}
                                   for (e, h, lb), n in sorted(by_kind.items())],
                      "outbound": [r for r in rows if not r["loopback"] and r["event"] in OUTBOUND]}}
    print(f"Python server process: {len(rows)} network events logged, "
          f"{len(out['python']['outbound'])} to a host that is not loopback")
    for a in out["python"]["attempts"]:
        print(f"  {a['count']:5d}  {a['event']:22s} {a['host'] or '(local bind)'}  "
              f"{'loopback' if a['loopback'] else 'NOT LOOPBACK, refused'}")
    if net_log:
        nl = summarize_net_log(net_log)
        out["browser"] = nl
        print(f"Browser net log: {nl['events']} events, {len(nl['urls'])} distinct URLs, "
              f"{len(nl['outbound_urls'])} not on loopback")
        print(f"  requests started by the app pages: {nl['page_requests']}, "
              f"not on loopback: {len(nl['page_outbound'])} {nl['page_outbound']}")
        print(f"  URLs not on loopback started by the browser itself: {len(nl['browser_outbound'])}, "
              f"answered: {len(nl['outbound_answered'])} {nl['outbound_answered']}")
        for u in sorted(nl["urls"]):
            print(f"  {'NOT LOOPBACK' if u in nl['outbound_urls'] else 'loopback    '}  {u}")
        for h in nl["outbound_hosts"]:
            print(f"  host lookup, not loopback: {h}")
        for c in nl["outbound_connects"]:
            print(f"  connect, not loopback: {c}")
    return out


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "serve":
        serve(Path(sys.argv[2]), sys.argv[3:])
        return 0
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["summarize"])
    ap.add_argument("log", type=Path)
    ap.add_argument("--net-log", type=Path)
    ap.add_argument("--out", type=Path, help="also write the summary as JSON")
    args = ap.parse_args()
    out = summarize(args.log, args.net_log)
    if args.out:
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
