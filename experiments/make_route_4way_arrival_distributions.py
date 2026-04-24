#!/usr/bin/env python3
"""Generate 4-way turn routes for flow x arrival-distribution matrix."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


ROUTES = {
    "route_n_left": "n_t t_e",
    "route_n_straight": "n_t t_s",
    "route_n_right": "n_t t_w",
    "route_e_left": "e_t t_s",
    "route_e_straight": "e_t t_w",
    "route_e_right": "e_t t_n",
    "route_s_left": "s_t t_w",
    "route_s_straight": "s_t t_n",
    "route_s_right": "s_t t_e",
    "route_w_left": "w_t t_n",
    "route_w_straight": "w_t t_e",
    "route_w_right": "w_t t_s",
}

# Left / straight / right split per incoming direction.
TURN_WEIGHTS = {"left": 0.2, "straight": 0.6, "right": 0.2}


def _add_routes(root: ET.Element) -> None:
    for rid, edges in ROUTES.items():
        ET.SubElement(root, "route", id=rid, edges=edges)


def _flows_for_direction(flow: int) -> dict[str, int]:
    vals = {
        turn: max(1, int(round(flow * ratio))) for turn, ratio in TURN_WEIGHTS.items()
    }
    # Adjust rounding drift to keep exact per-direction total.
    drift = int(flow - sum(vals.values()))
    if drift != 0:
        vals["straight"] = max(1, vals["straight"] + drift)
    return vals


def _write_uniform(root: ET.Element, begin: int, end: int, flow: int) -> None:
    for d in ["n", "e", "s", "w"]:
        per_turn = _flows_for_direction(flow)
        for turn in ["left", "straight", "right"]:
            vph = per_turn[turn]
            period = max(0.1, 3600.0 / float(vph))
            ET.SubElement(
                root,
                "flow",
                id=f"flow_{d}_{turn}_uniform",
                route=f"route_{d}_{turn}",
                begin=str(begin),
                end=str(end),
                period=f"{period:.4f}",
                departSpeed="max",
                departPos="base",
                departLane="best",
            )


def _write_poisson(root: ET.Element, begin: int, end: int, flow: int) -> None:
    for d in ["n", "e", "s", "w"]:
        per_turn = _flows_for_direction(flow)
        for turn in ["left", "straight", "right"]:
            ET.SubElement(
                root,
                "flow",
                id=f"flow_{d}_{turn}_poisson",
                route=f"route_{d}_{turn}",
                begin=str(begin),
                end=str(end),
                vehsPerHour=str(per_turn[turn]),
                departSpeed="max",
                departPos="base",
                departLane="best",
            )


def _write_burst(root: ET.Element, begin: int, end: int, flow: int) -> None:
    span = max(3, end - begin)
    seg = span // 3

    # SUMO requires flows to be globally sorted by departure time.
    # Collect first, then write in chronological order across all directions/turns.
    flow_rows: list[tuple[int, int, str, str, int]] = []

    for d in ["n", "e", "s", "w"]:
        per_turn = _flows_for_direction(flow)
        for turn in ["left", "straight", "right"]:
            base = per_turn[turn]
            low = max(1, int(round(base * 0.5)))
            high = max(1, int(round(base * 1.8)))
            pieces = [
                (begin, begin + seg, low),
                (begin + seg, begin + 2 * seg, high),
                (begin + 2 * seg, end, low),
            ]
            for idx, (b, e, vph) in enumerate(pieces, start=1):
                flow_rows.append((b, e, f"flow_{d}_{turn}_burst_{idx}", f"route_{d}_{turn}", vph))

    for b, e, flow_id, route_id, vph in sorted(flow_rows, key=lambda x: (x[0], x[1], x[2])):
        ET.SubElement(
            root,
            "flow",
            id=flow_id,
            route=route_id,
            begin=str(b),
            end=str(e),
            vehsPerHour=str(vph),
            departSpeed="max",
            departPos="base",
            departLane="best",
        )


def write_route(out_path: Path, begin: int, end: int, flow: int, dist: str) -> None:
    root = ET.Element("routes")
    _add_routes(root)

    if dist == "uniform":
        _write_uniform(root, begin, end, flow)
    elif dist == "poisson":
        _write_poisson(root, begin, end, flow)
    elif dist == "burst":
        _write_burst(root, begin, end, flow)
    else:
        raise ValueError(f"Unknown dist: {dist}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate a 4-way route file with uniform/poisson/burst arrivals.",
    )
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--begin", type=int, default=0)
    p.add_argument("--end", type=int, default=600)
    p.add_argument("--flow", type=int, default=600, help="Per incoming direction flow (veh/h).")
    p.add_argument("--dist", type=str, choices=["uniform", "poisson", "burst"], required=True)
    args = p.parse_args()

    write_route(Path(args.out), args.begin, args.end, args.flow, args.dist)
    print(f"Wrote route file: {args.out}")


if __name__ == "__main__":
    main()
