import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def _add_routes(root: ET.Element) -> None:
    ET.SubElement(root, "route", id="route_ns", edges="n_t t_s")
    ET.SubElement(root, "route", id="route_we", edges="w_t t_e")


def _write_uniform(root: ET.Element, begin: int, end: int, ns: int, we: int) -> None:
    period_ns = max(0.1, 3600.0 / float(ns))
    period_we = max(0.1, 3600.0 / float(we))
    ET.SubElement(root, "flow", id="flow_ns_uniform", route="route_ns", begin=str(begin), end=str(end), period=f"{period_ns:.4f}", departSpeed="max", departPos="base", departLane="best")
    ET.SubElement(root, "flow", id="flow_we_uniform", route="route_we", begin=str(begin), end=str(end), period=f"{period_we:.4f}", departSpeed="max", departPos="base", departLane="best")


def _write_poisson(root: ET.Element, begin: int, end: int, ns: int, we: int) -> None:
    ET.SubElement(root, "flow", id="flow_ns_poisson", route="route_ns", begin=str(begin), end=str(end), vehsPerHour=str(ns), departSpeed="max", departPos="base", departLane="best")
    ET.SubElement(root, "flow", id="flow_we_poisson", route="route_we", begin=str(begin), end=str(end), vehsPerHour=str(we), departSpeed="max", departPos="base", departLane="best")


def _write_burst(root: ET.Element, begin: int, end: int, ns: int, we: int) -> None:
    span = max(3, end - begin)
    seg = span // 3

    flows: list[tuple[str, str, int, int, int]] = []

    def add_three(prefix: str, route_id: str, base_flow: int) -> None:
        low = max(1, int(base_flow * 0.5))
        high = max(1, int(base_flow * 1.8))
        flows.extend(
            [
                (f"{prefix}_1", route_id, begin, begin + seg, low),
                (f"{prefix}_2", route_id, begin + seg, begin + 2 * seg, high),
                (f"{prefix}_3", route_id, begin + 2 * seg, end, low),
            ]
        )

    add_three("flow_ns_burst", "route_ns", ns)
    add_three("flow_we_burst", "route_we", we)

    for flow_id, route_id, b, e, vph in sorted(flows, key=lambda item: (item[2], item[0])):
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


def write_route(out_path: Path, begin: int, end: int, ns: int, we: int, dist: str) -> None:
    root = ET.Element("routes")
    _add_routes(root)

    if dist == "uniform":
        _write_uniform(root, begin, end, ns, we)
    elif dist == "poisson":
        _write_poisson(root, begin, end, ns, we)
    elif dist == "burst":
        _write_burst(root, begin, end, ns, we)
    else:
        raise ValueError(f"Unknown dist: {dist}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def main() -> None:
    prs = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate single-intersection route files with arrival distributions (uniform/poisson/burst).",
    )
    prs.add_argument("--out", type=str, required=True, help="Output .rou.xml")
    prs.add_argument("--begin", type=int, default=0)
    prs.add_argument("--end", type=int, default=600)
    prs.add_argument("--ns", type=int, default=600)
    prs.add_argument("--we", type=int, default=600)
    prs.add_argument("--dist", type=str, choices=["uniform", "poisson", "burst"], required=True)
    args = prs.parse_args()

    write_route(Path(args.out), args.begin, args.end, args.ns, args.we, args.dist)
    print(f"Wrote route file: {args.out}")


if __name__ == "__main__":
    main()
